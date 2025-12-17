import pygame
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Drawing:
    def __init__(self, prev=None):
        self.prev = prev
        self.next = None
        self.pixels = [[0 for _ in range(28)] for _ in range(28)]

    def getColor(self, x, y):
        value = self.pixels[y][x]
        return (value, value, value)

    def getValue(self, x, y):
        return self.pixels[y][x]

    def setValue(self, x, y, value):
        self.pixels[y][x] = value

    def testAgainstNeuralNetworkModel(self, model):
        img = np.array(self.pixels, dtype=np.float32)

        # Normalize to [0,1]
        img /= 255.0

        # Invert colors (MNIST convention)
        img = 1.0 - img

        # Add batch dimension: (1, 28, 28)
        img = img.reshape(1, 28, 28)

        prediction = model.predict(img, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return digit, confidence
    
    # Function to get guess and confidence from a Drawing object
    def testAgainstSVM(self, classifier, scaler):
        # Convert to 1x784 vector
        img = np.array(self.pixels, dtype=np.float32).reshape(1, -1) / 255.0
        img = scaler.transform(img)
        
        guess = classifier.predict(img)
        return guess.item()
    
    def testAgainstConvNetModel(self, model):
        # Convert to tensor and reshape to (1,28,28,1)
        img = np.array(self.pixels, dtype=np.float32) / 255.0
        img = 1.0 - img  # invert colors
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img, verbose=0)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return digit, confidence

class BoardHistory:
    def __init__(self, board: Drawing):
        self.currentBoard = board

    def getPrevBoard(self):
        if self.currentBoard.prev is not None:
            self.currentBoard = self.currentBoard.prev
        return self.currentBoard

    def getNextBoard(self):
        if self.currentBoard.next is not None:
            self.currentBoard = self.currentBoard.next
        return self.currentBoard

    def setNextBoard(self, board: Drawing):
        if self.currentBoard.next is not None:
            self._deleteChain(self.currentBoard.next)

        board.prev = self.currentBoard
        board.next = None
        self.currentBoard.next = board
        self.currentBoard = board

    def _deleteChain(self, node):
        while node is not None:
            nxt = node.next
            node.prev = None
            node.next = None
            node = nxt


def prepareNeuralNetworkModel():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=3)
    return model

# Prepare MNIST dataset for SVM
def prepareSVMModel():
    print("Prepare SVM Training")
    print("Training. . .")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten 28x28 images to 784
    x_train_flat = x_train.reshape(len(x_train), -1).astype(np.float32)
    x_test_flat = x_test.reshape(len(x_test), -1).astype(np.float32)

    # Normalize
    x_train_flat /= 255.0
    x_test_flat /= 255.0

    #Use a smaller subset to speed up training
    x_train_flat = x_train_flat[:10000]
    y_train = y_train[:10000]

    # Standardize features
    scaler = StandardScaler()
    x_train_flat = scaler.fit_transform(x_train_flat)
    x_test_flat = scaler.transform(x_test_flat)

    # Train multi-class SVM
    classifier = SVC(kernel='rbf')
    classifier.fit(x_train_flat, y_train)
    print("Training Complete.")
    return classifier, scaler

def prepareConvNetModel():
    print("Preparing CNN Model...")
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize to [0,1] and reshape for Conv2D input (28,28,1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

    x_train = x_train[:15000]
    y_train = y_train[:15000] 

    # Define the ConvNet architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # Compile model
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # Train the model
    model.fit(x_train, y_train, epochs=3, batch_size=64)
    print("CNN training complete.")
    return model
    
def draw(board, lastMousePos, brush_radius=.5):
    mouse_pressed = pygame.mouse.get_pressed()
    if not mouse_pressed[0]:
        return None  # Reset when not drawing

    mouse_x, mouse_y = pygame.mouse.get_pos()
    grid_x = mouse_x // 20
    grid_y = mouse_y // 20

    if lastMousePos is None:
        lastMousePos = (grid_x, grid_y)

    lx, ly = lastMousePos
    dx = grid_x - lx
    dy = grid_y - ly
    steps = max(abs(dx), abs(dy))
    for i in range(steps + 1):
        t = i / steps if steps != 0 else 0
        x = int(lx + dx * t)
        y = int(ly + dy * t)

        radius_squares = max(1, int(brush_radius * 2))
        for rx in range(-radius_squares, radius_squares + 1):
            for ry in range(-radius_squares, radius_squares + 1):
                nx = x + rx
                ny = y + ry
                if 0 <= nx < 28 and 0 <= ny < 28:
                    if (rx ** 2 + ry ** 2) ** 0.5 <= brush_radius * 2:
                        board.setValue(nx, ny, 255)

    return (grid_x, grid_y)

def updateDisplay(WIN, board: Drawing, Nguess, Nconfidence, Sguess, Cguess, Cconfidence):
    #Control Space Background
    WIN.fill(GREEN)
    #Primary Drawing Cells
    for x in range(28):
        for y in range(28):
            rect = pygame.Rect(x * 20, y * 20, 20, 20)
            pygame.draw.rect(WIN, board.getColor(x, y), rect)
    
    #Grid Markers
    for x in range(28):
        pygame.draw.rect(WIN, (40, 40, 100), pygame.Rect(x * 20, 0, 1, 560))
        pygame.draw.rect(WIN, (40, 40, 100), pygame.Rect(0, x * 20, 560, 1))
            
    #Lightgreen Barrier between drawing space and control space
    pygame.draw.rect(WIN, (70, 120, 70), pygame.Rect(560, 0, 10, 560))
    
    # Neural Network Prediction Display 
    if Nguess is not None:
        font_title = pygame.font.SysFont("Arial", 20)
        font_digit = pygame.font.SysFont("Arial", 100, bold=True)

        title_surf = font_title.render("Neural Network", True, (255, 255, 255))
        WIN.blit(title_surf, (575, 0))

        digit_surf = font_digit.render(str(Nguess), True, (255, 255, 255))
        WIN.blit(digit_surf, (575, 40))

        if Nconfidence is not None:
            font_conf = pygame.font.SysFont("Arial", 18)
            conf_surf = font_conf.render(f"Confidence: {Nconfidence:.2f}", True, (255, 255, 255))
            WIN.blit(conf_surf, (575, 160))

    # SVM Prediction Display 
    if Sguess is not None:
        font_title = pygame.font.SysFont("Arial", 20)
        font_digit = pygame.font.SysFont("Arial", 100, bold=True)

        title_surf = font_title.render("SVM", True, (255, 255, 255))
        WIN.blit(title_surf, (575, 200))

        digit_surf = font_digit.render(str(Sguess), True, (255, 255, 255))
        WIN.blit(digit_surf, (575, 240))

    # Convolutional Network Prediction Display 
    if Cguess is not None:
        font_title = pygame.font.SysFont("Arial", 20)
        font_digit = pygame.font.SysFont("Arial", 100, bold=True)

        title_surf = font_title.render("Convolutional Net", True, (255, 255, 255))
        WIN.blit(title_surf, (575, 360))

        digit_surf = font_digit.render(str(Cguess), True, (255, 255, 255))
        WIN.blit(digit_surf, (575, 400))

        if Cconfidence is not None:
            font_conf = pygame.font.SysFont("Arial", 18)
            conf_surf = font_conf.render(f"Confidence: {Cconfidence:.2f}", True, (255, 255, 255))
            WIN.blit(conf_surf, (575, 520))

    pygame.display.update()

neuralModel = prepareNeuralNetworkModel()
classifier, scalar = prepareSVMModel()
convModel = prepareConvNetModel()

pygame.init()
WIDTHPX, HEIGHTPX = 760, 560
WIN = pygame.display.set_mode((WIDTHPX, HEIGHTPX))
pygame.display.set_caption("Machine Learning Final Project")


BLACK = (0, 0, 0)
GREEN = (50, 100, 50)

def main():
    board = Drawing()
    boardHistory = BoardHistory(board)
    lastMousePos = None
    run = True
    Nguess = Nconfidence = Sguess = Cguess = Cconfidence = None
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                newBoard = deepcopy(board)
                boardHistory.setNextBoard(newBoard)
                board = newBoard

            if event.type == pygame.MOUSEBUTTONUP:
                Nguess, Nconfidence = board.testAgainstNeuralNetworkModel(neuralModel)
                Sguess = board.testAgainstSVM(classifier, scalar)
                Cguess, Cconfidence = board.testAgainstConvNetModel(convModel)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    board = boardHistory.getPrevBoard()
                if event.key == pygame.K_y:
                    board = boardHistory.getNextBoard()
                if event.key == pygame.K_r:
                    board = Drawing()
                    boardHistory = BoardHistory(board)
        
        #Handle Drawing on the Picture
        lastMousePos = draw(board, lastMousePos)

        updateDisplay(WIN, board, Nguess, Nconfidence, Sguess, Cguess, Cconfidence)


if __name__ == "__main__":
    main()
