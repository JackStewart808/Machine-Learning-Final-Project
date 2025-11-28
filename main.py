import pygame
import numpy as np
from copy import deepcopy

WIDTHPX, HEIGHTPX = 760, 560
WIN = pygame.display.set_mode((WIDTHPX, HEIGHTPX))
pygame.display.set_caption("Machine Learning Final Project")

FPS = 60

BLACK = (0, 0, 0)
GREEN = (50, 100, 50)


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

def draw(board, lastMousePos):
    mouseStatus = pygame.mouse.get_pressed()
    if mouseStatus[0] or mouseStatus[2]:
        mouseX, mouseY = pygame.mouse.get_pos()
        selectedX = mouseX // 20
        selectedY = mouseY // 20

        if 0 <= selectedX < 28 and 0 <= selectedY < 28:
            if lastMousePos is None:
                lastMousePos = (selectedX, selectedY)

            # Draw along the line between lastMousePos and current pos
            lx, ly = lastMousePos

            dx = selectedX - lx
            dy = selectedY - ly
            steps = max(abs(dx), abs(dy))
            if steps != 0:
                for i in range(steps + 1):
                    t = i / steps
                    x = int(lx + dx * t)
                    y = int(ly + dy * t)
                    if 0 <= x < 28 and 0 <= y < 28:
                        board.setValue(x, y, 255 if mouseStatus[0] else 0)
            else:
                board.setValue(selectedX, selectedY, 255 if mouseStatus[0] else 0)
            lastMousePos = (selectedX, selectedY)
    else:
        lastMousePos = None   # Reset when mouse not held
    return lastMousePos

def updateDisplay(WIN, board: Drawing):
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
    
    pygame.display.update()


def main():
    board = Drawing()
    boardHistory = BoardHistory(board)
    lastMousePos = None
    Clock = pygame.time.Clock()
    run = True
    while run:
        Clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                newBoard = deepcopy(board)
                boardHistory.setNextBoard(newBoard)
                board = newBoard

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

        updateDisplay(WIN, board)


if __name__ == "__main__":
    main()
