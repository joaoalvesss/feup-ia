# game/board.py

import numpy as np
from time import time

class Board:
    """
    A class to represent the game board.

    Attributes:
    rows (int): Number of rows on the board.
    cols (int): Number of columns on the board.
    board (list): A 2D list representing the board state.
    move_history (list): A list recording all moves made.
    current_pos (tuple): The current position on the board as a (row, col) tuple.

    Methods:
    __init__(self, rows, cols, pattern, init_pos): Initializes the board with rows, cols, pattern, and initial position.
    __str__(self): Returns a string representation of the board state.
    move(self, src, dest): Moves a piece from src to dest on the board if valid.
    is_valid_position(self, row, col): Checks if the position (row, col) is a valid move.
    get_next_color(self, current_color): Returns the next color in sequence.
    matches(self, other_board): Checks if the current board matches another board.
    match_degrees(self, other_board): Calculates the degree of mismatch with another board.
    child_boards(self): Generates all possible child boards from the current board state.
    print_board(self): Prints the board to the console.
    equals(self, other_board): Checks if the current board state is equal to another board state.
    copy(self): Creates a deep copy of the board.
    move_dir(self, move): Executes a move in a given direction if it's legal.
    legal_move(self, move): Checks if a move is legal based on the game rules.
    """
    
    def __init__(self, rows, cols, pattern, init_pos):
        """
        Constructs all the necessary attributes for the board object.

        Parameters:
        rows (int): Number of rows on the board.
        cols (int): Number of columns on the board.
        pattern (list): A 2D list representing the initial board state.
        init_pos (tuple): The starting position on the board as a (row, col) tuple.
        """
        self.rows = rows
        self.cols = cols
        self.board = []
        self.move_history = []
        for r in range(rows):
            row = []
            for c in range(cols):
                if r < len(pattern) and c < len(pattern[r]):
                    row.append(pattern[r][c])
                else:
                    row.append(' ') 
            self.board.append(row)
        self.current_pos = init_pos
                    
    def __str__(self):
        """
        Returns a string representation of the board, with each row on a new line and each value separated by a space.
        """
        return '\n'.join([' '.join(row) for row in self.board])

    def move(self, src, dest):
        """
        Moves a piece from the source position to the destination position if it is a legal move.
        It updates the board state and move history accordingly.

        Parameters:
        src (tuple): The source position as (row, col).
        dest (tuple): The destination position as (row, col).
        """
        if self.is_valid_position(dest[0], dest[1]):
            self.current_pos = (dest[1], dest[0])
            src_color = self.board[src[0]][src[1]]
            dest_color = self.board[dest[0]][dest[1]]
            self.move_history.append((src, dest))
            
            if src_color == dest_color:
                return
            if src_color == 'R' and dest_color == 'B':
                self.board[dest[0]][dest[1]] = 'Y'
            elif src_color == 'B' and dest_color == 'R':
                self.board[dest[0]][dest[1]] = 'Y'
            elif src_color == 'R' and dest_color == 'Y':
                self.board[dest[0]][dest[1]] = 'B'
            elif src_color == 'Y' and dest_color == 'R':
                self.board[dest[0]][dest[1]] = 'B'
            elif src_color == 'B' and dest_color == 'Y':
                self.board[dest[0]][dest[1]] = 'R'
            elif src_color == 'Y' and dest_color == 'B':
                self.board[dest[0]][dest[1]] = 'R'

    def is_valid_position(self, row, col):
        """
        Determines if a position is valid and can be moved to on the board.

        Parameters:
        row (int): The row index of the position.
        col (int): The column index of the position.

        Returns:
        bool: True if the position is within the bounds of the board and not an empty space (' '), False otherwise.
        """
        a = (0 <= row < self.rows and 0 <= col < self.cols and self.board[row][col] != ' ')
        return a

    def get_next_color(self, current_color):
        """
        Gets the next color in the sequence following the game's rules.

        Parameters:
        current_color (str): The current color ('R', 'B', 'Y').

        Returns:
        str: The next color in the sequence.
        """
        color_order = ['R', 'B', 'Y']
        next_index = (color_order.index(current_color) + 1) % len(color_order)
        return color_order[next_index]

    def matches(self, other_board):
        """
        Checks if the current board state matches the other board state.

        Parameters:
        other_board (Board): Another Board instance to compare against.

        Returns:
        bool: True if all corresponding positions match, False otherwise.
        """
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] != other_board.board[row][col]:
                    return False
        return True
    
    def match_degrees(self, other_board):
        """
        Calculates the number of differing positions between the current board and the other board.

        Parameters:
        other_board (Board): Another Board instance to compare against.

        Returns:
        int: The total number of positions where the two boards differ.
        """
        degree = 0
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] != other_board.board[row][col]:
                    degree += 1
        return degree
    
    def child_boards(self):
        """
        Generates all possible immediate successor board states from the current state.

        Returns:
        list: A list of tuples containing child Board instances and the corresponding move direction.
        """
        moves = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
        directions = np.array(['down', 'up', 'right', 'left'])
        child_boards = []
        for move, direction in zip(moves, directions):
            new_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
            if self.is_valid_position(new_pos[1], new_pos[0]):
                new_board = np.copy(self.board)
                child_board = Board(self.rows, self.cols, new_board, new_pos)
                child_board.move((self.current_pos[1], self.current_pos[0]), (new_pos[1], new_pos[0]))
                child_boards.append((child_board, direction))
        return child_boards

    def print_board(self):
        """
        Prints a visual representation of the board to the console.
        """
        for i in self.board:
            print(i)
        print(self.current_pos)
        print('\n')

    def equals(self, other_board):
        """
        Checks if the current board state is exactly equal to another board state, including the current position.

        Parameters:
        other_board (Board): Another Board instance to compare against.

        Returns:
        bool: True if both board states and the current positions are equal, False otherwise.
        """
        if self.matches(other_board) and self.current_pos == other_board.current_pos:
            return True
        return False
    
    def copy(self):
        """
        Creates a deep copy of the current board.

        Returns:
        Board: A new Board instance with the same state as the current board.
        """
        b = Board(self.rows, self.cols, self.board, self.current_pos)
        return b
    
    def move_dir(self, move):
        """
        Executes a move in a specified direction if it is legal.

        Parameters:
        move (str): The direction to move ('up', 'down', 'left', 'right').

        Side effects:
        Updates the board state and current position if the move is valid.
        """
        src = (self.current_pos[1], self.current_pos[0])
        
        if move == 'down':
            dest = (self.current_pos[0], self.current_pos[1] + 1)
        elif move == 'up':
            dest = (self.current_pos[0], self.current_pos[1] - 1)
        elif move == 'left':
            dest = (self.current_pos[0] - 1, self.current_pos[1])
        elif move == 'right':
            dest = (self.current_pos[0] + 1, self.current_pos[1])
        else: 
            return
        
        dest = (dest[1], dest[0])

        self.move(src, dest)

    def legal_move(self, move):
        """
        Determines if a move in a given direction is legal based on the current state of the board.

        Parameters:
        move (str): The direction to move ('up', 'down', 'left', 'right').

        Returns:
        bool: True if the move is legal, False otherwise.
        """
        src = (self.current_pos[1], self.current_pos[0])
        
        if move == 'down':
            dest = (self.current_pos[0], self.current_pos[1] + 1)
        elif move == 'up':
            dest = (self.current_pos[0], self.current_pos[1] - 1)
        elif move == 'left':
            dest = (self.current_pos[0] - 1, self.current_pos[1])
        elif move == 'right':
            dest = (self.current_pos[0] + 1, self.current_pos[1])
        else: 
            return

        return self.is_valid_position(dest[1], dest[0])
