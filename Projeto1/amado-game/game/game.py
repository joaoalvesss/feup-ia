# game/game.py

import random
import pygame
from .board import Board
from .solver import *
from .hint import *

class Game:
    """
    A Game class encapsulating the state and behavior of the puzzle game.

    Attributes:
        precision_mode (str): Mode of precision, affecting the difficulty of the puzzle.
        pattern (list): The current pattern of colors on the game board.
        init_pos (tuple): The initial position of the cursor on the game board.
        goal_pattern (list): The target pattern to reach to solve the puzzle.
        board (Board): The current state of the game board.
        goal_board (Board): The state of the goal board to be achieved.
        cursor_row (int): The row index of the cursor's current position.
        cursor_col (int): The column index of the cursor's current position.
        move_count (int): The number of moves made by the player.
        greedy_visited (list): A list to keep track of visited states for greedy hint.
        mt_visited (list): A list to keep track of visited states for Monte Carlo hint.
        solution (list): The solution path for the puzzle.
        moved (bool): A flag to check if a move has been made.
    """
    def __init__(self, precision_mode):
        """
        Initializes a Game with a specific precision mode.

        Parameters:
            precision_mode (str): The precision mode to set for the game ('high' or 'low').
        """
        self.precision_mode = precision_mode
        generated_pattern = self.generate_symmetric_pattern(10)
        self.pattern = generated_pattern[0]
        self.init_pos = generated_pattern[1]
        self.goal_pattern = self.shuffle_colors_for_goal(self.pattern)
        rows = len(self.pattern)
        cols = len(self.pattern[0])
        self.board = Board(rows, cols, self.pattern, self.init_pos)
        self.goal_board = Board(rows, cols, self.goal_pattern, (0, 0))
        self.cursor_row = 0 
        self.cursor_col = 0
        self.move_count = 0
        while self.pattern[self.cursor_row][0] == ' ':
            self.cursor_row += 1
        self.greedy_visited = []
        self.mt_visited = []
        self.solution = []
        self.moved = False

    def generate_symmetric_pattern(self, max_rows):
        """
        Generates a symmetric pattern for the initial state of the game board.

        Parameters:
            max_rows (int): Maximum number of rows for the pattern.

        Returns:
            tuple: A tuple containing the pattern (list of lists) and the initial position (tuple).
        """
        patterns = ['bisquare','razor', 'side', 'square']
        selected_pattern = random.choice(patterns)
        init_pos = (0, 0)
        if selected_pattern == 'square':
            if self.precision_mode == 'high':
                rows = 4
                cols = 4
            else:
                rows = 6
                cols = 6
        elif selected_pattern == 'bisquare':
            rows = 4
            cols = 4
        elif selected_pattern == 'razor':
            rows = 4
            cols = 4
            init_pos = (0, 1)
        elif selected_pattern == 'side':
            rows = 4
            cols = 4
            init_pos = (0, 2)
                  
        pattern = []
        colors = ['R', 'B', 'Y']

        for _1 in range(rows):
            row = [random.choice(colors) for _2 in range(cols)]
            pattern.append(row)
            
        if selected_pattern == 'bisquare':
            pattern[0][3] = ' '
            pattern[3][0] = ' '
        elif selected_pattern == 'razor':
            pattern[0][0] = ' '
            pattern[0][3] = ' '
            pattern[3][0] = ' '
            pattern[3][3] = ' '
        elif selected_pattern == 'side':
            pattern[0][0] = ' '
            pattern[1][0] = ' '
            pattern[3][3] = ' '
            pattern[2][3] = ' '
        return pattern, init_pos

    def shuffle_colors_for_goal(self, pattern):
        """
        Shuffles colors in a given pattern to generate a goal pattern different from the initial one.

        Parameters:
            pattern (list of lists): The initial pattern to shuffle.

        Returns:
            list of lists: A new pattern with colors shuffled for the goal state.
        """
        shuffled_pattern = []
        for row in pattern:
            new_row = []
            for tile in row:
                if tile != ' ':
                    possible_colors = ['R', 'B', 'Y']
                    possible_colors.remove(tile)
                    new_color = random.choice(possible_colors)
                    new_row.append(new_color)
                else:
                    new_row.append(' ')
            shuffled_pattern.append(''.join(new_row))
        return shuffled_pattern

    def move_cursor(self, direction):
        """
        Moves the cursor in the specified direction if possible.

        Parameters:
            direction (str): The direction to move the cursor ('up', 'down', 'left', 'right').
        """
        self.moved = True
        src_row, src_col = self.cursor_row, self.cursor_col
        if direction in ('up', 'w') and self.cursor_row > 0 and self.board.is_valid_position(self.cursor_row-1, self.cursor_col):
            self.cursor_row -= 1
        elif direction in ('down', 's') and self.cursor_row < self.board.rows - 1 and self.board.is_valid_position(self.cursor_row+1, self.cursor_col):
            self.cursor_row += 1
        elif direction in ('left', 'a') and self.cursor_col > 0 and self.board.is_valid_position(self.cursor_row, self.cursor_col-1):
            self.cursor_col -= 1
        elif direction in ('right', 'd') and self.cursor_col < self.board.cols - 1 and self.board.is_valid_position(self.cursor_row, self.cursor_col+1):
            self.cursor_col += 1

        self.move_count += 1
        self.board.move((src_row, src_col), (self.cursor_row, self.cursor_col))
        
    def get_greedy_path(self):
        """
        Retrieves the path determined by the greedy algorithm.
        
        Returns:
            list: The path as a list of moves determined by the greedy algorithm.
        """
        return improved_greedy(self.board, self.goal_board)
    
    def get_move_direction(self, other_board):
        """
        Determines the move direction based on the position difference between the current board and another board.
        
        Parameters:
            other_board (Board): The other board to compare with the current board.
        
        Returns:
            str: The direction to move ('up', 'down', 'left', 'right') or None if no direct move is identified.
        """
        if abs(other_board.current_pos[0] - self.current_pos[0]) + abs(other_board.current_pos[1] - self.current_pos[1]) == 1:
            if other_board.current_pos[0] < self.current_pos[0]:
                return 'up'
            elif other_board.current_pos[0] > self.current_pos[0]:
                return 'down'
            elif other_board.current_pos[1] < self.current_pos[1]:
                return 'left'
            elif other_board.current_pos[1] > self.current_pos[1]:
                return 'right'

        return None
    
    def get_new_position_based_on_hint(self, hint_direction):
        """
        Calculates the new position of the cursor based on the hint direction.
        
        Parameters:
            hint_direction (str): The direction of the hint ('up', 'down', 'left', 'right').
        
        Returns:
            tuple: The new cursor position as (column, row).
        """
        new_row, new_col = self.cursor_row, self.cursor_col
        if hint_direction == 'up':
            new_row -= 1
        elif hint_direction == 'down':
            new_row += 1
        elif hint_direction == 'left':
            new_col -= 1
        elif hint_direction == 'right':
            new_col += 1
        return new_col, new_row

    def set_auto_move_path(self, path):
        """
        Sets the automatic movement path for the game based on a provided path.
        
        Parameters:
            path (list): The path of moves for automatic execution.
        """
        if path:
            self.auto_move_path = path
            self.auto_move_index = 0
            self.last_auto_move_time = pygame.time.get_ticks()
      
    def complete_game_bfs(self):
        """
        Completes the game using the Breadth-First Search algorithm.
        """
        print('\n Inside Complete Game Solver BFS\n')
        path = find_path_bfs(self.board, self.goal_board)
        self.set_auto_move_path(path)

        
    def complete_game_dfs(self):
        """
        Completes the game using the Depth-First Search algorithm.
        """
        print('\n Inside Complete Game Solver DFS\n')
        path = find_path_dfs(self.board, self.goal_board)
        self.set_auto_move_path(path)

    
    def complete_game_iddfs(self):
        """
        Completes the game using the Iterative Deepening Depth-First Search algorithm.
        """
        print('\n Inside Complete Game Solver IDDFS\n')
        path = find_path_iddfs(self.board, self.goal_board) 
        self.set_auto_move_path(path)

        
    def complete_game_ucs(self):
        """
        Completes the game using the Uniform Cost Search algorithm.
        """
        print('\n Inside Complete Game Solver UCS\n')
        path = find_path_ucs(self.board, self.goal_board)     
        self.set_auto_move_path(path)

               
    def complete_game_astar(self):
        """
        Completes the game using the A* Search algorithm.
        """
        print('\n Inside Complete Game Solver ASTAR\n')
        path = find_path_astar(self.board, self.goal_board)
        self.set_auto_move_path(path)

        
    def complete_game_weighted_astar(self, weight=1.5):
        """
        Completes the game using the Weighted A* Search algorithm with a specified weight.
        
        Parameters:
            weight (float): The weight to use in the Weighted A* algorithm.
        """
        print('\n Inside Complete Game Solver Weighted ASTAR\n')
        if not self.moved:
            path = self.solution
        else:
            path = find_path_weighted_astar(self.board, self.goal_board)
        self.set_auto_move_path(path)


    def complete_game_quick_solver(self):
        """
        Completes the game using a quick solver method, which might employ various heuristics or shortcuts.
        """
        print('\n Inside Complete Game Quick Solver\n')
        if not self.moved:
            path = self.solution
        else:
            path = find_path_quick_solver(self.board, self.goal_board)
        self.set_auto_move_path(path)

            
    def get_random_hint(self, current_direction):
        """
        Generates a random hint for the next move, excluding the current direction.
        
        Parameters:
            current_direction (str): The current direction of the cursor movement.
        
        Returns:
            str: A random direction for the next move.
        """
        directions = ["up", "down", "left", "right"]
        directions.remove(current_direction)
        move = random.choice(directions)
        while not self.board.legal_move(move):
            move = random.choice(directions)
        return move

    def get_hint_greedy(self):
        """
        Generates a move hint using the greedy algorithm. If a cycle is detected with the greedy algorithm's hints,
        a random direction is chosen as the hint instead.

        Returns:
            tuple: The new cursor position based on the greedy hint.
        """
        print('\n Inside Get Hint Greedy \n')
        hint_move = greedy_hint(self.board, self.goal_board, color_distance_heuristic)
        
        if len(self.greedy_visited) == 2:
            if(self.greedy_visited[0][0].matches(self.greedy_visited[1][0]) and self.greedy_visited[0][1] == hint_move):
                hint_move = self.get_random_hint(hint_move)
            if len(self.greedy_visited) == 2:
                self.greedy_visited = []
        self.greedy_visited.append((self.board.copy(), hint_move))
        return self.get_new_position_based_on_hint(hint_move)
        
    def get_hint_monte_carlo(self):
        """
        Generates a move hint using the Monte Carlo method by simulating multiple random games and choosing
        the direction that appears most promising.

        Returns:
            tuple: The new cursor position based on the Monte Carlo hint.
        """
        print('\n Inside Get Hint Monte Carlo \n')
        hint_move = monte_carlo_hint(self.board, self.goal_board, color_distance_heuristic, 15)
        if len(self.mt_visited) == 2:
            if(self.mt_visited[0][0].matches(self.mt_visited[1][0]) and self.mt_visited[0][1] == hint_move):
                hint_move = self.get_random_hint(hint_move)
            if len(self.mt_visited) == 2:
                self.mt_visited = []
        self.mt_visited.append((self.board.copy(), hint_move))
        return self.get_new_position_based_on_hint(hint_move)

    def clicks_number_high_precision(self):
        """
        Determines the number of moves required to complete the game using a high precision method, typically
        using a more accurate but possibly slower algorithm.

        Returns:
            tuple: The path of moves and the number of moves in the path.
        """
        path = find_path_weighted_astar(self.board, self.goal_board)
        print('\n Number of clicks HIGH precision:', len(path), '\n')
        self.solution = path
        return path, len(path) 
            
    def clicks_number_low_precision(self):
        """
        Determines the number of moves required to complete the game using a low precision method, which might
        be faster but less accurate.

        Returns:
            tuple: The path of moves and the number of moves in the path.
        """
        path = find_path_quick_solver(self.board, self.goal_board)
        print('\n Number of clicks LOW precision:', len(path), '\n')
        self.solution = path
        return path, len(path) 
                    
    def is_goal_reached(self):
        """
        Checks if the current board configuration matches the goal configuration.

        Returns:
            bool: True if the current board matches the goal board, False otherwise.
        """
        return self.board.matches(self.goal_board)