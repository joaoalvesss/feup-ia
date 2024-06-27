from .board import Board
from .solver import *
import random

def get_random_hint(current_direction):
    """
    Provides a random direction that is different from the current one.

    Parameters:
    - current_direction (str): The current direction of movement.

    Returns:
    - str: A random direction chosen from "up", "down", "left", "right", excluding the current direction.
    """
    directions = ["up", "down", "left", "right"]
    directions.remove(current_direction)  # Remove the current direction from the list
    return random.choice(directions)

def top_score(scores):
    """
    Identifies the position of the highest score in a list of (direction, score) pairs.

    Parameters:
    - scores (list of tuples): A list of (direction, score) pairs.

    Returns:
    - int: The position of the highest score in the list.
    """
    score = 0
    pos = 0
    for i, (direction, s) in enumerate(scores):
        if s > score:
            score = s
            pos = i
    return pos

def greedy_hint(board, goal_board, color_distance_heuristic):
    """
    Provides a greedy hint for the next move based on the color distance heuristic.

    Parameters:
    - board (Board): The current game board.
    - goal_board (Board): The goal state of the game board.
    - color_distance_heuristic (function): Heuristic function that takes two Board objects and returns a score.

    Returns:
    - str: The direction for the best move according to the greedy approach.
    """
    best_score = float('inf')
    best_move = None

    for child_board, direction in board.child_boards():
        score = color_distance_heuristic(child_board, goal_board)
        if score < best_score:
            best_score = score
            best_move = direction

    print('Best Greedy move:', best_move, '\n')
    return best_move

def monte_carlo_hint(board, goal_board, heuristic_func, num_simulations=15):
    """
    Provides a hint for the next move using the Monte Carlo simulation method.

    Parameters:
    - board (Board): The current game board.
    - goal_board (Board): The goal state of the game board.
    - heuristic_func (function): Heuristic function for scoring boards.
    - num_simulations (int): Number of simulations to run for each possible move.

    Returns:
    - str: The direction for the best move according to the Monte Carlo approach.
    """
    best_move = None
    best_move_score = float('-inf')
    
    for move_board, move_direction in board.child_boards():
        total_score = sum(run_simulation(move_board, goal_board, heuristic_func) for _ in range(num_simulations))
        average_score = total_score / num_simulations
        if average_score > best_move_score:
            best_move_score = average_score
            best_move = move_direction

    print('Best Monte Carlo move:', best_move, '\n')
    return best_move

def run_simulation(board, goal_board, heuristic_func):
    """
    Runs a single simulation to evaluate a board state.

    Parameters:
    - board (Board): The board to simulate.
    - goal_board (Board): The goal state of the board.
    - heuristic_func (function): The heuristic function to evaluate the board state.

    Returns:
    - int: The negative score of the board based on the heuristic function. Negative is used to align with optimization direction in the Monte Carlo method.
    """
    return -heuristic_func(board, goal_board)
