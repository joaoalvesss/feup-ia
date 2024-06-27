# game/solver.py

import random
from queue import PriorityQueue
from itertools import count
from .board import Board
from collections import deque

''' ------------------------------------ OPTIMAL ALGORITHMS ------------------------------------ '''

def bfs(board, goal_board):
    """
    Breadth-first search algorithm to find the shortest path from the current board state to the goal state.

    Parameters:
    - board (Board): The current board state.
    - goal_board (Board): The goal board state.

    Returns:
    - list: The sequence of moves to reach the goal state, or None if no solution is found.
    """
    queue = deque([(board, [])])
    visited = set()

    while queue:
        current_board, path = queue.popleft()
        if current_board.matches(goal_board):
            return path

        if current_board not in visited:                        
            visited.add(current_board)
            child_nodes = current_board.child_boards()
            for child, direction in child_nodes:
                if child not in visited:
                    new_path = path[:]
                    new_path.append(direction)
                    queue.append((child, new_path))

    return None

def dfs(board, goal_board, depth_limit=10):
    """
    Depth-first search algorithm with a depth limit to find a path from the current board state to the goal state.

    Parameters:
    - board (Board): The current board state.
    - goal_board (Board): The goal board state.
    - depth_limit (int): The maximum depth to search.

    Returns:
    - list: The sequence of moves to reach the goal state, or None if no solution is found within the depth limit.
    """
    stack = [(board, [], 0)] 
    visited = set()

    while stack:
        current_board, path, depth = stack.pop()
        if current_board.matches(goal_board):
            return path

        if depth < depth_limit and str(current_board.board) not in visited:
            visited.add(str(current_board.board)) 
            child_boards = current_board.child_boards()

            for child, direction in child_boards:
                if str(child.board) not in visited:
                    new_path = path + [direction]
                    stack.append((child, new_path, depth + 1))

    return None 

def iddfs(root_board, goal_board, max_depth):
    """
    Iterative deepening depth-first search algorithm to find a path from the current board state to the goal state.

    Parameters:
    - root_board (Board): The current board state.
    - goal_board (Board): The goal board state.
    - max_depth (int): The maximum depth to search.

    Returns:
    - list: The sequence of moves to reach the goal state, or None if no solution is found.
    """
    def dls(node, goal, depth, visited):
        if depth == 0 and node.matches(goal):
            return []
        if depth > 0:
            for child, move in node.child_boards():
                if str(child.board) not in visited:
                    visited.add(str(child.board)) 
                    path = dls(child, goal, depth - 1, visited.copy())
                    if path is not None:
                        return [move] + path
                    visited.remove(str(child.board))
        return None

    for depth in range(max_depth):
        visited = set([str(root_board.board)])  
        path = dls(root_board, goal_board, depth, visited)
        if path is not None:
            return path

    return None

def uniform_cost_search(start_board, goal_board):
    """
    Uniform cost search algorithm to find the least-cost path from the current board state to the goal state.

    Parameters:
    - start_board (Board): The current board state.
    - goal_board (Board): The goal board state.

    Returns:
    - list: The sequence of moves to reach the goal state, or None if no solution is found.
    """
    to_explore = PriorityQueue()
    unique_id = count()
    to_explore.put((0, next(unique_id), start_board, []))
    visited = set()

    while not to_explore.empty():
        cost, _, current_board, path = to_explore.get()

        if current_board.matches(goal_board):
            return path

        board_str = str(current_board)
        if board_str not in visited:
            visited.add(board_str)

            for child_board, move in current_board.child_boards():
                if str(child_board) not in visited:
                    to_explore.put((cost + 1, next(unique_id), child_board, path + [move]))

    return None


''' ------------------------------------ ROUNDED ALGORITHMS HIGH PRECISION ------------------------------------ '''

def astar(initial_board, goal_board, heuristic_func):
    """
    A* search algorithm to find the least-cost path from the current board state to the goal state using a heuristic.

    Parameters:
    - initial_board (Board): The current board state.
    - goal_board (Board): The goal board state.
    - heuristic_func (function): A function that estimates the cost from the current state to the goal.

    Returns:
    - list: The sequence of moves to reach the goal state, or None if no solution is found.
    """
    unique_id = count() 
    open_set = PriorityQueue()
    open_set.put((0 + heuristic_func(initial_board, goal_board), next(unique_id), 0, initial_board, []))
    visited = set()
    visited.add(str(initial_board.board))

    while not open_set.empty():
        _, _, cost, current_board, path = open_set.get()
        
        if current_board.matches(goal_board):
            return path
        
        for child_board, direction in current_board.child_boards():
            board_state_str = str(child_board.board)
            if board_state_str in visited:
                continue
            visited.add(board_state_str)
            
            g = cost + 1
            h = heuristic_func(child_board, goal_board)
            f = g + h
            open_set.put((f, next(unique_id), g, child_board, path + [direction]))
            
    return None    

def weighted_astar(initial_board, goal_board, heuristic_func, weight=1.5):
    """
    Weighted A* search algorithm, a variation of A* that allows trading off optimality for speed by using a weight.

    Parameters:
    - initial_board (Board): The current board state.
    - goal_board (Board): The goal board state.
    - heuristic_func (function): A function that estimates the cost from the current state to the goal.
    - weight (float): The weight applied to the heuristic function to allow faster execution at the cost of optimality.

    Returns:
    - list: The sequence of moves to reach the goal state, or None if no solution is found.
    """
    unique_id = count()
    open_set = PriorityQueue()
    open_set.put((0 + weight * heuristic_func(initial_board, goal_board), next(unique_id), 0, initial_board, []))
    visited = set()
    visited.add(str(initial_board.board))

    while not open_set.empty():
        _, _, g_cost, current_board, path = open_set.get()

        if current_board.matches(goal_board):
            return path

        for child_board, direction in current_board.child_boards():
            board_state_str = str(child_board.board)
            if board_state_str not in visited:
                visited.add(board_state_str)

                g = g_cost + 1
                h = heuristic_func(child_board, goal_board)
                f = g + weight * h
                open_set.put((f, next(unique_id), g, child_board, path + [direction]))

    return None

''' ------------------------------------ HEURISTICS ------------------------------------ '''

def color_distance_heuristic(board, goal_board):
    """
    Heuristic function that calculates the distance based on color transitions between the current board and the goal.

    Parameters:
    - board (Board): The current board state.
    - goal_board (Board): The goal board state.

    Returns:
    - int: The estimated cost to reach the goal state from the current state.
    """
    distance = 0
    color_transitions = {
        ('R', 'B'): 'Y',
        ('B', 'R'): 'Y',
        ('R', 'Y'): 'B',
        ('Y', 'R'): 'B',
        ('B', 'Y'): 'R',
        ('Y', 'B'): 'R'
    }
    for r in range(board.rows):
        for c in range(board.cols):
            current_color = board.board[r][c]
            goal_color = goal_board.board[r][c]
            if current_color != goal_color and current_color != ' ':
                if goal_color == ' ':
                    continue
                if color_transitions.get((current_color, goal_color)) or color_transitions.get((goal_color, current_color)):
                    distance += 1
                else:
                    distance += 3
    return distance

def tile_color_mismatch_heuristic(board, goal_board):
    """
    Heuristic function that counts the number of tiles that do not match their color with the goal board's corresponding tiles.

    Parameters:
    - board (Board): The current board state.
    - goal_board (Board): The goal board state.

    Returns:
    - int: The number of mismatched tiles.
    """
    mismatch_count = 0
    for r in range(board.rows):
        for c in range(board.cols):
            if board.board[r][c] != goal_board.board[r][c]:
                mismatch_count += 1
    return mismatch_count

def minimum_color_transitions_heuristic(board, goal_board):
    """
    Heuristic function that estimates the minimum number of color transitions needed to match the goal board.

    Parameters:
    - board (Board): The current board state.
    - goal_board (Board): The goal board state.

    Returns:
    - int: The estimated number of color transitions required.
    """
    transition_count = 0
    for r in range(board.rows):
        for c in range(board.cols):
            if board.board[r][c] != goal_board.board[r][c]:
                if (board.board[r][c] == 'R' and goal_board.board[r][c] == 'B') or \
                   (board.board[r][c] == 'B' and goal_board.board[r][c] == 'Y') or \
                   (board.board[r][c] == 'Y' and goal_board.board[r][c] == 'R'):
                    transition_count += 1
                else:
                    transition_count += 2
    return transition_count

def combined_heuristic(board, goal_board):
    """
    A combined heuristic function that uses both tile color mismatch and minimum color transitions heuristics.

    Parameters:
    - board (Board): The current board state.
    - goal_board (Board): The goal board state.

    Returns:
    - float: The combined heuristic value.
    """
    return 0.5 * tile_color_mismatch_heuristic(board, goal_board) + \
           0.5 * minimum_color_transitions_heuristic(board, goal_board)

''' ------------------------------------ FUNCTION CALLERS ------------------------------------ '''

def find_path_astar(initial_board, goal_board):
    """
    Initiates an A* search to find the optimal path from the initial board to the goal board using a color distance heuristic.

    Parameters:
    - initial_board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: A list of moves that leads from the initial to the goal state.
    """
    return astar(initial_board, goal_board, color_distance_heuristic)

def find_path_weighted_astar(initial_board, goal_board):
    """
    Initiates a weighted A* search to potentially speed up the search process at the cost of optimality.

    Parameters:
    - initial_board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: A list of moves that leads from the initial to the goal state, may not be optimal.
    """
    return weighted_astar(initial_board, goal_board, color_distance_heuristic)

def find_path_bfs(initial_board, goal_board):
    """
    Uses Breadth-First Search to find the shortest path from the initial board state to the goal state.

    Parameters:
    - initial_board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: The sequence of moves to reach the goal state, if a solution exists.
    """
    return bfs(initial_board, goal_board)

def find_path_dfs(initial_board, goal_board):
    """
    Uses Depth-First Search to explore possible paths from the initial board state to the goal state.

    Parameters:
    - initial_board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: A possible sequence of moves to reach the goal state, if found.
    """
    return dfs(initial_board, goal_board)

def find_path_iddfs(initial_board, goal_board, max_depth=40):
    """
    Employs Iterative Deepening Depth-First Search to combine the depth limits of DFS with the optimality of BFS.

    Parameters:
    - initial_board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.
    - max_depth (int): The maximum depth to search to.

    Returns:
    - list: The optimal sequence of moves to reach the goal state, if a solution is found within the max depth.
    """
    return iddfs(initial_board, goal_board, max_depth)

def find_path_ucs(initial_board, goal_board):
    """
    Utilizes Uniform Cost Search to find the least-cost path from the initial to the goal state, considering each move has the same cost.

    Parameters:
    - initial_board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: The sequence of moves with the least total cost to reach the goal state.
    """
    return uniform_cost_search(initial_board, goal_board)

def find_path_quick_solver(initial_board, goal_board):
    """
    A heuristic-based method intended to rapidly find a solution path from the initial to the goal state with lower computational demands.

    Parameters:
    - initial_board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: A sequence of moves that may lead to the goal state more quickly than traditional methods.
    """
    return quick_solver(initial_board, goal_board)

''' ------------------------------------ OTHERS ------------------------------------ '''

def greedy(board, goal_board):
    """
    A greedy approach that selects the move leading to the state with the highest immediate benefit, without regard for future consequences.

    Parameters:
    - board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: A sequence of moves chosen based on the greedy strategy.
    """
    path = []
    counter = 0
    while not board.matches(goal_board) and counter < 100:
        child_nodes = board.child_boards()
        match_degrees = []
        for child in child_nodes:
            match_degrees.append(child[0].match_degrees(goal_board))
        for i in range(len(match_degrees)):
            match_degrees[i] += random.random()
        path.append(child_nodes[match_degrees.index(min(match_degrees))][1])
        board = child_nodes[match_degrees.index(min(match_degrees))][0]
        counter += 1
    return path
    
def heuristic(board, goal_board):
    """
    A custom heuristic function that evaluates the board state based on the alignment of tiles with their target positions.

    Parameters:
    - board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - int: The heuristic score representing the board's closeness to the goal state.
    """
    score = 0
    for r in range(board.rows):
        for c in range(board.cols):
            if board.board[r][c] == goal_board.board[r][c]:
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board.rows and 0 <= nc < board.cols:
                        if board.board[nr][nc] == goal_board.board[nr][nc]:
                            score += 1
    return -score

def improved_greedy(board, goal_board):
    """
    An enhanced greedy search that incorporates additional strategies or information to select moves that lead towards the goal state.

    Parameters:
    - board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: A possibly more efficient sequence of moves towards the goal state, based on the improved greedy strategy.
    """
    path = []
    counter = 0
    visited = set()
    while not board.matches(goal_board) and counter < 100:
        child_nodes = board.child_boards()
        best_score = float('inf')
        best_child = None
        best_direction = None
        
        for child, direction in child_nodes:
            board_hash = hash(str(child.board))
            if board_hash not in visited:
                score = heuristic(child, goal_board)
                if score < best_score:
                    best_score = score
                    best_child = child
                    best_direction = direction
        if best_child is not None:
            path.append(best_direction)
            board = best_child
            visited.add(hash(str(board.board)))
            counter += 1
        else:
            break 
        
    return path

''' ------------------------------------ QUICK SOLVER LOW PRECISION ------------------------------------ '''

def get_centered_submatrix(matrix, pos):
    """
    Extracts a 3x3 submatrix centered around a given position in a larger matrix, padding with ' ' if necessary.

    Parameters:
    - matrix: The larger matrix from which to extract the submatrix.
    - pos (tuple): The center position (row, col) of the desired submatrix.

    Returns:
    - list: The 3x3 submatrix centered at the given position.
    """
    col, row = pos
    submatrix = []
    rows = len(matrix)
    cols = len(matrix[0])

    for i in range(row - 1, row + 2):
        row_values = []
        for j in range(col - 1, col + 2):
            if 0 <= i < rows and 0 <= j < cols:
                row_values.append(matrix[i][j])
            else:
                row_values.append(' ')
        submatrix.append(row_values)

    return submatrix

def random_move_sequence(current_board):
    """
    Generates a random sequence of legal moves from the current board state.

    Parameters:
    - current_board (Board): The current state of the board.

    Returns:
    - list: A sequence of up to three random, legal moves.
    """
    moves = []
    board = current_board.copy()
    for i in range(3):
        possible_moves = ['right', 'left', 'up', 'down']
        legal_moves = []
        for move in possible_moves:
            if board.legal_move(move):
                legal_moves.append(move)
        selected = random.choice(legal_moves)
        moves.append(selected)
        board.move_dir(selected)

    return moves

def quick_solver(current_board, goal_board):
    """
    A fast, heuristic-based algorithm designed to quickly find a path to the goal state by solving smaller sub-problems.

    Parameters:
    - current_board (Board): The current state of the board.
    - goal_board (Board): The target state of the board.

    Returns:
    - list: A sequence of moves that leads towards the goal state, potentially more efficiently than traditional exhaustive search methods.
    """
    board = current_board.copy()
    path = []
    while not board.matches(goal_board):
        pattern = get_centered_submatrix(board.board, board.current_pos)
        goal_pattern = get_centered_submatrix(goal_board.board, board.current_pos)
        small_board = Board(3, 3, pattern, (1, 1))
        small_goal_board = Board(3, 3, goal_pattern, (1, 1))


        temp_path = weighted_astar(small_board, small_goal_board, color_distance_heuristic)
        if temp_path == [] or temp_path == None:
            temp_path = random_move_sequence(board)

        if temp_path != None:
            path += temp_path
        else:
            break
        for move in temp_path:
            board.move_dir(move)
    return path