# Amado

[Final Presentation](docs/Final_AI_Presentation.pdf)

Amado is a fun and interactive way to witness various search algorithms in action.In Amado, players are presented with a grid where each cell can be one of three colors or left empty. The game begins with a randomly generated pattern, and the player's task is to transform this initial state into a target configuration using the least number of moves possible. To achieve this, players can move a cursor around the board, changing the colors of the tiles based on the game's unique color-changing rules: moving from one color to another transforms the destination tile into a third color, according to a predefined color cycle. 

![Default amado image](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/e27a5c3695835ac9bff35f68316b134250560942/default.png?raw=true)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

- Python 3.6 or higher
- Pygame library
- NumPy library

To install Pygame and NumPy, run the following commands:

```bash
pip install pygame
pip install numpy
```

## Running the Game

To run the game, simply execute the main script:

```bash
python main.py
```

### How to Play

Navigate the game board and match the colors of the tiles to achieve the goal board configuration. Use intuitive controls and advanced algorithms to find your way through the puzzles. **Use mouse in menus and arrow keys in game.**

![Hover menu](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/69e73745dbdedec9a3fa20aef20043660dd3bebd/hover_menu.png?raw=true)

#### Controls:
- **Mouse Interaction:** Every game menu can be navigated using the mouse. To select a menu option, hover over the option until it highlights, then click with the left mouse button to select it.
- **Arrow Keys:** Move the cursor across the board to navigate between tiles.
- **`C`:** Activate the auto-complete feature, allowing the game's AI to solve the puzzle.
- **`H`:** Request a hint for the next best move.

![Commands](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/e27a5c3695835ac9bff35f68316b134250560942/commands.png?raw=true)

#### Game Modes and Algorithms:
The game features two distinct modes, each employing specific algorithms for puzzle solving and hints:

![Precision Modes](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/69e73745dbdedec9a3fa20aef20043660dd3bebd/precision_menu.png?raw=true)

- **High Precision Mode:**
  - **Auto-complete (`C`):** Utilizes the **Weighted A* algorithm** for puzzle solving, offering a balance between speed and optimality.

```python
def weighted_astar(initial_board, goal_board, heuristic_func, weight=1.5):
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
```
  - **Hint (`H`):** Employs a **Monte Carlo search**, providing high-quality suggestions by simulating numerous potential outcomes.
```python
def monte_carlo_hint(board, goal_board, heuristic_func, num_simulations=15):
    best_move = None
    best_move_score = float('-inf')
    
    for move_board, move_direction  in board.child_boards():
        total_score = 0
        for _ in range(num_simulations):
            # Run a simulation
            score = run_simulation(move_board, goal_board, heuristic_func)
            total_score += score
            
        average_score = total_score / num_simulations
        if average_score > best_move_score:
            best_move_score = average_score
            best_move = move_direction
    print('Best Monte Carlo move: ', best_move, '\n')
    return best_move

def run_simulation(board, goal_board, heuristic_func):
    return -heuristic_func(board, goal_board) 
```
- **Low Precision Mode:**
  - **Auto-complete (`C`):** Leverages the **Quick Solver**, prioritizing speed over optimality for faster resolutions.
```python
  def quick_solver(current_board, goal_board):
    board = current_board.copy()
    path = []
    while not board.matches(goal_board):
        pattern = get_centered_submatrix(board.board, board.current_pos)
        goal_pattern = get_centered_submatrix(goal_board.board, board.current_pos)
        small_board = Board(3, 3, pattern, (1, 1))
        small_goal_board = Board(3, 3, goal_pattern, (1, 1))


        temp_path = weighted_astar(small_board, small_goal_board, color_distance_heuristic)
        #temp_path = bfs(small_board, small_goal_board)
        if temp_path == [] or temp_path == None:
            temp_path = random_move_sequence(board)

        if temp_path != None:
            path += temp_path
        else:
            break
        for move in temp_path:
            board.move_dir(move)
    return path
  ```
  - **Hint (`H`):** Uses a **greedy algorithm** for quick, straightforward hints based on the current game state.
```python
  def greedy_hint(board, goal_board, color_distance_heuristic):
      # Initialize the best score to a high number and best move to None
    best_score = float('inf')
    best_move = None

    # Consider all possible moves from the current position
    for child_board, direction in board.child_boards():
        # Evaluate the heuristic function for the child board
        score = color_distance_heuristic(child_board, goal_board)
        
        # If this move has a better score, update the best score and best move
        if score < best_score:
            best_score = score
            best_move = direction
            
    return best_move
  ```
### Basic Features

- **Interactive Gameplay:** Visual representation of the game board that challenges user's problem-solving abilities and strategic planning, with interactive buttons to move, execute algorithms, reset the game or quit.
- **Time tracker:** Time-tracking feature, setting a finite window for achieving the game's objective.
- **Algorithm Visualization:** Witness firsthand the process and efficiency of multiple search algorithms as they find a solution to the puzzle.
- **Educational Insights:** Learn the differences between various search strategies, including both uninformed and informed search algorithms.
- **Real-time Comparison:** Instantly switch between algorithms and precision modes to compare their paths and performance metrics, providing a deeper understanding of their characteristics and applications.

### Main Features

- **Hint system:** Provides immediate algorithm-generated suggestions for the next move. This tool is designed to assist in navigating complex puzzles and to demonstrate the application of search strategies in real-time. It helps players understand the reasoning behind specific moves recommended by different algorithms.
- **Game Solver:** An automated feature that solves the given puzzle configuration using various search algorithms. It displays each step from the initial state to the solution, allowing for the analysis of different algorithms in terms of efficiency, number of moves, and execution time.
- **Optimal number o clicks:**  Challenges players to solve puzzles with the minimum number of moves, emphasizing strategy and planning. This feature highlights the concept of solution optimality in search algorithms, encouraging an understanding of efficient problem-solving techniques and the exploration of algorithmic optimization.


### Visual Guides

Understanding the visual elements of Amado enhances the gameplay experience, providing clarity on goals, progress, and outcomes. Below are key visuals from the game:

- **Current Board State:** The dynamic state of the game board as you navigate and attempt to match the goal configuration. Visual cues and the cursor's movement reflect your interactions and decisions in real-time.
  
  ![Current Board](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/69e73745dbdedec9a3fa20aef20043660dd3bebd/current_board.png?raw=true)

- **Goal Board Configuration:** The target layout you aim to achieve by matching the colors of the tiles according to this template. It serves as your guide and objective throughout the game.
  
  ![Goal Board](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/69e73745dbdedec9a3fa20aef20043660dd3bebd/goal_board.png?raw=true)

- **Commands Guide:**  A quick reference to the game's controls, offering an at-a-glance overview of how to navigate and interact with the game efficiently. Activating the auto-solver, and requesting hints, Back to Menu.
  
  ![Commands](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/e27a5c3695835ac9bff35f68316b134250560942/commands.png?raw=true)

- **Time Tracking and Optimal Clicks:** A dual-purpose visual that represents both the urgency of solving puzzles within a set timeframe and the challenge of minimizing the number of moves to reach the optimal clicks number. 
  
  ![Time Tracking and Optimal Clicks](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/69e73745dbdedec9a3fa20aef20043660dd3bebd/clicks_time.png?raw=true)

- **Game Victory Screen:** A celebratory screen that appears upon successfully matching the current board with the goal configuration. It signifies the completion of the puzzle and your victory.
  
  ![Game Win](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/69e73745dbdedec9a3fa20aef20043660dd3bebd/game_win.png?raw=true)

- **Game Over Screen:** This screen is displayed when the player fails to complete the puzzle within the allotted time, marking the end of the current game session.
  
  ![Game Over](https://github.com/Alberto-Serra/IART-IMAGES-23-24/blob/69e73745dbdedec9a3fa20aef20043660dd3bebd/game_over.png?raw=true)

These visuals are integral to the gameplay, offering both a challenge and a reference point as players navigate through Amado's puzzles.


## Built With

- [Python](https://www.python.org/) - The programming language used.
- [Pygame](https://www.pygame.org/news) - The game library used to create the GUI.
- [NumPy](https://numpy.org/) - Used for efficient numerical computations.

## Authors

- João Alves - up202108670
- Alberto Serra - up202103627
- Eduardo Sousa - up202103342
