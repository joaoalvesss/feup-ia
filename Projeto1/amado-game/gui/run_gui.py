# gui/run_gui.py

import pygame
from game.game import Game
import random

# Constants for the game's visual aspects
TILE_SIZE = 80  # The size of each tile in pixels
TILE_OFFSET = 10  # The space between tiles in pixels
TOTAL_TILE_SIZE = TILE_SIZE + TILE_OFFSET  # The total size of a tile including the offset
BOARD_OFFSET = (50, 50)  # The offset from the top left corner to start drawing the board
COLORS = {
    'R': (255, 0, 0),  # Red color for the tile
    'B': (0, 0, 255),  # Blue color for the tile
    'Y': (255, 255, 0),  # Yellow color for the tile
    ' ': (0, 0, 0),  # Black color for empty space
    None: (0, 0, 0)  # Black for undefined/unused tiles
}
CURSOR_COLOR = (76, 187, 23)  # Color of the cursor that highlights the current tile
CURSOR_THICKNESS = 5  # The thickness of the cursor's border
HINT_CURSOR_COLOR = (0, 255, 255)  # Color for the hint cursor that suggests a move

def draw_board(screen, game, board, offset=(0, 0), is_goal=False):
    """
    Draws the game board on the screen.

    Parameters:
    - screen: The Pygame display surface to draw on.
    - game: The current game instance.
    - board: The board to be drawn.
    - offset: A tuple (x, y) representing the offset to start drawing from.
    - is_goal: A boolean indicating if this board is the goal state.
    """
    if is_goal:
        scale_factor = 1/2
        tile_size_scaled = TILE_SIZE * scale_factor
        tile_offset_scaled = TILE_OFFSET * scale_factor
        total_tile_size_scaled = tile_size_scaled + tile_offset_scaled
    else:
        tile_size_scaled = TILE_SIZE
        tile_offset_scaled = TILE_OFFSET
        total_tile_size_scaled = TOTAL_TILE_SIZE

    # Draw each tile on the board
    for row in range(board.rows):
        for col in range(board.cols):
            color = COLORS[board.board[row][col]]
            if is_goal:
                pygame.draw.rect(screen, color, (
                    offset[0] + col * total_tile_size_scaled, 
                    offset[1] + row * total_tile_size_scaled, 
                    tile_size_scaled, tile_size_scaled))
            else:
                pygame.draw.rect(screen, color, (
                    offset[0] + col * TOTAL_TILE_SIZE, 
                    offset[1] + row * TOTAL_TILE_SIZE, 
                    TILE_SIZE, TILE_SIZE))
    
    if not is_goal:
        pygame.draw.rect(screen, CURSOR_COLOR, (
            offset[0] + game.cursor_col * TOTAL_TILE_SIZE, 
            offset[1] + game.cursor_row * TOTAL_TILE_SIZE, 
            TILE_SIZE, TILE_SIZE), CURSOR_THICKNESS)
        
def draw_clicks_number(screen, clicks, position, bar_size=(50, 10), outline_color=(255, 255, 0)):
    """
    Draws the number of clicks (moves) made on the screen.

    Parameters:
    - screen: The Pygame display surface to draw on.
    - clicks: The number of clicks (moves) to display.
    - position: The position to draw the number of clicks.
    - bar_size: The size of the bar representing a single click.
    - outline_color: The color of the outline around the click number.
    """
    font = pygame.font.Font(None, 45)
    outline_rect_position = (position[0] + bar_size[0] + 40, position[1] - 10)
    outline_rect_size = (90, 60)
    pygame.draw.rect(screen, outline_color, pygame.Rect(outline_rect_position, outline_rect_size), 4)
    
    text_surf = font.render(str(clicks), True, (255, 255, 255))
    text_pos = outline_rect_position[0] + (outline_rect_size[0] - text_surf.get_width()) // 2, \
               outline_rect_position[1] + (outline_rect_size[1] - text_surf.get_height()) // 2
    screen.blit(text_surf, text_pos)

def draw_time_bars(screen, remaining_time, position, bar_size=(50, 10), spacing=4, outline_color=(255, 255, 0), color=(200, 0, 0)):
    """
    Draws the remaining time as a series of bars on the screen.

    Parameters:
    - screen: The Pygame display surface to draw on.
    - remaining_time: The remaining time in milliseconds.
    - position: The position to start drawing the time bars.
    - bar_size: The size of each individual time bar.
    - spacing: The spacing between each time bar.
    - outline_color: The color of the outline around the time bars.
    - color: The fill color of the time bars.
    """
    total_seconds = 150
    seconds_left = remaining_time // 1000
    max_bars = total_seconds // 10
    bars_to_draw = seconds_left // 10 + (1 if seconds_left % 10 != 0 else 0)

    for i in range(max_bars - bars_to_draw, max_bars):
        if i < 2: 
            color = (0, 200, 0)
        else:
            color = (200, 0, 0)

        bar_position = (position[0], position[1] + (max_bars + i - 1) * (bar_size[1] + spacing) - 350)
        pygame.draw.rect(screen, color, pygame.Rect(bar_position, bar_size))


def show_menu(screen):
    """
    Displays the main menu and handles user interaction.

    Parameters:
    - screen: The Pygame display surface to draw the menu on.
    
    Returns:
    - boolean: True if the user selects 'Start Game', False if the user selects 'Quit'.
    """
    menu_running = True
    menu_font = pygame.font.Font(None, 36)
    amado_font = pygame.font.Font(None, 225)

    start_color = (255, 255, 255)
    quit_color = (255, 255, 255)
    color_change_interval = 2000
    last_color_change_time = pygame.time.get_ticks()
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255)]
    current_color_index = 0

    while menu_running:
        screen.fill((0, 0, 0))
        mouse_pos = pygame.mouse.get_pos()
        start_text = menu_font.render('Start Game', True, start_color)
        quit_text = menu_font.render('Quit', True, quit_color)
        amado_text = amado_font.render('A M A D O', True, colors[current_color_index])
        amado = 'A M A D O'
        start_rect = start_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 20))
        quit_rect = quit_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 60))
        amado_rect = amado_text.get_rect(center=(screen.get_width() // 2, 250))

        if start_rect.collidepoint(mouse_pos):
            start_color = (255, 0, 0)
        else:
            start_color = (255, 255, 255)

        if quit_rect.collidepoint(mouse_pos):
            quit_color = (255, 0, 0)
        else:
            quit_color = (255, 255, 255)

        current_time = pygame.time.get_ticks()
        if current_time - last_color_change_time >= color_change_interval:
            current_color_index = (current_color_index + 1) % len(colors)
            last_color_change_time = current_time

        for i in range(5):
            screen.blit(amado_font.render(amado[0:9-2*i], True, colors[(current_color_index+i)%len(colors)]), amado_rect)
            

        screen.blit(start_text, start_rect)
        screen.blit(quit_text, quit_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                menu_running = False
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if start_rect.collidepoint(event.pos):
                    menu_running = False
                    return True
                elif quit_rect.collidepoint(event.pos):
                    menu_running = False
                    return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    menu_running = False
                    pygame.quit()
                    return False
                elif event.key == pygame.K_RETURN:
                    menu_running = False
                    return True
        pygame.display.flip()
        
def show_precision_mode_menu(screen):
    """
    Displays a menu for the user to select the precision mode (high or low).

    Parameters:
    - screen: The Pygame display surface to draw the precision mode menu on.
    
    Returns:
    - string: 'high' for High Precision Mode, 'low' for Low Precision Mode, or 'quit' to exit.
    """
    selection_running = True
    font = pygame.font.Font(None, 36)
    amado_font = pygame.font.Font(None, 225)

    screen_width = screen.get_width()
    screen_height = screen.get_height()
    options = ["High Precision Mode", "Low Precision Mode", "Back to Menu"]
    options_colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255)]
    color_change_interval = 2000
    last_color_change_time = pygame.time.get_ticks()
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255)]
    current_color_index = 0

    # Rects and surfaces for options
    options_rects = []
    options_surfaces = []

    for option in options:
        text_surface = font.render(option, True, (255, 255, 255))
        options_surfaces.append(text_surface)
        text_rect = text_surface.get_rect(center=(screen_width // 2, screen_height // 2 + options.index(option) * 50))
        options_rects.append(text_rect)

    selected_precision_mode = None

    while selection_running:
        screen.fill((0, 0, 0))
        mouse_pos = pygame.mouse.get_pos()
        amado_text = amado_font.render('A M A D O', True, colors[current_color_index])
        amado = 'A M A D O'
        amado_rect = amado_text.get_rect(center=(screen.get_width() // 2, 250))

        # Update color based on hover and redraw options
        for i, rect in enumerate(options_rects):
            if rect.collidepoint(mouse_pos):
                options_colors[i] = (255, 0, 0)  # Hover color
            else:
                options_colors[i] = (255, 255, 255)  # Default color
            text_surface = font.render(options[i], True, options_colors[i])
            screen.blit(text_surface, rect.topleft)
            
        current_time = pygame.time.get_ticks()
        if current_time - last_color_change_time >= color_change_interval:
            current_color_index = (current_color_index + 1) % len(colors)
            last_color_change_time = current_time
            
        for i in range(5):
            screen.blit(amado_font.render(amado[0:9-2*i], True, colors[(current_color_index+i)%len(colors)]), amado_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                selection_running = False
                return "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, rect in enumerate(options_rects):
                    if rect.collidepoint(event.pos):
                        if i == 2:  # The index of the "Back to Menu" option
                            return "menu"
                        else:
                            selected_precision_mode = "high" if i == 0 else "low"
                            selection_running = False

        pygame.display.flip()
    return selected_precision_mode

def show_completion_screen(screen, move_count, won):
    """
    Displays a screen upon game completion, showing the result and options to play again or quit.

    Parameters:
    - screen: The Pygame display surface to draw the completion screen on.
    - move_count: The number of moves taken by the player.
    - won: A boolean indicating if the player won the game.
    
    Returns:
    - string: 'play_again' if the user chooses to play again, 'quit' to exit.
    """
    completion_running = True
    button_font = pygame.font.Font(None, 50)
    screen_width = screen.get_width()
    button_width = 200
    button_height = 50
    play_again_button_x = screen_width // 2 - button_width // 2
    quit_button_x = screen_width // 2 - button_width // 2

    buttons = {
        "play_again": {"rect": pygame.Rect(play_again_button_x, 350, button_width, button_height), "text": "Play Again"},
        "quit": {"rect": pygame.Rect(quit_button_x, 410, button_width, button_height), "text": "Main Menu"},
    }
    result = None

    while completion_running:
        screen.fill((0, 0, 0))
        mouse_pos = pygame.mouse.get_pos()

        # Display congratulations text
        if won:
            congrats_text = button_font.render('Congratulations!', True, (255, 215, 0))
        else:
            congrats_text = button_font.render('Game Over!', True, (255, 215, 0))
        congrats_rect = congrats_text.get_rect(center=(screen_width // 2, 200))
        screen.blit(congrats_text, congrats_rect.topleft)

        # Display moves done
        moves_text = button_font.render(f'Moves: {move_count}', True, (255, 215, 0))
        moves_rect = moves_text.get_rect(center=(screen_width // 2, 275))
        screen.blit(moves_text, moves_rect.topleft)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                completion_running = False
                result = "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button_key, button_details in buttons.items():
                    if button_details["rect"].collidepoint(event.pos):
                        result = button_key
                        completion_running = False

        for button_key, button_details in buttons.items():
            if button_details["rect"].collidepoint(mouse_pos):
                text_color = (200, 0, 0)
            else:
                text_color = (255, 255, 255)

            pygame.draw.rect(screen, (0, 0, 0), button_details["rect"])
            text_surf = button_font.render(button_details["text"], True, text_color)
            text_rect = text_surf.get_rect(center=button_details["rect"].center)
            screen.blit(text_surf, text_rect)

        pygame.display.flip()
        if result:
            return result

def run_gui():
    """
    The main function that initializes the game, handles the game loop, and updates the display.
    """
    quit_game = False
    pygame.init()
    game = ''
    screen = pygame.display.set_mode((1280, 720))
    font = pygame.font.Font(None, 45)
    bigfont = pygame.font.Font(None, 55)
    sidefont = pygame.font.Font(None, 32)
    clicks_number = 0
    path_loaded = None
    if not show_menu(screen):
        pygame.quit()
        return
    else:
        precision_mode = show_precision_mode_menu(screen)
        if precision_mode == "menu":
            run_gui()
            return
        game = Game(precision_mode)
        if precision_mode == "high":
            path_loaded, clicks_number = game.clicks_number_high_precision()
            
        if precision_mode == "low":
           path_loaded, clicks_number = game.clicks_number_low_precision()
        initial_time = game.board.rows * 20
        countdown_ticks = initial_time * 1000 
        start_ticks = pygame.time.get_ticks()
    
    pygame.display.set_caption('Amado Game')
    clock = pygame.time.Clock()
    running = True
    hint_pos = None

    while running:
        current_ticks = pygame.time.get_ticks()
        elapsed_time = current_ticks - start_ticks
        remaining_time = countdown_ticks - elapsed_time

        if precision_mode == 'high':
            fps = 500
        else:
            fps = 100

        if hasattr(game, 'auto_move_path') and game.auto_move_index < len(game.auto_move_path):
            if current_ticks - game.last_auto_move_time >= fps:
                move = game.auto_move_path[game.auto_move_index]
                game.move_cursor(move) 
                game.auto_move_index += 1
                game.last_auto_move_time = current_ticks

                if game.auto_move_index == len(game.auto_move_path):
                    del game.auto_move_path
                    del game.auto_move_index
                    del game.last_auto_move_time

        if remaining_time <= 0:
            print("Time's up!")
            choice = show_completion_screen(screen, game.move_count, False)
            if choice == "play_again":
                precision_mode = show_precision_mode_menu(screen)
                game = Game(precision_mode)
                if precision_mode == "high":
                    path_loaded, clicks_number = game.clicks_number_high_precision()
                if precision_mode == "low":
                    path_loaded, clicks_number = game.clicks_number_low_precision()
                initial_time = game.board.rows * 20
                countdown_ticks = initial_time * 1000 
                start_ticks = pygame.time.get_ticks() 
                running = True 
            elif choice == "quit":
                break
     
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    game.move_cursor('up')
                    path_loaded = None
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    game.move_cursor('down')
                    path_loaded = None
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a: 
                    game.move_cursor('left')
                    path_loaded = None
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    game.move_cursor('right')
                    path_loaded = None
                    
                    ''' ------------------------------------ HIGH PRECISION MODE ------------------------------------ '''
                if game.precision_mode == "high":
                    if event.key == pygame.K_c:
                        if path_loaded:
                            game.set_auto_move_path(path_loaded)
                            path_loaded = None
                        else:
                            draw_text(screen, "Calculating path and playing...", (screen.get_width() // 2 - 180, screen.get_height() - 50))
                            pygame.display.flip() 
                            game.complete_game_weighted_astar()    
                    elif event.key == pygame.K_h:
                        hint_pos = game.get_hint_monte_carlo()
                        
                    ''' ------------------------------------ LOW PRECISION MODE ------------------------------------ '''
                elif game.precision_mode == "low":
                    if event.key == pygame.K_c:
                        if path_loaded:
                            game.set_auto_move_path(path_loaded)
                            path_loaded = None
                        else:
                            draw_text(screen, "Calculating path and playing...", (screen.get_width() // 2 - 180, screen.get_height() - 50))
                            pygame.display.flip() 
                            game.complete_game_quick_solver()
                    elif event.key == pygame.K_h:
                        hint_pos = game.get_hint_greedy()
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                    quit_game = True
                elif event.key == pygame.K_b or event.key == pygame.K_m:
                    running = False
    
            
        screen.fill((0, 0, 0))
        center_x = ((screen.get_width() - 350) + screen.get_width()) // 2
        goal_offset_y = 80
        goal_offset_x = center_x - ((game.goal_board.cols * (TILE_SIZE * 0.5 + TILE_OFFSET * 0.5)) // 2)
        playable_area_width = screen.get_width() - 350
        board_width = game.board.cols * TOTAL_TILE_SIZE
        board_height = game.board.rows * TOTAL_TILE_SIZE
        board_start_x = (playable_area_width - board_width) // 2
        board_start_y = (screen.get_height() - board_height) // 2 + 100
        board_offset = (board_start_x, board_start_y)

        draw_board(screen, game, game.board, board_offset)
        draw_board(screen, game, game.goal_board, (goal_offset_x, goal_offset_y), True)

        amado_text = font.render("A  M  A  D  O", True, (76, 187, 23))
        amado_pos = (center_x - amado_text.get_width() // 2, 30)  
        screen.blit(amado_text, amado_pos)

        time_bars_position = (screen.get_width() - 250, screen.get_height() - 75)
        
        draw_clicks_number(screen, clicks_number, time_bars_position)
        draw_time_bars(screen, remaining_time, time_bars_position)

        move_count_text = bigfont.render(f"Moves: {game.move_count}", True, (255, 255, 255))
        move_count_pos = (((screen.get_width() - 350) // 2) - (move_count_text.get_width() // 2), 30)
        screen.blit(move_count_text, move_count_pos)
        
        hint_text = sidefont.render("Hint - Press H", True, (255, 255, 255))
        complete_text = sidefont.render("Complete - Press C", True, (255, 255, 255))
        back_text = sidefont.render("Go back - Press B", True, (255, 255, 255))
        hint_screen = (30, 50)
        complete_screen = (30, 80)
        back_screen = (30, 110)
        screen.blit(hint_text, hint_screen)
        screen.blit(complete_text, complete_screen)
        screen.blit(back_text, back_screen)
        
        line_color = (200, 0, 0)
        line_start_pos = (screen.get_width() - 350, 0) 
        line_end_pos = (screen.get_width() - 350, screen.get_height())
        pygame.draw.line(screen, line_color, line_start_pos, line_end_pos, 4)
        
        if hint_pos and (game.cursor_col, game.cursor_row) != hint_pos:
            hint_col, hint_row = hint_pos
            hint_cursor_rect = (
                board_offset[0] + hint_col * TOTAL_TILE_SIZE, 
                board_offset[1] + hint_row * TOTAL_TILE_SIZE, 
                TILE_SIZE, TILE_SIZE
            )
            pygame.draw.rect(screen, HINT_CURSOR_COLOR, hint_cursor_rect, CURSOR_THICKNESS)
        else:
            hint_pos = None

        if game.is_goal_reached():
            print("Goal reached!")
            choice = show_completion_screen(screen, game.move_count, True)
            if choice == "play_again":
                precision_mode = show_precision_mode_menu(screen)
                game = Game(precision_mode)
                if precision_mode == "high":
                    path_loaded, clicks_number = game.clicks_number_high_precision()
                if precision_mode == "low":
                    path_loaded, clicks_number = game.clicks_number_low_precision()
                initial_time = game.board.rows * 20
                countdown_ticks = initial_time * 1000 
                start_ticks = pygame.time.get_ticks() 
                running = True 
            elif choice == "quit":
                break

        pygame.display.flip()
        clock.tick(60)

    if not quit_game:
        run_gui()
    else:
        pygame.quit()

def draw_text(screen, text, position, color=(255, 255, 255)):
    """
    Utility function to draw text on the screen.

    Parameters:
    - screen: The Pygame display surface to draw the text on.
    - text: The text string to be displayed.
    - position: The position where the text is to be displayed.
    - color: The color of the text.
    """
    font = pygame.font.Font(None, 30)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)
    screen.blit(text_surface, text_rect)

if __name__ == '__main__':
    run_gui()
