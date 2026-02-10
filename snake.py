
import pygame
import time
import random

# Initialize Pygame
pygame.init()

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (213, 50, 80)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

# Define Window Size (Slightly bigger than before)
width = 400
height = 300

# Define Block Size for Snake
block_size = 10
initial_snake_speed = 10  # Snake starts slower, but will speed up

# Create the game window
game_window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

# Create Clock Object to control speed
clock = pygame.time.Clock()

# Set up fonts
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

# Function to display the score
def your_score(score):
    value = score_font.render(f"Your Score: {score}", True, BLACK)
    game_window.blit(value, [0, 0])

# Function to draw the snake
def draw_snake(block_size, snake_list):
    for block in snake_list:
        pygame.draw.rect(game_window, GREEN, [block[0], block[1], block_size, block_size])

# Function to display messages on the screen
def message(msg, color):
    mesg = font_style.render(msg, True, color)
    game_window.blit(mesg, [width / 6, height / 3])

# Main game function
def gameLoop():
    game_over = False
    game_close = False

    # Initial snake position
    x1 = width / 2
    y1 = height / 2

    # Movement of snake (velocity)
    x1_change = 0
    y1_change = 0

    # Snake list to track body blocks
    snake_list = []
    snake_length = 1

    # Initial snake speed
    snake_speed = initial_snake_speed

    # Generate initial food position
    foodx = round(random.randrange(0, width - block_size) / 10.0) * 10.0
    foody = round(random.randrange(0, height - block_size) / 10.0) * 10.0

    # Game loop
    while not game_over:

        # Game over loop
        while game_close == True:
            game_window.fill(BLUE)
            message("You lost! Press Q to Quit or C to Play Again", RED)
            your_score(snake_length - 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()

        # Event handler for key presses
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -block_size
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = block_size
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -block_size
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = block_size
                    x1_change = 0

        # Check for boundary collisions
        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:
            game_close = True

        # Update snake's head position
        x1 += x1_change
        y1 += y1_change
        game_window.fill(WHITE)

        # Draw the food
        pygame.draw.rect(game_window, BLUE, [foodx, foody, block_size, block_size])

        # Track the snake's body
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)

        # Remove the oldest segment if the snake exceeds its length
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Check for collisions with itself
        for segment in snake_list[:-1]:
            if segment == snake_head:
                game_close = True

        # Draw the snake
        draw_snake(block_size, snake_list)
        your_score(snake_length - 1)

        pygame.display.update()

        # Check if the snake has eaten the food
        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - block_size) / 10.0) * 10.0
            foody = round(random.randrange(0, height - block_size) / 10.0) * 10.0
            snake_length += 1
            snake_speed += 1  # Increase snake speed after eating food

        # Control the speed of the snake (speeds up after each food)
        clock.tick(snake_speed)

    pygame.quit()
    quit()

# Start the game
gameLoop()
