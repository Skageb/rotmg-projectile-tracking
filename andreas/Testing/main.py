# main.py
import sys
import pygame
from movement_generator import MovementGenerator, SCREEN_WIDTH, SCREEN_HEIGHT
from predictor import predict_future

# Initialize pygame.
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Path Estimation")
clock = pygame.time.Clock()

# Create a movement generator instance with configurable sine parameters.
# (sine_amplitude=15, sine_frequency=10 produce a rapidly oscillating sine.)
mov_gen = MovementGenerator(speed=200, sine_amplitude=15, sine_frequency=10)

# History of (position, velocity) tuples.
history = []
max_history_length = 40  # store a limited history

# Configurable radius for the circles drawn around each predicted future point.
feature_radius = 10  # in pixels

running = True
while running:
    dt = clock.tick(60) / 1000.0  # delta time in seconds (target 60 FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the projectile state.
    pos, vel, segment_changed = mov_gen.update(dt)
    
    # Clear history when the movement type changes.
    if segment_changed:
        history = []
    
    history.append((pos, vel))
    if len(history) > max_history_length:
        history.pop(0)
    
    # Prepare history data for prediction (predictor expects a list of (pos, vel) tuples).
    future_points = predict_future(history, num_points=100, dt=1/60)
    
    # Fill the screen with a dark, pleasant green.
    screen.fill((10, 50, 10))
    
    # Draw the predicted future path as a soft grey connected line.
    if future_points:
        pygame.draw.lines(screen, (200, 200, 200), False,
                          [(int(p[0]), int(p[1])) for p in future_points], 2)
        
        # For each predicted future point, draw a filled circle.
        # Use a soft red for points less than 1 second into the future,
        # and a soft yellow for points further than 1 second.
        for i, p in enumerate(future_points, start=1):
            offset_time = 1/30 * i  # since future dt is 0.1 seconds per point
            if offset_time < 1.0:
                color = (200, 100, 100)  # soft red
            else:
                color = (220, 220, 150)  # soft yellow
            pygame.draw.circle(screen, color, (int(p[0]), int(p[1])), feature_radius, 0)
    
    # Draw the projectile as a blue dot.
    pygame.draw.circle(screen, (0, 0, 255), (int(pos[0]), int(pos[1])), 6)
    
    # Draw the velocity vector as a green line.
    # vel_scale = 1
    # end_point = (int(pos[0] + vel[0] * vel_scale), int(pos[1] + vel[1] * vel_scale))
    # pygame.draw.line(screen, (0, 255, 0), (int(pos[0]), int(pos[1])), end_point, 2)
    
    pygame.display.flip()

pygame.quit()
sys.exit()
