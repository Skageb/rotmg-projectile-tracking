# main.py
import sys
import pygame
from movement_generator import MovementGenerator, SCREEN_WIDTH, SCREEN_HEIGHT

def generate_path(dt=1/60, total_time=20):
    mov_gen = MovementGenerator(speed=200, sine_amplitude=15, sine_frequency=10)
    path = []
    t = 0.0
    while t < total_time:
        pos, vel, _ = mov_gen.update(dt)
        path.append((pos, vel))
        t += dt
    return path

# Initialize pygame.
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Projectile Path Video")
clock = pygame.time.Clock()

# Generate the path.
dt = 1/60
total_time = 20  # seconds
path = generate_path(dt=dt, total_time=total_time)
num_frames = len(path)
current_index = 0

running = True
while running:
    clock.tick(60)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                path = generate_path(dt=dt, total_time=total_time)
                num_frames = len(path)
                current_index = 0

    screen.fill((10, 50, 10))
    
    if len(path) > 1:
        pts = [(int(pos[0]), int(pos[1])) for (pos, _) in path]
        pygame.draw.lines(screen, (150, 150, 150), False, pts, 2)
    
    pos, vel = path[current_index]
    pygame.draw.circle(screen, (0, 0, 255), (int(pos[0]), int(pos[1])), 6)
    
    vel_scale = 0.1
    end_point = (int(pos[0] + vel[0]*vel_scale), int(pos[1] + vel[1]*vel_scale))
    pygame.draw.line(screen, (0, 255, 0), (int(pos[0]), int(pos[1])), end_point, 2)
    
    current_index = (current_index + 1) % num_frames
    
    pygame.display.flip()

pygame.quit()
sys.exit()
