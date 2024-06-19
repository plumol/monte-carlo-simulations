import pygame
import numpy as np

pygame.init()

screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
stationary_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

radius = 40
v = [100, 100]

def overlap(p1, p2):
    distance = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    #print(distance)
    if distance < 2 * radius:
        print("overlap")

def collision_time(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    # a = ((v[0] + v[1]))**2
    # b = 2 * (v[0] * dx + v[1] * dy)
    # c = dx**2 + dy**2 - (radius + radius)**2

    a = (v[0]**2 + v[1]**2)
    b = 2 * - (dx * v[0] + dy * v[1])
    c = dx**2 + dy**2 - (radius +radius)**2

    discriminant = b**2 - 4 * a * c

    print(a, b, c)

    if v[0] > 0:
        t_v_wall = (screen.get_width() - radius - p1.x) / v[0]
    elif v[0] < 0:
        t_v_wall = (- p1.x + radius) / v[0]
    else:
        t_v_wall = float('inf')  # No horizontal movement

    # Calculate time to hit the horizontal walls
    if v[1] > 0:
        t_h_wall = (screen.get_height() - radius - p1.y + 1) / v[1]
    elif v[1] < 0:
        t_h_wall = (- p1.y + radius + 1) / v[1]
    else:
        t_h_wall = float('inf')  # No vertical movement
    print(t_v_wall, t_h_wall)
    
    if discriminant < 0:
        # No real solution means no collision
        return float('inf')

    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)

    
    # We want the smallest positive time
    if t1 > 0 and t2 > 0:
        return min(t1, t2)
    elif t1 > 0:
        return t1
    elif t2 > 0:
        return t2
    else:
        return float('inf')

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")

    overlap(player_pos, stationary_pos)
    t_c = collision_time(player_pos, stationary_pos)
    print(t_c)

    pygame.draw.circle(screen, "red", player_pos, radius)
    pygame.draw.circle(screen, "blue", stationary_pos, radius)

    y_inter = player_pos.y - np.sqrt(v[0]**2 + v[1]**2)
    normalized = np.array(v)/np.sqrt(v[0]**2 + v[1]**2)
    length = 1000
    
    end_coord = (player_pos.x + normalized[0] * length, player_pos.y + normalized[1] * length)
    #print(end_coord)
    pygame.draw.line(screen, "black", player_pos, end_coord)

    player_pos.x = player_pos.x + v[0] * dt
    player_pos.y = player_pos.y + v[1] * dt

    if player_pos.x > screen.get_width() - radius:
        player_pos.x = 2 * (screen.get_width() - radius) - player_pos.x
        v[0] = -v[0]
    elif player_pos.x < 0 + radius:
        player_pos.x = 2 * (radius) - player_pos.x
        v[0] = -v[0]

    if player_pos.y > screen.get_height() - radius:
        player_pos.y = 2 * (screen.get_height() - radius) - player_pos.y
        v[1] = -v[1]
    elif player_pos.y < 0 + radius:
        player_pos.y = 2 * (radius) - player_pos.y
        v[1] = -v[1]

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        dy = - 300 * dt
        if player_pos.y + dy >= 0 + radius:
            player_pos.y += dy
    if keys[pygame.K_s]:
        dy = 300 * dt
        if player_pos.y + dy <= screen.get_height() - radius:
            player_pos.y += dy
    if keys[pygame.K_a]:
        dx = - 300 * dt
        if player_pos.x + dx >= 0:
            player_pos.x += dx
    if keys[pygame.K_d]:
        dx = 300 * dt
        if player_pos.x + dx <= screen.get_width() + radius:
            player_pos.x += dx
    if keys[pygame.K_r]:
        v = [np.random.uniform(-100, 100), np.random.uniform(-100, 100)]
    if keys[pygame.K_t]:
        if t_c == float("inf"):
            pass
        player_pos.x = player_pos.x + v[0]*t_c
        player_pos.y = player_pos.y + v[1]*t_c

    # renders the screen
    pygame.display.flip()

    dt = clock.tick(60) / 1000

pygame.quit()
