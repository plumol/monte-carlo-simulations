import pandas as pd
import numpy as np
import pygame

pygame.init()

screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
bounding_box = pygame.Rect(screen.get_width()/2, screen.get_height()/2, 400, 1)
bounding_box.center = (screen.get_width()/2, screen.get_height()/2)

tick_speed = 10
running = True

moves = pd.read_csv("accelerated_first_chain.csv")
print(len(moves))

count = 0
print(moves.iloc[count])
n_particles = len(moves.iloc[count])
diameter = bounding_box.width/(2*n_particles)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")
    pygame.draw.rect(screen, "black", bounding_box, 1)

    for positions in moves.iloc[count]:
        #print(positions)
        pygame.draw.circle(screen, "red", (bounding_box.left + positions, screen.get_height()/2), diameter/2)


    pygame.display.flip()
    clock.tick(tick_speed)


    count += 1
    if count >= len(moves):
        count = 0


pygame.quit()


