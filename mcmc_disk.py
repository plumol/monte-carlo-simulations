import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

pygame.init()

screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
dt = 0
tick_speed = 2

m = 250

class Particle(pygame.Rect):
    
    def __init__(self, color, radius, width = 0, init_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2), moveset = "normal", pbc = False, bounding_box = None) -> None:

        self.color = color
        self.pos = init_pos
        self.radius = radius
        self.width = width
        self.pbc = pbc
        self.moveset = moveset
        self.bounding_box = bounding_box

        self.moves = []
        self.accepted = 0
        self.total = 0

        self.theta = np.random.uniform(0, 2 * np.pi)
        self.v = [0.0, 0.0]
        self.render()

        # this line is supposed to be an initial movement, but it doesn't actually move until you press R
        #self.check_valid_move(self.dx, self.dy)

    def move(self, pdx, pdy, reset = False):

        if self.moveset == "normal":
            self.pos.x += pdx
            self.pos.y += pdy
        elif self.moveset == "random" or reset:
            self.pos.x = pdx
            self.pos.y = pdy

        pass

    def render(self):
        pygame.draw.circle(screen, self.color, self.pos, self.radius, self.width)
    
    def is_collision(self, other_particle):
        distance = np.sqrt((self.pos.x - other_particle.pos.x)**2 + (self.pos.y - other_particle.pos.y)**2)
        return distance < self.radius + other_particle.radius and distance > 0
            
    def propose_new_move(self):
        if self.moveset == "normal":
            self.total += 1
            self.accepted += 1
            pdx = np.random.uniform(-self.radius, self.radius)
            pdy = np.random.uniform(-self.radius, self.radius)
            #valid_move = pygame.Rect(0, 0, width=self.radius, height=self.radius)
            #valid_move.center = self.center
            #pygame.draw.rect(screen, "red", valid_move, width=3)
            if pdx**2 + pdy**2 > 0.5 * self.radius**2:
                self.accepted -= 1
                pdx = 0
                pdy = 0
        elif self.moveset == "random":
            self.total += 1
            self.accepted += 1
            pdx = np.random.uniform(self.bounding_box.left + self.radius, self.bounding_box.right - self.radius)
            pdy = np.random.uniform(self.bounding_box.top + self.radius, self.bounding_box.bottom - self.radius)
        return pdx, pdy

    def check_valid_move(self):
        
        if self.pos.x - self.radius < self.bounding_box.left or self.pos.x + self.radius > self.bounding_box.right:
            return False
        elif self.pos.y - self.radius < self.bounding_box.top or self.pos.y + self.radius > self.bounding_box.bottom:
            return False
        else:
            return True
        
    def draw_lines(self):
        if len(self.moves) > 1:
            pygame.draw.lines(screen, "black", False, self.moves[:], 1)


colors = ["red", "blue", "black", "green", "yellow", "pink", "brown",]

class Simulation():

    def __init__(self, sampling_method, n_trials, bounding_box_size=200, ):
        self.x_pos = []
        self.y_pos = []
        self.particle_list = []
        self.dt = 0
        self.running = True
        self.trials = n_trials

        if sampling_method == "direct":
            self.sampling_method = self.direct_sampling_sequence
            moveset = "random"
        elif sampling_method == "markov":
            self.sampling_method = self.markov_chain_sequence
            moveset = "normal"
        elif sampling_method == "event":
            self.sampling_method = self.event_chain_sequence
            moveset = "normal"

        self.rect_value = pygame.Rect(screen.get_width()/4, screen.get_height()/4, 200, 200)
        self.rect_value.center = (screen.get_width()/2, screen.get_height()/2)

        self.populate_spawning(2, 24, 3, bounding_box=self.rect_value, moveset=moveset, spawning_protocol="uniform")
        

    #print(particle_list)
    def save_positions(self, file_name):
        df = pd.DataFrame(self.x_pos, self.y_pos)
        df.to_csv(file_name)

    def populate_spawning(self, n, radius, width, bounding_box, moveset="random", spawning_protocol="random"):
        # spawn balls within the bounding box
        # REQUIREMENTS: balls cannot overlap each other, balls must spawn in locations with the bounding box
        # TODO: implement a fixed spawning protocol
        max_balls_per_row = bounding_box.width // (radius * 2)
        current_row = 1
        current_column = 1
        for i in range(n):
            while True:
                if spawning_protocol == "random":
                    init_x = np.random.uniform(bounding_box.left + radius + 0.1, bounding_box.right - radius - 0.1)
                    init_y  = np.random.uniform(bounding_box.top + radius + 0.1, bounding_box.bottom - radius - 0.1)
                elif spawning_protocol == "uniform":
                    init_x = bounding_box.left + (2.05 * radius * current_column) - radius
                    init_y = bounding_box.top + (2.05 * radius * current_row) - radius
                particle = Particle("red", radius, width=width, init_pos=pygame.Vector2(init_x, init_y), moveset=moveset, bounding_box=self.rect_value)
                for existing_particle in self.particle_list:
                    if particle.is_collision(existing_particle):
                        break
                else:
                    self.particle_list.append(particle)
                    #print(len(particle_list))
                    current_column += 1
                    if current_column > max_balls_per_row:
                        current_row += 1
                        current_column = 1
                    break

    def markov_chain_sequence(self, particles):
        k = np.random.randint(0, len(particles))
        idx, idy = particles[k].pos.x, particles[k].pos.y
        pdx, pdy = particles[k].propose_new_move()
        particles[k].move(pdx, pdy)
        valid = True
        if not particles[k].check_valid_move():
            particles[k].accepted -= 1
            #print("STOP")
            particles[k].move(-pdx, -pdy)
            #valid = False
        for particle in particles:
            if particles[k].is_collision(particle):
                # setting random colors per collision
                #rand = np.random.randint(0, len(colors))
                #particle.color = colors[rand]
                particles[k].accepted -= 1
                particles[k].move(-pdx, -pdy)
        
        #particle_list[k].moves.append(particle_list[k].pos[:])
        #particle_list[0].draw_lines()
        
        self.x_pos.append(self.particle_list[k].pos.x)
        self.y_pos.append(self.particle_list[k].pos.y)

    def direct_sampling_sequence(self, particles):
        while True:
            accepted = []
            rejected = False

            for i in range(len(particles)):
                pdx, pdy = particles[i].propose_new_move()
                particles[i].move(pdx, pdy)
                for j in range(i):
                    if particles[i].is_collision(particles[j]):
                        rejected = True
                        break
                if rejected:
                    break
                # counts valid moves
                accepted.append((pdx, pdy))
            
            # once all moves are valid, then we accept the configuration
            if len(accepted) == len(particles):
                for move in accepted:
                    self.x_pos.append(move[0])
                    self.y_pos.append(move[1])
                break

    def event_chain_sequence(self, particles):
        k = np.random.randint(0, len(particles))
        v = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        # v = [1, 1]
        particles[k].v[0], particles[k].v[1] = v[0], v[1]

        tau_chain = np.random.exponential(10)
        while tau_chain > 0:
            print("initial velocity", particles[k].v)
            print("tau chain", tau_chain)
            # initialize all particles to be infinite colliding time away
            colliding_times = [(i, float("inf")) for i in range(len(particles))]
            # pairwise collision time collisions
            for idx, particle in enumerate(particles):
                
                # pass collision time calculation for the same particle
                if idx == k:
                    continue
                dx = particle.pos.x - particles[k].pos.x
                dy = particle.pos.y - particles[k].pos.y

                a = (particles[k].v[0]**2 + particles[k].v[1]**2)
                b = 2 * ( -particles[k].v[0] * dx - particles[k].v[1] * dy)
                c = dx**2 + dy**2 - (particle.radius + particles[k].radius)**2

                # if discriminant < 0, no real roots and no collisions, default is already set to inf
                if b**2 - 4 * a * c < 0:
                    continue
                # quadratic equation
                t_1 = (-b + np.sqrt(b**2 -4 * a * c))/(2*a)
                t_2 = (-b - np.sqrt(b**2 -4 * a * c))/(2*a)

                #print(a, b, c)
                # print(t1, t2)
                # 
                t_collide_x = min((t for t in (t_1, t_2) if t >= 0), default=float("inf"))
                colliding_times[idx] = (idx, t_collide_x)
                
            # calculate wall collision
            if particles[k].v[0] > 0:
                t_h_wall = (self.rect_value.right - particles[k].radius - particles[k].pos.x) / particles[k].v[0]
            elif particles[k].v[0] < 0:
                t_h_wall = (self.rect_value.left - particles[k].pos.x + particles[k].radius) / particles[k].v[0]
            else:
                t_h_wall = float('inf')  # No horizontal movement

            # Calculate time to hit the horizontal walls
            if particles[k].v[1] > 0:
                t_v_wall = (self.rect_value.bottom - particles[k].radius - particles[k].pos.y + 1) / particles[k].v[1]
            elif particles[k].v[1] < 0:
                t_v_wall = (self.rect_value.top - particles[k].pos.y + particles[k].radius + 1) / particles[k].v[1]
            else:
                t_v_wall = float('inf')  # No vertical movement

            colliding_times.append((len(particles), t_h_wall))
            colliding_times.append((len(particles) + 1, t_v_wall))
            print("colliding times", colliding_times)

            min_colliding_time = min((c_time for c_time in colliding_times if c_time[1] > 0), key=lambda t: t[1])
            print("min_collide", min_colliding_time)

            if min_colliding_time[1] < tau_chain:
                # move to collision
                pdx = particles[k].v[0] * min_colliding_time[1]
                pdy = particles[k].v[1] * min_colliding_time[1]
                particles[k].move(pdx, pdy)

                if not particles[k].check_valid_move():
                    print("AHhhhhh")

                if min_colliding_time[0] == len(particles):
                    particles[k].v[0] = -particles[k].v[0]
                    # pygame.time.delay(2000)
                    
                elif min_colliding_time[0] == len(particles) + 1:
                    particles[k].v[1] = -particles[k].v[1]
                    # pygame.time.delay(2000)
                    
                else:
                    # reset current particle velocity to 0
                    # update particle idx to the new particle
                    # update particle velocity 
                    print("before transferring", particles[k].v, particles[min_colliding_time[0]].v)
                    particles[k].v[0], particles[k].v[1] = 0, 0
                    old_k = k
                    k = min_colliding_time[0]
                    particles[k].v[0], particles[k].v[1] = v[0], v[1]
                    print(old_k, k)
                    print("after transferring", particles[old_k].v, particles[min_colliding_time[0]].v)
            else:
                # move to x + v * tau_chain
                pdx = particles[k].v[0] * tau_chain
                pdy = particles[k].v[1] * tau_chain
                particles[k].move(pdx, pdy)
                
            tau_chain -= min_colliding_time[1]
        
        # reset all velocities to zero
        for particle in particles:
            particle.v[0], particle.v[1] = 0, 0

    def simulate(self):
        print("Starting simulation.")
        count = 0

        tic = time.perf_counter()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            screen.fill("white")
            
            bounding_box = pygame.draw.rect(screen, "black", self.rect_value, 1)
            
            keys = pygame.key.get_pressed()

            # TODO: given n particles, choose one at random
            # generate a random move x, y
            # check collisions and boundary conditions, if accept then move, if reject then don't move
            # update positions
            for particle in self.particle_list:
                particle.render()
            
            self.sampling_method(particles=self.particle_list)

            # renders the screen
            pygame.display.flip()
            #pygame.time.delay(1000)

            self.dt = 1 / 100
            clock.tick(tick_speed)
            
            count += 1
            if count % 10000 == 0:
                print(f"Acceptance: {[particle.accepted/particle.total for particle in self.particle_list]}")
                print(count)
                toc = time.perf_counter()
                print(f"Time taken: {toc - tic}")
                tic = time.perf_counter()
            if count >= self.trials:
                self.running = False

        print("Finished!")
        return self.x_pos, self.y_pos 


# plt.hist(list(x_pos.keys()), 20, density=True)
# plt.show()

# plt.hist2d(list(x_pos.keys()), list(y_pos.keys()), 20, density=True)
# plt.show()

# x_ticks = np.arange(24, 200 - 24, 24)
# y_ticks = np.arange(24, 200 - 24, 24)
# plt.hist2d(np.array(x_pos) - 540, np.array(y_pos) - 260, 20, density=False)
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)

#pygame.draw.lines(screen, "black", False, particle_list[0].moves, 10)
#direct = Simulation("direct", 1000000, 200)
#d_x_pos, d_y_pos = direct.simulate()

#markov = Simulation("markov", 1000000, 200)
#m_x_pos, m_y_pos = markov.simulate()

ecmc = Simulation("event", 100000, 200)
e_x_pos, e_y_pos = ecmc.simulate()

# direct_df = pd.read_csv("direct_sampling_1mil-0617.csv", names=["x", "y"])
# d_x_pos = direct_df["x"].to_list()
# d_y_pos = direct_df["y"].to_list()

# markov_df = pd.read_csv("markov_sampling_1mil-0617.csv", names=["x", "y"])
# m_x_pos = markov_df["x"].to_list()
# m_y_pos = markov_df["y"].to_list()


#direct.save_positions("direct_sampling_1mil-0617.csv")
#markov.save_positions("markov_sampling_1mil-0617.csv")

# plt.hist(np.array(d_x_pos) - 540, 40, density=True, histtype='step', label="direct x")
# plt.hist(np.array(d_y_pos) - 260, 40, density=True, histtype='step', label="direct y")

# plt.hist(np.array(m_x_pos) - 540, 40, density=True, histtype='step', label="markov x")
# plt.hist(np.array(m_y_pos) - 260, 40, density=True, histtype='step', label="markov y")

plt.legend()


#plt.clf()
#fig, ax = plt.subplots()
#hist = 0

plt.show()
pygame.quit()
