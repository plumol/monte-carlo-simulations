import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

pygame.init()

screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
dt = 0
tick_speed = 1

m = 250

class Particle(pygame.Rect):
    
    def __init__(self, color, radius, width = 0, init_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2), moveset = "normal", pbc = False, bounding_box = None, h_i = 0) -> None:

        self.color = color
        self.pos = init_pos
        self.radius = radius
        self.width = width
        self.pbc = pbc
        self.moveset = moveset
        self.bounding_box = bounding_box
        self.h_i = h_i

        self.moves = []
        self.accepted = 0
        self.total = 0

        self.theta = np.random.uniform(0, 2 * np.pi)
        self.v = [0.0, 0.0]
        self.render()

        # this line is supposed to be an initial movement, but it doesn't actually move until you press R
        #self.check_valid_move(self.dx, self.dy)

    def move(self, pdx, pdy, reset = False):

        self.pos.x += pdx
        self.pos.y += pdy

        

    def render(self, type=None):
        """
        Renders particles in a box. 
        Type: "circular": renders particles along a circular path
        """
        
        if type == "circular":
            r = (self.bounding_box.width + self.radius)/(2*np.pi)
            #r = 200
            x = r * np.sin((self.pos.x - screen.get_width())/r) + screen.get_width()/2
            y = r * np.cos((self.pos.x - screen.get_width())/r) + screen.get_height()/2
            
            pygame.draw.circle(screen, self.color, (x,y), self.radius, self.width)
        else:
            
            pygame.draw.circle(screen, self.color, self.pos, self.radius, self.width)
    
    def is_collision(self, other_particle):
        # distance = np.sqrt((self.pos.x - other_particle.pos.x)**2 + (self.pos.y - other_particle.pos.y)**2)
        # return distance < self.radius + other_particle.radius and distance > 0
    
        distance = np.abs((self.pos[0] - other_particle.pos[0] + self.bounding_box.width/2)%self.bounding_box.width - self.bounding_box.width/2)
        return distance < self.radius + other_particle.radius - 0.000000001 and distance > 0

            
    def propose_new_move(self):
        if self.moveset == "normal":
            self.total += 1
            self.accepted += 1
            pdx = np.random.uniform(0, 100)
            pdy = np.random.uniform(0, 0)
            # #valid_move = pygame.Rect(0, 0, width=self.radius, height=self.radius)
            # #valid_move.center = self.center
            # #pygame.draw.rect(screen, "red", valid_move, width=3)
            # if pdx**2 + pdy**2 >  (0.3*self.radius)**2:
            #     self.accepted -= 1
            #     pdx = 0
            #     pdy = 0
            pdx = 1
        elif self.moveset == "random":
            self.total += 1
            self.accepted += 1
            pdx = np.random.uniform(self.bounding_box.left + self.radius, self.bounding_box.right - self.radius)
            pdy = np.random.uniform(self.bounding_box.top + self.radius, self.bounding_box.bottom - self.radius)
        return pdx, pdy

    def check_valid_move(self):
        
        if self.pos.x < self.bounding_box.left:
            return (False, 0)
        elif self.pos.x > self.bounding_box.right:
            return (False, 1)
        else:
            return (True, 2)
        
    def draw_lines(self):
        if len(self.moves) > 1:
            pygame.draw.lines(screen, "black", False, self.moves[:], 1)


colors = ["red", "blue", "black", "green", "yellow", "pink", "brown",]

class Simulation():

    def __init__(self, sampling_method, n_trials, bounding_box_size=200, n_particles=4, spawning_protocol="random"):
        """
        Initializes a Monte Carlo simulation of hard disks within a bounding box. 

        Sampling methods: "direct", "markov", "event"
        Spawning protocols: "random", "uniform"
            Random spawning leads to randomly intiialized particles within the bounding box, may have a specific limit based
            on particle radius and number of particles
            Uniform spawning leads to uniformly initialized particles in a row, column formation.
        """
        self.x_pos = []
        self.y_pos = []
        self.particle_list = []
        self.structure_factors = []
        self.events = []
        self.dt = 0
        self.running = True
        self.trials = n_trials
        self.sm = sampling_method
        self.diameter = 3

        self.mean = (bounding_box_size - n_particles*2*self.diameter)/n_particles

        if sampling_method == "direct":
            self.sampling_method = self.direct_sampling_sequence
            moveset = "random"
        elif sampling_method == "markov":
            self.sampling_method = self.markov_chain_sequence
            moveset = "normal"
        elif sampling_method == "event":
            self.sampling_method = self.event_chain_sequence
            moveset = "normal"
        elif sampling_method == "event_ff":
            self.sampling_method = self.event_chain_ff_sequence_nr
            moveset = "normal"

        self.rect_value = pygame.Rect(screen.get_width()/4, screen.get_height()/4, bounding_box_size, 1)
        self.rect_value.center = (screen.get_width()/2, screen.get_height()/2)

        self.populate_spawning(n_particles, 5, 3, bounding_box=self.rect_value, moveset=moveset, spawning_protocol=spawning_protocol)
        #self.active_idx = np.random.randint(0, len(self.particle_list))
        self.active_idx = len(self.particle_list) - 1
        #self.particle_list[len(self.particle_list)-1].h_i = 250
        self.particle_list[0].h_i = self.mean

    #print(particle_list)
    

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
                    # init_y  = np.random.uniform(bounding_box.top + radius + 0.1, bounding_box.bottom - radius - 0.1)
                    init_y = screen.get_height()/2
                elif spawning_protocol == "uniform":
                    init_x = bounding_box.left + (2.001 * radius * current_column) - radius
                    # init_x = bounding_box.left + (bounding_box.width/n * current_column) - radius
                    # init_y = bounding_box.top + (2.05 * radius * current_row) - radius
                    init_y = screen.get_height()/2
                particle = Particle("red", radius, width=width, init_pos=pygame.Vector2(init_x, init_y), 
                                    moveset=moveset, bounding_box=self.rect_value, h_i=(len(self.particle_list)+1)*radius*2)
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
    
    def render(self, type=None):
        if type == "circular":
            r = (self.rect_value.width + self.particle_list[0].radius)/(2*np.pi)
            pygame.draw.circle(screen, "black", (screen.get_width()/2, screen.get_height()/2), r, 2)
        else:
            pygame.draw.rect(screen, "black", self.rect_value, 1)
        for particle in self.particle_list:
                particle.render(type)

    def structure_factor(self):
        q = (2*np.pi)/self.rect_value.width

        if self.sm == "md":
            structure = 1/len(self.particle_list) * np.sum(np.array([np.exp(q*1j * particle.pos.x) for particle in self.particle_list]))**2
        
        elif self.sm == "markov":
            structure = 1/len(self.x_pos) * np.sum(np.array([np.exp(q*1j * position) for position in self.x_pos]))**2
            
        elif self.sm == "event" or self.sm == "event_ff":
            structure = 1/len(self.x_pos) * np.abs(np.sum([np.exp(q*1j * np.array(self.x_pos))]))**2

        self.structure_factors.append(structure)

    def save_structure_factors(self, file_name):
        df = pd.DataFrame(self.structure_factors, columns=["sf"])
        df.to_csv(file_name)

    def save_positions(self, file_name):
        df = pd.DataFrame(zip(self.x_pos, self.y_pos), columns=["x", "y"])
        df.to_csv(file_name)

    def markov_chain_sequence(self, particles):
        k = np.random.randint(0, len(particles))
        idx, idy = particles[k].pos.x, particles[k].pos.y
        pdx, pdy = particles[k].propose_new_move()
        particles[k].move(pdx, 0)
        valid, side = particles[k].check_valid_move()
        if not valid:
            if side == 0:
                particles[k].move(self.rect_value.width , 0)
                pdx += self.rect_value.width 
            elif side == 1:
                particles[k].move(-self.rect_value.width , 0)
                pdx += -self.rect_value.width 
            #particles[k].accepted -= 1
            #print("STOP")
            
            #valid = False
        for particle in particles:
            if particles[k].is_collision(particle):
                # setting random colors per collision
                #rand = np.random.randint(0, len(colors))
                #particle.color = colors[rand]
                particles[k].accepted -= 1
                particles[k].move(-pdx, 0)
                #valid, side = particles[k].check_valid_move()
                #print(valid)
        
        #particle_list[k].moves.append(particle_list[k].pos[:])
        #particle_list[0].draw_lines()
        # sanity check for collisions
        # for particle in particles:
        #     if particles[k].is_collision(particle):
        #         print("COLLIDED INVALID")
        
        self.x_pos.append(particles[k].pos.x)
        self.y_pos.append(particles[k].pos.y)

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
        v = [1, 0]
        # v = [1, 1]
        particles[k].v[0], particles[k].v[1] = v[0], 0

        # tau_chain = 400
        tau_chain = np.random.exponential(10)
        while tau_chain > 0:
            # print("initial velocity", particles[k].v)
            # print("tau chain", tau_chain)

            # initialize all particles to be infinite colliding time away, stored as (idx, time)

            # we can find next particle by starting at the next idx k+1 mod len(particles), correctly gives (last, first) pair
            next_idx = (k+1)%len(particles)
            #print(k, next_idx)
            # particle is the STATIONARY particle
            particle = particles[next_idx]
            # pairwise collision time collisions
                
            # pass collision time calculation for the same particle

            # calculating collision times for PBC, if MOVING.x < STATIONARY.x, calculate normal
            # else if MOVING.X > STATIONARY.x, meaning it would have to wrap around due to PBC, PRETEND next collision is in the next box over
            if particles[k].pos.x < particle.pos.x:
                dx = particle.pos.x - particles[k].pos.x
            else:
                dx = (particle.pos.x + self.rect_value.width)  - particles[k].pos.x
            dy = particle.pos.y - particles[k].pos.y

            # MAIN COLLISION TIME CALCULATION
            a = (particles[k].v[0]**2 + particles[k].v[1]**2)
            b = 2 * -( particles[k].v[0] * dx + particles[k].v[1] * dy)
            c = dx**2 + dy**2 - (particle.radius + particles[k].radius)**2

            # if discriminant < 0, no real roots and no collisions, default is already set to inf
            if b**2 - 4 * a * c < 0:
                colliding_times = float("inf")
            # quadratic equation
            else:
                t_1 = (-b + np.sqrt(b**2 -4 * a * c))/(2*a)
                t_2 = (-b - np.sqrt(b**2 -4 * a * c))/(2*a)
                colliding_times = min((t for t in (t_1, t_2) if t >= 0), default=float("inf"))

            #print(a, b, c)
            # print(t1, t2)
            # 
            
            # print("colliding times", colliding_times)

            # remove any that are currently colliding bc of small floating point precision errors:
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         del colliding_times[idx]
            # check minimum collding time > 0 
            #print(colliding_times, tau_chain)
            colliding_times = min(colliding_times, tau_chain)
            # print("min_collide", min_colliding_time)
            
            if colliding_times < tau_chain:
                # move to collision
                pdx = particles[k].v[0] * colliding_times
                pdy = particles[k].v[1] * colliding_times
                particles[k].move(pdx, 0)
            #pygame.draw.circle(screen, "black", (particles[k].pos.x + pdx, particles[k].pos.y + pdy), particles[k].radius, 3)

            # print(colliding_times)
            #print(particles[k].pos)
            # PBC conditions 
                valid, side = particles[k].check_valid_move()
                if not valid:
                    #print("help")
                    if side == 1:
                        particles[k].move(-self.rect_value.width , 0)
                        pdx += -self.rect_value.width

            # reset current particle velocity to 0
            # update particle idx to the new particle
            # update particle velocity 
                particles[k].v[0], particles[k].v[1] = 0, 0
                k = next_idx
                particles[k].v[0], particles[k].v[1] = v[0], 0
            else:
                # move to x + v * tau_chain
                pdx = particles[k].v[0] * tau_chain
                pdy = particles[k].v[1] * tau_chain
                particles[k].move(pdx, pdy)

                # PBC conditions
                valid, side = particles[k].check_valid_move()
                if not valid:
                    #print("help")
                    if side == 1:
                        particles[k].move(-self.rect_value.width , 0)
                        pdx += -self.rect_value.width 
            
            tau_chain -= colliding_times
            print(tau_chain)

            # sanity check for collisions
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         print("COLLIDED INVALID")
            #         print(particles[k].pos.x - self.rect_value.left, particles[idx].pos.x - self.rect_value.left)

        
        # reset all velocities to zero and store positions
        for particle in particles:
            self.x_pos.append(particle.pos.x)
            self.y_pos.append(particle.pos.y)
            particle.accepted += 1
            particle.total += 1
            particle.v[0], particle.v[1] = 0, 0

    def event_chain_ff_sequence_nr(self, particles):
        #k = np.random.randint(0, len(particles))
        k = self.active_idx
        v = [1, 0]
        # v = [1, 1]
        particles[k].v[0], particles[k].v[1] = v[0], 0

        events = 0
        if len(self.events) < 1:
            tau_chain = self.rect_value.width*(len(particles) + 1)/4
        else:
            tau_chain = self.rect_value.width
        # tau_chain = 400
        print("tau chain", tau_chain)
        # P_T = len(particles)/(self.rect_value.width - len(particles)*particles[k].radius*2)
        #print(P_T)
        
        while tau_chain > 0:
            # sampled_u = np.random.uniform(0, 1)
            # P_T = np.random.exponential(1)
            # x_ff = -1/P_T * np.log(sampled_u)
            #x_ff = np.random.exponential(self.mean)
            x_ff = np.random.exponential(particles[k].h_i)
            
            #print(x_ff)
            
            # print("initial velocity", particles[k].v)
            # print("tau chain", tau_chain)

            # initialize all particles to be infinite colliding time away, stored as (idx, time)

            next_idx = (k+1)%len(particles)
            prev_idx = (k-1)%len(particles)
            #print(k, next_idx)
            # particle is the STATIONARY particle
            particle = particles[next_idx]
            # pairwise collision time collisions
                
                # pass collision time calculation for the same particle

            # calculating collision times for PBC, if MOVING.x < STATIONARY.x, calculate normal
            # else if MOVING.X > STATIONARY.x, meaning it would have to wrap around due to PBC, PRETEND next collision is in the next box over
            if particles[k].pos.x < particle.pos.x:
                dx = particle.pos.x - particles[k].pos.x - 2 * particles[k].radius
            else:
                dx = (particle.pos.x + self.rect_value.width)  - particles[k].pos.x - 2 * particles[k].radius

            # dy = particle.pos.y - particles[k].pos.y

            # a = (particles[k].v[0]**2 + particles[k].v[1]**2)
            # b = 2 * -( particles[k].v[0] * dx + particles[k].v[1] * dy)
            # c = dx**2 + dy**2 - (particle.radius + particles[k].radius)**2

            # # if discriminant < 0, no real roots and no collisions, default is already set to inf
            # if b**2 - 4 * a * c < 0:
            #     colliding_times = float("inf")
            # # quadratic equation
            # else:
            #     t_1 = (-b + np.sqrt(b**2 -4 * a * c))/(2*a)
            #     t_2 = (-b - np.sqrt(b**2 -4 * a * c))/(2*a)
            #     colliding_times = min((t for t in (t_1, t_2) if t >= 0), default=float("inf"))

            #print(a, b, c)
            # print(t1, t2)
            # 
            # print("colliding times", colliding_times)

            # remove any that are currently colliding:
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         del colliding_times[idx]
            # check minimum collding time > 0 

            # TODO: solve for x_ff_t
            colliding_times = dx/particles[k].v[0]
            x_ff_t = x_ff/particles[k].v[0]
            #print(x_ff_t, colliding_times)
            #print("hi")
            # if k == len(self.particle_list) - 1:
            #     print(x_ff_t, colliding_times, self.particle_list[k].h_i)
            if x_ff_t < colliding_times:
                colliding_times = x_ff_t
                lifted_particle = prev_idx
            else:
                lifted_particle = next_idx
            
            # print("min_collide", min_colliding_time)
            #print(colliding_times)
            print("collide ", colliding_times, (k, lifted_particle), tau_chain, particles[k].pos[0] - 1280/4, particles[k].pos[0] + colliding_times - 1280/4)

            if colliding_times < tau_chain:
                # move to collision
                pdx = particles[k].v[0] * colliding_times
                pdy = particles[k].v[1] * colliding_times
                #pygame.draw.circle(screen, "black", (particles[k].pos.x + pdx, particles[k].pos.y + pdy), particles[k].radius, 3)

                # print(colliding_times)
                #print(particles[k].pos)
                valid, side = particles[k].check_valid_move()
                if not valid:
                    #print("help")
                    if side == 1:
                        particles[k].move(-self.rect_value.width , 0)
                        pdx += -self.rect_value.width

                screen.fill("white")
                self.render()
                pygame.display.flip()
                # self.render()
                # pygame.display.flip()
                # reset current particle velocity to 0
                # update particle idx to the new particle
                # update particle velocity 
                particles[k].v[0], particles[k].v[1] = 0, 0
                k = lifted_particle
                particles[k].v[0], particles[k].v[1] = v[0], 0
            else:
                # move to x + v * tau_chain
                pdx = particles[k].v[0] * tau_chain
                pdy = particles[k].v[1] * tau_chain
                particles[k].move(pdx, pdy)
                valid, side = particles[k].check_valid_move()
                if not valid:
                    #print("help")
                    
                    if side == 1:
                        particles[k].move(-self.rect_value.width , 0)
                        pdx += -self.rect_value.width 
                screen.fill("white")
                self.render()
                pygame.display.flip()
                
            tau_chain -= colliding_times
            events+=1
            

            # # sanity check for collisions
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         print("COLLIDED INVALID")
            #         print(particles[k].pos.x - self.rect_value.left, particles[idx].pos.x - self.rect_value.left)

        self.active_idx = k
        self.events.append(events)
        # reset all velocities to zero
        for particle in particles:
            self.x_pos.append(particle.pos.x)
            self.y_pos.append(particle.pos.y)
            particle.accepted += 1
            particle.total += 1
            particle.v[0], particle.v[1] = 0, 0

    def event_chain_ff_sequence_wr(self, particles):
        k = np.random.randint(0, len(particles))
        #k = self.active_idx
        v = [1, 0]
        # v = [1, 1]
        particles[k].v[0], particles[k].v[1] = v[0], 0

        tau_chain = 400
        # P_T = len(particles)/(self.rect_value.width - len(particles)*particles[k].radius*2)
        #print(P_T)
        
        while tau_chain > 0:
            # sampled_u = np.random.uniform(0, 1)
            # P_T = np.random.exponential(1)
            # x_ff = -1/P_T * np.log(sampled_u)
            x_ff = np.random.exponential(self.mean)
            #print(x_ff)
            
            # print("initial velocity", particles[k].v)
            # print("tau chain", tau_chain)

            # initialize all particles to be infinite colliding time away, stored as (idx, time)

            next_idx = (k+1)%len(particles)
            prev_idx = (k-1)%len(particles)
            #print(k, next_idx)
            # particle is the STATIONARY particle
            particle = particles[next_idx]
            # pairwise collision time collisions
                
                # pass collision time calculation for the same particle

            # calculating collision times for PBC, if MOVING.x < STATIONARY.x, calculate normal
            # else if MOVING.X > STATIONARY.x, meaning it would have to wrap around due to PBC, PRETEND next collision is in the next box over
            if particles[k].pos.x < particle.pos.x:
                dx = particle.pos.x - particles[k].pos.x
            else:
                dx = (particle.pos.x + self.rect_value.width)  - particles[k].pos.x

            dy = particle.pos.y - particles[k].pos.y

            a = (particles[k].v[0]**2 + particles[k].v[1]**2)
            b = 2 * -( particles[k].v[0] * dx + particles[k].v[1] * dy)
            c = dx**2 + dy**2 - (particle.radius + particles[k].radius)**2

            # if discriminant < 0, no real roots and no collisions, default is already set to inf
            if b**2 - 4 * a * c < 0:
                colliding_times = float("inf")
            # quadratic equation
            else:
                t_1 = (-b + np.sqrt(b**2 -4 * a * c))/(2*a)
                t_2 = (-b - np.sqrt(b**2 -4 * a * c))/(2*a)
                colliding_times = min((t for t in (t_1, t_2) if t >= 0), default=float("inf"))

            #print(a, b, c)
            # print(t1, t2)
            # 
            # print("colliding times", colliding_times)

            # remove any that are currently colliding:
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         del colliding_times[idx]
            # check minimum collding time > 0 

            # TODO: solve for x_ff_t
            x_ff_t = x_ff/particles[k].v[0]
            #print(x_ff_t, colliding_times)
            
            if x_ff_t < colliding_times:
                colliding_times = x_ff_t
                lifted_particle = prev_idx
            else:
                lifted_particle = next_idx
            
            # print("min_collide", min_colliding_time)
            #print(colliding_times)

            if colliding_times < tau_chain:
                # move to collision
                pdx = particles[k].v[0] * colliding_times
                pdy = particles[k].v[1] * colliding_times
                #pygame.draw.circle(screen, "black", (particles[k].pos.x + pdx, particles[k].pos.y + pdy), particles[k].radius, 3)

                # print(colliding_times)
                #print(particles[k].pos)
                valid, side = particles[k].check_valid_move()
                if not valid:
                    #print("help")
                    if side == 1:
                        particles[k].move(-self.rect_value.width , 0)
                        pdx += -self.rect_value.width

                
                # reset current particle velocity to 0
                # update particle idx to the new particle
                # update particle velocity 
                particles[k].v[0], particles[k].v[1] = 0, 0
                k = lifted_particle
                particles[k].v[0], particles[k].v[1] = v[0], 0
            else:
                # move to x + v * tau_chain
                pdx = particles[k].v[0] * tau_chain
                pdy = particles[k].v[1] * tau_chain
                particles[k].move(pdx, pdy)
                valid, side = particles[k].check_valid_move()
                if not valid:
                    #print("help")
                    
                    if side == 1:
                        particles[k].move(-self.rect_value.width , 0)
                        pdx += -self.rect_value.width 
                
            tau_chain -= colliding_times

            # # sanity check for collisions
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         print("COLLIDED INVALID")
            #         print(particles[k].pos.x - self.rect_value.left, particles[idx].pos.x - self.rect_value.left)

        #self.active_idx = k
        # reset all velocities to zero
        for particle in particles:
            self.x_pos.append(particle.pos.x)
            self.y_pos.append(particle.pos.y)
            particle.accepted += 1
            particle.total += 1
            particle.v[0], particle.v[1] = 0, 0

    def simulate(self):
        print("Starting simulation.")
        count = 0

        tic = time.perf_counter()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # screen.fill("white")
            # self.render()
            # pygame.display.flip()

            
            #bounding_box = pygame.draw.rect(screen, "black", self.rect_value, 1)
            
            keys = pygame.key.get_pressed()

            # TODO: given n particles, choose one at random
            # generate a random move x, y
            # check collisions and boundary conditions, if accept then move, if reject then don't move
            # update positions
            

            count += 1
            
            self.sampling_method(particles=self.particle_list)

            if count == 1:
                for particle in self.particle_list:
                    particle.h_i = self.mean

            # renders the screen
            #pygame.time.delay(1000)

            self.dt = 1 / 100
            clock.tick(tick_speed)
            
            
            if count % 10000 == 0:
                print(f"Acceptance: {[particle.accepted/particle.total for particle in self.particle_list]}")
                print(count)
                toc = time.perf_counter()
                print(f"Time taken: {toc - tic}")
                tic = time.perf_counter()
                self.structure_factor()
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

# markov = Simulation("markov", 100000, 400, n_particles=4, spawning_protocol="uniform")
# m_x_pos, m_y_pos = markov.simulate()

# ecmc = Simulation("event", 2, 400, n_particles=20, spawning_protocol="uniform")
# e_x_pos, e_y_pos = ecmc.simulate()

ecmc_ff = Simulation("event_ff", 2, 400, n_particles=16, spawning_protocol="uniform")
e_ff_x_pos, e_ff_y_pos = ecmc_ff.simulate()

# SAVING
# markov.save_positions("markov_sampling_1mil-0620.csv")
# ecmc.save_positions("ecmc_1mil-1D.csv")
# ecmc_ff.save_positions("ecmc_ff_1mil-1D.csv")

# markov.save_structure_factors("markov_sf_1m.csv")
# ecmc.save_structure_factors("ecmc_sf_1m.csv")
# ecmc_ff.save_structure_factors("ecmc_ff_sf_1m.csv")

# load saved CSV files
# markov_df = pd.read_csv("markov_sampling_1mil-0620.csv")
# m_x_pos = markov_df["x"].to_list()
# m_y_pos = markov_df["y"].to_list()

# ecmc_df = pd.read_csv("ecmc_1mil-1D.csv")
# e_x_pos = ecmc_df["x"].to_list()
# e_y_pos = ecmc_df["y"].to_list()

# ecmc_ff_df = pd.read_csv("ecmc_ff_1mil-1D.csv")
# e_ff_x_pos = ecmc_ff_df["x"].to_list()
# e_ff_y_pos = ecmc_ff_df["y"].to_list()

# load saved Structure Factor CSV
# markov_sf_df = pd.read_csv("markov_sf_1m.csv")
# markov_sf = markov_sf_df["sf"].to_list()

# ecmc_sf_df = pd.read_csv("ecmc_sf_1m.csv")
# ecmc_sf = ecmc_sf_df["sf"].to_list()

# ecmc_ff_sf_df = pd.read_csv("ecmc_ff_sf_1m.csv")
# ecmc_ff_sf = ecmc_ff_sf_df["sf"].to_list()

# x pos pdf
#fig = plt.figure()

# plt.hist(np.array(m_x_pos) - markov.rect_value.left, 40, density=True, histtype='step', label="markov x")

# plt.hist(np.array(e_x_pos) - ecmc.rect_value.left, 40, density=True, histtype='step', label="ecmc x")

# plt.hist(np.array(e_ff_x_pos) - ecmc_ff.rect_value.left, 40, density=True, histtype='step', label="ecmc ff x")

#plt.show()

# structure factors
#fig = plt.figure()
# plt.plot([10000 * i for i in range(len(markov.structure_factors))], np.array(markov.structure_factors, dtype=complex), label="markov" )
# plt.plot([10000 * i for i in range(len(ecmc.structure_factors))], np.array(ecmc.structure_factors, dtype=complex), label="ecmc")
# plt.plot([10000 * i for i in range(len(ecmc_ff.structure_factors))], np.array(ecmc_ff.structure_factors, dtype=complex), label="ecmc ff")

# after loading SF!
# plt.hist(np.array(markov_sf, dtype=complex), label="markov", density=True )
# plt.hist(np.array(ecmc_sf, dtype=complex), label="ecmc", density=True )
# plt.hist(np.array(ecmc_ff_sf, dtype=complex), label="ecmc ff", density=True )
#plt.title("Structure Factor")

# after loading!
# plt.hist(np.array(d_x_pos) - 260, 40, density=True, histtype='step', label="direct x")
# plt.hist(np.array(d_y_pos) - 540, 40, density=True, histtype='step', label="direct y")

# plt.hist(np.array(m_x_pos) - 260, 40, density=True, histtype='step', label="markov x")
# plt.hist(np.array(m_y_pos) - 540, 40, density=True, histtype='step', label="markov y")

# plt.hist(np.array(e_x_pos) - ecmc.rect_value.left, 40, density=True, histtype='step', label="ecmc x")
# plt.hist(np.array(e_y_pos) - 540, 40, density=True, histtype='step', label="ecmc y")

# plt.hist(np.array(e_ff_x_pos) - ecmc_ff.rect_value.left, 40, density=True, histtype='step', label="ecmc ff x")
# plt.hist(np.array(e_y_pos) - 540, 40, density=True, histtype='step', label="ecmc y")

# plt.xlim(0)
#plt.legend()


#plt.clf()
#fig, ax = plt.subplots()
#hist = 0

#plt.show()
pygame.quit()
