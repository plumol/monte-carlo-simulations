import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

screen_width = 1280
screen_height = 720

class Particle():
    
    def __init__(self, color, radius, width = 0, init_pos = [screen_width/2, screen_height/2], moveset = "normal", pbc = False, bounding_box = None) -> None:

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

        # this line is supposed to be an initial movement, but it doesn't actually move until you press R
        #self.check_valid_move(self.dx, self.dy)

    def move(self, pdx, pdy, reset = False):

        if self.moveset == "normal":
            self.pos[0] += pdx
            self.pos[1] += pdy
        elif self.moveset == "random" or reset:
            self.pos[0] = pdx
            self.pos[1] = pdy

        pass
    
    def is_collision(self, other_particle):
        distance = np.sqrt((self.pos[0] - other_particle.pos[0])**2 + (self.pos[1] - other_particle.pos[1])**2)
        return distance < self.radius + other_particle.radius and distance > 2 * self.radius + 0.00000001
            
    def propose_new_move(self):
        if self.moveset == "normal":
            self.total += 1
            self.accepted += 1
            pdx = np.random.uniform(0, self.radius)
            pdy = 0
            if pdx**2 + pdy**2 > (0.33*self.radius)**2:
                self.accepted -= 1
                pdx = 0
                pdy = 0
        elif self.moveset == "random":
            self.total += 1
            self.accepted += 1
            pdx = np.random.uniform(self.radius, self.bounding_box - self.radius)
            pdy = np.random.uniform(self.radius, self.bounding_box - self.radius)
        return pdx, pdy

    def check_valid_move(self):
        
        if self.pos[0] < 0:
            return (False, 0)
        elif self.pos[0] > self.bounding_box:
            return (False, 1)
        else:
            return (True, 2)
        


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
        self.inner_sf = 0
        self.structure_factors = []
        self.qt = (2*np.pi)/bounding_box_size
        self.dt = 0
        self.running = True
        self.trials = n_trials
        self.sm = sampling_method

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
            self.sampling_method = self.event_chain_ff_sequence
            moveset = "normal"

        self.rect_value = bounding_box_size
        # self.rect_value.center = (screen.get_width()/2, screen.get_height()/2)

        self.populate_spawning(n_particles, 24, 3, bounding_box=self.rect_value, moveset=moveset, spawning_protocol=spawning_protocol)
        

    #print(particle_list)
    

    def populate_spawning(self, n, radius, width, bounding_box, moveset="random", spawning_protocol="random"):
        # spawn balls within the bounding box
        # REQUIREMENTS: balls cannot overlap each other, balls must spawn in locations with the bounding box
        # TODO: implement a fixed spawning protocol
        max_balls_per_row = bounding_box // (radius * 2)
        current_row = 1
        current_column = 1
        for i in range(n):
            while True:
                if spawning_protocol == "random":
                    init_x = np.random.uniform(0 + radius + 0.1, bounding_box - radius - 0.1)
                    # init_y  = np.random.uniform(bounding_box.top + radius + 0.1, bounding_box.bottom - radius - 0.1)
                    # init_y = screen.get_height()/2
                elif spawning_protocol == "uniform":
                    init_x = 0 + (2.05 * radius * current_column) - radius
                    # init_x = 0 + (bounding_box/n * current_column) - radius
                    # init_y = bounding_box.top + (2.05 * radius * current_row) - radius
                    # init_y = screen.get_height()/2
                particle = Particle("red", radius, width=width, init_pos=[init_x, 0], moveset=moveset, bounding_box=self.rect_value)
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

    def structure_factor(self):
        q = (2*np.pi)/self.rect_value

        if self.sm == "md":
            structure = 1/len(self.particle_list) * np.sum(np.array([np.exp(q*1j * particle.pos.x) for particle in self.particle_list]))**2
        
        elif self.sm == "markov":
            structure = 1/len(self.x_pos) * np.abs(np.sum([np.exp(q*1j * np.array(self.x_pos))]))**2
            
        elif self.sm == "event" or self.sm == "event_ff":
            # structure = 1/len(self.x_pos) * np.abs(np.sum([np.exp(q*1j * np.array(self.x_pos))]))**2
            structure = 1/len(self.x_pos) * np.abs(self.inner_sf)**2

        self.structure_factors.append(structure)

    def save_structure_factors(self, file_name):
        df = pd.DataFrame(self.structure_factors, columns=["sf"])
        df.to_csv(file_name)

    def save_positions(self, file_name):
        df = pd.DataFrame(zip(self.x_pos, self.y_pos), columns=["x", "y"])
        df.to_csv(file_name)

    def markov_chain_sequence(self, particles):
        k = np.random.randint(0, len(particles))
        pdx, pdy = particles[k].propose_new_move()
        particles[k].move(pdx, 0)
        valid, side = particles[k].check_valid_move()
        if not valid:
            if side == 0:
                particles[k].move(self.rect_value , 0)
                pdx += self.rect_value 
            elif side == 1:
                particles[k].move(-self.rect_value , 0)
                pdx += -self.rect_value 
            #particles[k].accepted -= 1
            #print("STOP")
            
            #valid = False
        for idx, particle in enumerate(particles):
            if idx == k:
                continue
            if particles[k].is_collision(particle):
                # setting random colors per collision
                #rand = np.random.randint(0, len(colors))
                #particle.color = colors[rand]
                particles[k].accepted -= 1
                particles[k].move(-pdx, 0)
                # valid, side = particles[k].check_valid_move()
                #print(valid)
        
        # sanity check for collisions
        # for particle in particles:
        #     if particles[k].is_collision(particle):
        #         print("COLLIDED INVALID")
        
        self.x_pos.append(particles[k].pos[0])
        # self.y_pos.append(particles[k].pos[1])
        # for particle in particles:
        #     self.x_pos.append(particle.pos[0])


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
        particles[k].v[0] = v[0]

        tau_chain = 10
        while tau_chain > 0:

            # initialize all particles to be infinite colliding time away, stored as (idx, time)

            # we can find next particle by starting at the next idx k+1 mod len(particles), correctly gives (last, first) pair
            next_idx = (k+1)%len(particles)

            # particle is the STATIONARY particle
            particle = particles[next_idx]
            # pairwise collision time collisions
                
            # pass collision time calculation for the same particle

            # calculating collision times for PBC, if MOVING.x < STATIONARY.x, calculate normal
            # else if MOVING.X > STATIONARY.x, meaning it would have to wrap around due to PBC, PRETEND next collision is in the next box over
            if particles[k].pos[0] < particle.pos[0]:
                dx = particle.pos[0] - particles[k].pos[0] - 2 * particles[k].radius
            else:
                dx = (particle.pos[0] + self.rect_value)  - particles[k].pos[0] - 2 * particles[k].radius
            
            colliding_times = dx/particles[k].v[0]

            colliding_times = min(colliding_times, tau_chain)
            
            # move to collision
            pdx = particles[k].v[0] * colliding_times
            particles[k].move(pdx, 0)

            # PBC conditions 
            valid, side = particles[k].check_valid_move()
            if not valid:
                #print("help")
                if side == 1:
                    particles[k].move(-self.rect_value , 0)
                    pdx += -self.rect_value

            # reset current particle velocity to 0
            # update particle idx to the new particle
            # update particle velocity 
            particles[k].v[0] = 0
            k = next_idx
            particles[k].v[0] = v[0]
            
            tau_chain -= colliding_times

            # sanity check for collisions
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         print("COLLIDED INVALID")
            #         print(particles[k].pos[0], particles[idx].pos[0])

        
        # reset all velocities to zero and store positions
        for particle in particles:
            self.inner_sf += np.exp(self.qt*1j * particle.pos[0])
            self.x_pos.append(particle.pos[0])
            #self.y_pos.append(particle.pos[1])
            particle.accepted += 1
            particle.total += 1
            particle.v[0] = 0

    def event_chain_ff_sequence(self, particles):
        k = np.random.randint(0, len(particles))
        v = [1, 0]
        # v = [1, 1]
        particles[k].v[0] = v[0]

        tau_chain = 10
        # P_T = len(particles)/(self.rect_value.width - len(particles)*particles[k].radius*2)
        #print(P_T)
        
        while tau_chain > 0:
            sampled_u = np.random.uniform(0, 1)
            P_T = np.random.exponential(0.2)
            x_ff = -np.log(sampled_u)/P_T

            next_idx = (k+1)%len(particles)
            prev_idx = (k-1)%len(particles)

            # particle is the STATIONARY particle
            particle = particles[next_idx]
            # pairwise collision time collisions
                
            # pass collision time calculation for the same particle

            # calculating collision times for PBC, if MOVING.x < STATIONARY.x, calculate normal
            # else if MOVING.X > STATIONARY.x, meaning it would have to wrap around due to PBC, PRETEND next collision is in the next box over
            if particles[k].pos[0] < particle.pos[0]:
                dx = particle.pos[0] - particles[k].pos[0] - 2 * particles[k].radius
            else:
                dx = (particle.pos[0] + self.rect_value) - particles[k].pos[0] - 2 * particles[k].radius

            colliding_times = dx/particles[k].v[0] 

            # TODO: solve for x_ff_t
            x_ff_t = x_ff/particles[k].v[0]
            #print(x_ff_t, colliding_times)
            
            # choose factor field or regular collision time, and pick lifting particle
            # if ff, then next particle is i-1, if regular collision, next particle is i + 1
            if x_ff_t < colliding_times:
                colliding_times = x_ff_t
                lifted_particle = prev_idx
            else:
                lifted_particle = next_idx
            

            if colliding_times < tau_chain:
                # move to collision
                pdx = particles[k].v[0] * colliding_times
                #pdy = particles[k].v[1] * colliding_times
                particles[k].move(pdx, 0)
                valid, side = particles[k].check_valid_move()
                if not valid:
                    if side == 1:
                        particles[k].move(-self.rect_value , 0)
                        pdx += -self.rect_value

                
                # reset current particle velocity to 0
                # update particle idx to the new particle
                # update particle velocity 
                particles[k].v[0] = 0
                k = lifted_particle
                particles[k].v[0] = v[0]
            else:
                # move to x + v * tau_chain
                pdx = particles[k].v[0] * tau_chain
                #pdy = particles[k].v[1] * tau_chain
                particles[k].move(pdx, 0)
                valid, side = particles[k].check_valid_move()
                if not valid:
                    #print("help")
                    
                    if side == 1:
                        particles[k].move(-self.rect_value , 0)
                        pdx += -self.rect_value 
                
            tau_chain -= colliding_times

            # # sanity check for collisions
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         print("COLLIDED INVALID")
            #         print(particles[k].pos[0], particles[idx].pos[0])

        
        # reset all velocities to zero
        for particle in particles:
            self.inner_sf += np.exp(self.qt*1j * particle.pos[0])
            self.x_pos.append(particle.pos[0])
            particle.accepted += 1
            particle.total += 1
            particle.v[0] = 0

    def simulate(self):
        print("Starting simulation.")
        count = 0

        tic = time.perf_counter()
        while count <= self.trials:
            
            # TODO: given n particles, choose one at random
            # generate a random move x, y
            # check collisions and boundary conditions, if accept then move, if reject then don't move
            # update positions
            
            self.sampling_method(particles=self.particle_list)

            count += 1
            if count % 10000 == 0:
                print(f"Acceptance: {[particle.accepted/particle.total for particle in self.particle_list]}")
                print(count)
                toc = time.perf_counter()
                print(f"Time taken: {toc - tic}")
                tic = time.perf_counter()
                self.structure_factor()

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

N_TRIALS = 10_000_00

# markov = Simulation("markov", N_TRIALS, 400, n_particles=4, spawning_protocol="uniform")
# m_x_pos, m_y_pos = markov.simulate()

ecmc = Simulation("event", N_TRIALS, 400, n_particles=4, spawning_protocol="uniform")
e_x_pos, e_y_pos = ecmc.simulate()

ecmc_ff = Simulation("event_ff", N_TRIALS, 400, n_particles=4, spawning_protocol="uniform")
e_ff_x_pos, e_ff_y_pos = ecmc_ff.simulate()

# SAVING
# markov.save_positions("markov_sampling_1mil-0620.csv")
# ecmc.save_positions("ecmc_1mil-1D.csv")
# ecmc_ff.save_positions("ecmc_ff_1mil-1D.csv")

# markov.save_structure_factors("markov_sf_10m-3.csv")
# ecmc.save_structure_factors("ecmc_sf_50m.csv")
# ecmc_ff.save_structure_factors("ecmc_ff_sf_50m.csv")

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
# markov_sf_df = pd.read_csv("markov_sf_10m-3.csv")
# markov_sf = markov_sf_df["sf"].to_list()

# ecmc_sf_df = pd.read_csv("ecmc_sf_50m.csv")
# ecmc_sf = ecmc_sf_df["sf"].to_list()

# ecmc_ff_sf_df = pd.read_csv("ecmc_ff_sf_50m.csv")
# ecmc_ff_sf = ecmc_ff_sf_df["sf"].to_list()

# x pos pdf
fig = plt.figure()
# plt.hist(np.array(m_x_pos), 40, density=True, histtype='step', label="markov x")

plt.hist(np.array(e_x_pos), 40, density=True, histtype='step', label="ecmc x")

plt.hist(np.array(e_ff_x_pos), 40, density=True, histtype='step', label="ecmc ff x")

plt.legend()
plt.show()

# structure factors
fig = plt.figure()
# plt.plot([10000 * i for i in range(len(markov.structure_factors))], np.array(markov.structure_factors), label="markov" )
plt.plot([10000 * i for i in range(len(ecmc.structure_factors))], np.array(ecmc.structure_factors, dtype=complex), label="ecmc")
plt.plot([10000 * i for i in range(len(ecmc_ff.structure_factors))], np.array(ecmc_ff.structure_factors, dtype=complex), label="ecmc ff")

# after loading SF!
# plt.hist(np.array(markov_sf, dtype=complex), 50, histtype='step', label="markov", density=True )
# plt.hist(np.array(ecmc_sf, dtype=complex), 50, histtype='step', label="ecmc", density=True, cumulative=True)
# plt.hist(np.array(ecmc_ff_sf, dtype=complex), 50, histtype='step', label="ecmc ff", density=True, cumulative=True )

# plt.plot([10000 * i for i in range(len(markov.structure_factors))], np.array(markov.structure_factors, dtype=complex), label="markov" )
# plt.plot([10000 * i for i in range(len(ecmc_sf))], np.array(ecmc_sf, dtype=complex), label="ecmc")
# plt.plot([10000 * i for i in range(len(ecmc_ff_sf))], np.array(ecmc_ff_sf, dtype=complex), label="ecmc ff")

plt.title("Structure Factor")

# after loading!
# plt.hist(np.array(m_x_pos) - 260, 40, density=True, histtype='step', label="markov x")
# plt.hist(np.array(m_y_pos) - 540, 40, density=True, histtype='step', label="markov y")

# plt.hist(np.array(e_x_pos) - ecmc.rect_value.left, 40, density=True, histtype='step', label="ecmc x")
# plt.hist(np.array(e_y_pos) - 540, 40, density=True, histtype='step', label="ecmc y")

# plt.hist(np.array(e_ff_x_pos) - ecmc_ff.rect_value.left, 40, density=True, histtype='step', label="ecmc ff x")
# plt.hist(np.array(e_y_pos) - 540, 40, density=True, histtype='step', label="ecmc y")

# plt.xlim(0)
plt.legend()

plt.show()
