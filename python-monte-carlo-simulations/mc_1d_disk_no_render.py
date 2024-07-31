import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

screen_width = 1280
screen_height = 720

class Particle():
    
    def __init__(self, color, radius, width = 0, init_pos = [screen_width/2, screen_height/2], moveset = "normal", pbc = False, bounding_box = None, h_i = 0) -> None:

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
        self.v = 0.0
        self.delta = 10
        self.h_i = h_i

        # this line is supposed to be an initial movement, but it doesn't actually move until you press R
        #self.check_valid_move(self.dx, self.dy)

    def move(self, pdx, pdy, reset = False):
        self.pos[0] += pdx
        pass
    
    def is_collision(self, other_particle):
        distance = np.abs((self.pos[0] - other_particle.pos[0] + self.bounding_box/2)%self.bounding_box - self.bounding_box/2)
        return distance < self.radius + other_particle.radius - 0.000000001 and distance > 0
            
    def propose_new_move(self):
        self.total += 1
        self.accepted += 1
        pdx = np.random.uniform(-self.delta, self.delta)
        return pdx

    def check_valid_move(self):
        if self.pos[0] < 0:
            return (False, 0)
        elif self.pos[0] > self.bounding_box:
            return (False, 1)
        else:
            return (True, 2)
        


class Simulation():

    def __init__(self, sampling_method, n_trials, bounding_box_size=200, n_particles=4, diameter=48, exp_var = 10, spawning_protocol="random"):
        """
        Initializes a Monte Carlo simulation of hard disks within a bounding box. 

        Sampling methods: "direct", "markov", "event"
        Spawning protocols: "random", "uniform"
            Random spawning leads to randomly intiialized particles within the bounding box, may have a specific limit based
            on particle radius and number of particles
            Uniform spawning leads to uniformly initialized particles in a row, column formation.
        """
        self.x_pos = []
        self.attemped_x = []
        self.collided_x = []
        self.y_pos = []
        self.particle_list = []
        self.structure_factors = []
        self.events = []
        self.var_mix = []
        self.moves = []

        self.dt = 0
        self.running = True
        self.trials = n_trials
        self.sm = sampling_method
        self.mean = (bounding_box_size - n_particles*diameter)/n_particles
        

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
            self.sampling_method = self.event_chain_ff_sequence_acc
            moveset = "normal"

        self.rect_value = bounding_box_size

        self.populate_spawning(n_particles, diameter/2, 3, bounding_box=self.rect_value, moveset=moveset, spawning_protocol=spawning_protocol)
        # print([particle.pos for particle in self.particle_list])
        #self.active_idx = np.random.randint(0, len(self.particle_list))
        self.active_idx = len(self.particle_list) - 1

        #self.particle_list[0].h_i = self.mean
        
        total_h_i = 0
        for particle in self.particle_list:
            total_h_i += particle.h_i
        print(total_h_i)
        print([particle.h_i for particle in self.particle_list])
        

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
                elif spawning_protocol == "uniform":
                    init_x = 0 + (2.000000001 * radius * current_column) - radius
                    #init_x = 0 + (bounding_box/n * current_column) - radius
                particle = Particle("red", radius, width=width, init_pos=[init_x, 0], h_i=n/2 * self.mean,
                                    moveset=moveset, bounding_box=self.rect_value)
                #h_i=(len(self.particle_list)+1)*DIAMETER
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
        inner_sf = 0.0
        q = (2*np.pi)/self.rect_value

        for particle in self.particle_list:
            inner_sf += np.exp(q*1j * particle.pos[0])
        structure = np.abs(inner_sf)**2/len(self.particle_list)

        self.structure_factors.append(structure)
    
    def mixing_times2(self):
        # x_mix = [particle.pos[0] if particle.pos[0] > self.particle_list[0].pos[0] else particle.pos[0] + self.rect_value for particle in self.particle_list]
        x_mix = [particle.pos[0] for particle in self.particle_list]
        #sort better
        #x_mix = sorted(x_mix)
        #print(x_mix)
        w = []
        
        n_particles = len(self.particle_list)
        # first iter

        # exact s_i+n/2 - s_i, reduced with notes
        for i in range(1, n_particles+1):
            w_i = 0
            for j in range(i + int(n_particles/2), i, -1):
                distance = x_mix[(j-1)%n_particles] - x_mix[(j-2)%n_particles] - self.particle_list[0].radius*2
                w_i += distance
                if distance < 0:
                    w_i += self.rect_value
            w.append(w_i)
        
        self.var_mix.append(np.var(w))

    def save_structure_factors(self, file_name):
        df = pd.DataFrame(zip(self.structure_factors, self.events, self.var_mix), columns=["sf", "events", "var_mix"])
        df.to_csv(file_name)

    def save_positions(self, file_name):
        df = pd.DataFrame(zip(self.x_pos), columns=["x"])
        df.to_csv(file_name)

    def markov_chain_sequence(self, particles):
        k = np.random.randint(0, len(particles))
        pdx = particles[k].propose_new_move()
        init_pos = particles[k].pos[0]
        particles[k].move(pdx, 0)

        # PBC
        particles[k].pos[0] = particles[k].pos[0] % self.rect_value

        for idx, particle in enumerate(particles):
            if idx == k:
                continue
            if particles[k].is_collision(particle):
                particles[k].accepted -= 1
                particles[k].pos[0] = init_pos
                break
        

        # sanity check for collisions
        # for particle in particles:
        #     if particles[k].is_collision(particle):
        #         print("COLLIDED INVALID")
        #         print(particles[k].pos[0], particles[idx].pos[0])
        # self.x_pos.append(particles[k].pos[0])
        # self.y_pos.append(particles[k].pos[1])
        # if valid:
        #     for particle in particles:
        self.x_pos.append(particles[k].pos[0])


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
        v = 1
        particles[k].v = v

        tau_chain = np.random.exponential(10)
        events = 0
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
            
            colliding_times = dx/particles[k].v

            colliding_times = min(colliding_times, tau_chain)
            
            # move to collision
            pdx = particles[k].v * colliding_times
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
            particles[k].v = 0
            k = next_idx
            particles[k].v = v
            
            # print(tau_chain)
            events += 1
            tau_chain -= colliding_times

            # sanity check for collisions
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         print("COLLIDED INVALID")
            #         print(particles[k].pos[0], particles[idx].pos[0])
        self.events.append(events)
        
        # reset all velocities to zero and store positions
        for particle in particles:
            self.x_pos.append(particle.pos[0])
            #self.y_pos.append(particle.pos[1])
            particle.accepted += 1
            particle.total += 1
            particle.v = 0

    def event_chain_ff_sequence(self, particles):
        k = self.active_idx
        v = 1
        # v = [1, 1]
        particles[k].v = v
        events = 0 
        #tau_chain = np.random.exponential(10)
        tau_chain = self.rect_value
        # P_T = len(particles)/(self.rect_value.width - len(particles)*particles[k].radius*2)
        #print(P_T)
        
        while tau_chain > 0:
            #x_ff = np.random.exponential(self.mean)
            x_ff = np.random.exponential(self.mean)
            

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

            colliding_times = dx/particles[k].v
            #print(colliding_times)
            # TODO: solve for x_ff_t
            x_ff_t = x_ff/particles[k].v
            
            # choose factor field or regular collision time, and pick lifting particle
            # if ff, then next particle is i-1, if regular collision, next particle is i + 1
            if x_ff_t < colliding_times:
                colliding_times = x_ff_t
                lifted_particle = prev_idx
            else:
                lifted_particle = next_idx
            

            if colliding_times < tau_chain:
                # move to collision
                pdx = particles[k].v * colliding_times
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
                particles[k].v = 0
                k = lifted_particle
                particles[k].v = v
            else:
                # move to x + v * tau_chain
                pdx = particles[k].v * tau_chain
                particles[k].move(pdx, 0)
                valid, side = particles[k].check_valid_move()
                if not valid:
                    
                    if side == 1:
                        particles[k].move(-self.rect_value , 0)
                        pdx += -self.rect_value 
            
            events += 1
            tau_chain -= colliding_times
            self.x_pos.append(particles[k].pos[0])

            # # sanity check for collisions
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         print("COLLIDED INVALID")
            #         print(particles[k].pos[0], particles[idx].pos[0])
        self.active_idx = k
        self.events.append(events)
        # reset all velocities to zero
        for particle in particles:
            
            particle.accepted += 1
            particle.total += 1
            particle.v = 0
    
    def event_chain_ff_sequence_acc(self, particles):
        k = self.active_idx
        v = 1
        # v = [1, 1]
        particles[k].v = v
        events = 0 
        a = "collision"
        # if len(self.events) < 2:
        #     tau_chain = self.rect_value*(len(particles) + 1)/4
        # else:
        #     tau_chain = self.rect_value
        tau_chain = self.rect_value
        # P_T = len(particles)/(self.rect_value.width - len(particles)*particles[k].radius*2)
        #print(P_T)
        
        while tau_chain > 0:
            #x_ff = np.random.exponential(self.mean)
            x_ff = np.random.exponential(particles[k].h_i)
            # print(particles[k].h_i)
            # print([particle.h_i for particle in particles])
            

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

            # front_dist = np.abs((particles[k].pos[0] - particle.pos[0] + self.rect_value/2)%self.rect_value - self.rect_value/2)
            # back_dist = np.abs((particles[k].pos[0] - particles[prev_idx].pos[0] + self.rect_value/2)%self.rect_value - self.rect_value/2)
            # middx = (front_dist + back_dist)/2

            # x_ff = np.random.exponential(middx)
            # calculate distance with (i, i-1)
            
            
            # try:
            #     h = max(dx-particles[k].radius*2, self.mean)
            #     #print(h)
            #     x_ff = np.random.exponential(dx)
            # except:
            #     print("EXP BROKE", dx)

            colliding_times = dx/particles[k].v
            #print(colliding_times)
            # TODO: solve for x_ff_t
            x_ff_t = x_ff/particles[k].v
            
            #print(colliding_times, x_ff)
            # choose factor field or regular collision time, and pick lifting particle
            # if ff, then next particle is i-1, if regular collision, next particle is i + 1
            if x_ff_t < colliding_times:
                colliding_times = x_ff_t
                lifted_particle = prev_idx
                # particles[lifted_particle].h_i = 0.9 * particles[k].h_i
                # particles[k].h_i = self.mean
                a = "FF"
            else:
                lifted_particle = next_idx
                a = "COLLISION"
            
            #print(f"collide {a}", colliding_times, (k, lifted_particle), tau_chain, particles[k].pos[0], particles[k].pos[0] + colliding_times)

            if colliding_times < tau_chain:
                # move to collision
                pdx = particles[k].v * colliding_times
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
                particles[k].v = 0
                #particles[k].h_i = self.mean
                k = lifted_particle
                particles[k].v = v
            else:
                # move to x + v * tau_chain
                pdx = particles[k].v * tau_chain
                particles[k].move(pdx, 0)
                valid, side = particles[k].check_valid_move()
                if not valid:
                    
                    if side == 1:
                        particles[k].move(-self.rect_value , 0)
                        pdx += -self.rect_value 
            
            events += 1
            tau_chain -= colliding_times
            #self.moves.append([particle.pos[0] for particle in particles])
            
            #time.sleep(2)

            # # sanity check for collisions
            # for idx, particle in enumerate(particles):
            #     if particles[k].is_collision(particle):
            #         print("COLLIDED INVALID")
            #         print(particles[k].pos[0], particles[idx].pos[0])
        self.active_idx = k
        self.events.append(events)

        # print("total events: ", events)

        # print("final pos ", [particle.pos[0] for particle in self.particle_list])
        # reset all velocities to zero
        for particle in particles:
            self.x_pos.append(particle.pos[0])
            particle.accepted += 1
            particle.total += 1
            particle.v = 0

    def simulate(self):
        print("Starting simulation.")
        count = 0
        mixed_time = 0
        swept = False

        for particle in self.particle_list:
            self.x_pos.append(particle.pos[0])
        self.moves.append([particle.pos[0] for particle in self.particle_list])

        tic = time.perf_counter()
        self.mixing_times2()
        self.structure_factor()
        self.events.append(0)
        while count <= self.trials:
            
            # TODO: given n particles, choose one at random
            # generate a random move x, y
            # check collisions and boundary conditions, if accept then move, if reject then don't move
            # update positions
            
            self.sampling_method(particles=self.particle_list)

            count += 1

            if mixed_time == 0: 
                if self.var_mix[count -1]/var_equil < 1:
                    print(np.sum(self.events), count)
                    mixed_time = count
                    #time.sleep(2)

            # if not swept:
            #     if count == 3:
            #         for particle in self.particle_list:
            #             particle.h_i = self.mean
            #         swept = True
            if count == (N_PARTICLES*self.rect_value/4)/self.rect_value:
                for particle in self.particle_list:
                    particle.h_i = self.mean
                swept = True
            
            if count % 50000 == 0:
                print(f"Acceptance: {[particle.accepted/particle.total for particle in self.particle_list]}")
                print(count)
                toc = time.perf_counter()
                print(f"Time taken: {toc - tic}. Estimated remaining {(self.trials-count)/100000*(toc-tic)}")
                tic = time.perf_counter()
            if count % 1 == 0 :
                self.structure_factor()
                self.mixing_times2()

        print("Finished!")
        return self.x_pos, self.y_pos, mixed_time


# plt.hist(list(x_pos.keys()), 20, density=True)
# plt.show()

# plt.hist2d(list(x_pos.keys()), list(y_pos.keys()), 20, density=True)
# plt.show()

# x_ticks = np.arange(24, 200 - 24, 24)
# y_ticks = np.arange(24, 200 - 24, 24)
# plt.hist2d(np.array(x_pos) - 540, np.array(y_pos) - 260, 20, density=False)
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)

N_TRIALS = 25_000
SYSTEM_LENGTH = 400
N_PARTICLES = 16
DIAMETER = SYSTEM_LENGTH/(2*N_PARTICLES)

var_equil = (400-N_PARTICLES*DIAMETER)**2/(4*(N_PARTICLES+1))

# markov = Simulation("markov", N_TRIALS, SYSTEM_LENGTH, n_particles=N_PARTICLES, diameter=DIAMETER, spawning_protocol="uniform")
# m_x_pos, m_y_pos = markov.simulate()

# ecmc = Simulation("event", N_TRIALS, 400, n_particles=N_PARTICLES, diameter=DIAMETER, spawning_protocol="uniform")
# e_x_pos, e_y_pos = ecmc.simulate()

# run loop
ecmc_ff_var_mix = []
ecmc_ff_events = []
ecmc_ff_sf = []
for i in range(0, 10):
    print(i)
    ecmc_ff = Simulation("event_ff", N_TRIALS, 400, n_particles=N_PARTICLES, diameter=DIAMETER, spawning_protocol="uniform")
    e_ff_x_pos, e_ff_y_pos, mixed_time = ecmc_ff.simulate()
    ecmc_ff_var_mix.append(ecmc_ff.var_mix)
    ecmc_ff_events.append(ecmc_ff.events)
    ecmc_ff_sf.append(ecmc_ff.structure_factors)
ecmc_ff_var_mix = np.mean(ecmc_ff_var_mix, axis=0)
ecmc_ff_events = np.mean(ecmc_ff_events, axis=0)
ecmc_ff_sf = np.mean(ecmc_ff_sf, axis=0)

# moves_df = pd.DataFrame(ecmc_ff.moves)
# moves_df.to_csv("accelerated_first_chain.csv")
# SAVING
# markov.save_positions("markov_sampling_1mil-0620.csv")
# ecmc.save_positions("test.csv")
# ecmc_ff.save_positions("ecmc_ff_1mil-8m2.csv")
# 
# markov.save_structure_factors("markov_sf_10m-120.csv")
# ecmc.save_structure_factors("ecmc_sf_10m-2-py.csv")
# ecmc_ff.save_structure_factors("ecmc_ff_16_base-2.csv")
# x_8 = np.linspace(0, np.sum(ecmc_ff.events)/16, len(e_ff_x_pos))
# plt.plot(np.array(e_ff_x_pos[:3000])-SYSTEM_LENGTH/2, x_8[:3000])
# plt.show()

# df = pd.DataFrame(zip(ecmc_ff_var_mix, ecmc_ff_events, ecmc_ff_sf), columns=["var_mix", "events", "sf"])
# df.to_csv("ecmc_ff_16_c-4.csv")

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
# markov_sf_df = pd.read_csv("markov_sf_10m-2010.csv")
# markov_sf = markov_sf_df["sf"].to_list()

# ecmc_sf_df = pd.read_csv("ecmc_sf_10m-2010.csv")
# ecmc_sf = ecmc_sf_df["sf"].to_list()

# ecmc_ff_sf_df = pd.read_csv("ecmc_ff_sf_10m-2010.csv")
# ecmc_ff_sf = ecmc_ff_sf_df["sf"].to_list()

# x pos pdf
# fig = plt.figure()

# plt.hist(np.array(m_x_pos), 40, density=True, histtype='step', label="markov x")

# plt.hist(np.array(e_x_pos), 40, density=True, histtype='step', label="ecmc x")

plt.hist(np.array(e_ff_x_pos), 40, density=True, histtype='step', label="ecmc ff x")

plt.show()

#final configuration of particles
# plt.scatter(np.array(m_x_pos[-N_PARTICLES:]), [0 for i in range(N_PARTICLES)], 10, alpha=0.3, label="markov x")

# plt.scatter(np.array(e_x_pos[-N_PARTICLES:]), [0.5 for i in range(N_PARTICLES)], 40, alpha=0.3, label="ecmc x")
step = 0
plt.scatter(np.array(e_ff_x_pos[N_PARTICLES*step:N_PARTICLES*(step+1)]), [0.7 for i in range(N_PARTICLES)], DIAMETER, alpha=0.3, label="ecmc ff x")
plt.scatter(np.array(e_ff_x_pos[N_PARTICLES*(step +1):N_PARTICLES*(step+2)]), [0.701 for i in range(N_PARTICLES)], DIAMETER, alpha=0.3, label="ecmc ff x")
plt.scatter(np.array(e_ff_x_pos[N_PARTICLES*(step +2):N_PARTICLES*(step+3)]), [0.702 for i in range(N_PARTICLES)], DIAMETER, alpha=0.3, label="ecmc ff x")
plt.scatter(np.array(e_ff_x_pos[N_PARTICLES*(step +3):N_PARTICLES*(step+4)]), [0.703 for i in range(N_PARTICLES)], DIAMETER, alpha=0.3, label="ecmc ff x")
plt.scatter(np.array(e_ff_x_pos[N_PARTICLES*(step +4):N_PARTICLES*(step+5)]), [0.704 for i in range(N_PARTICLES)], DIAMETER, alpha=0.3, label="ecmc ff x")


plt.xlim(0, 400)
# plt.legend()
plt.show()

eff_base_df = pd.read_csv("ecmc_ff_16_base-2.csv")
eff_base_df["events"] = eff_base_df["events"].shift(1)
eff_base_df["events"][0] = 0
eff_base_mix = np.array(eff_base_df["var_mix"].to_list())
eff_base_events = np.array(eff_base_df["events"].to_list())
eff_base_sf = np.array(eff_base_df['sf'])

# # structure factors
# fig = plt.figure()
# plt.plot([1 * i for i in range(len(markov.structure_factors))], np.array(markov.structure_factors), label="markov" )
# plt.plot([10000 * i for i in range(len(ecmc.structure_factors))], np.array(ecmc.structure_factors, dtype=complex), label="ecmc")
# plt.plot([10000 * i for i in range(len(ecmc_ff.structure_factors))], np.array(ecmc_ff.structure_factors, dtype=complex), label="ecmc ff")
plt.hist(np.array(ecmc_ff.structure_factors), 40, histtype='step', label="test")
plt.hist(np.array())
print(np.mean(ecmc_ff.structure_factors), np.mean(ecmc_ff.events), ecmc_ff_events[0:5])

# after loading SF!
# plt.hist(np.array(markov_sf, dtype=complex), 100, histtype='step', label="markov", density=True, cumulative=True)
# plt.hist(np.array(ecmc_sf, dtype=complex), 100, histtype='step', label="ecmc", density=True, cumulative=True)
# plt.hist(np.array(ecmc_ff_sf, dtype=complex), 100, histtype='step', label="ecmc ff", density=True, cumulative=True)

# plt.plot([10000 * i for i in range(len(markov_sf))], np.array(markov_sf, dtype=complex), label="markov" )
# plt.plot([10000 * i for i in range(len(ecmc_sf))], np.array(ecmc_sf, dtype=complex), label="ecmc")
# plt.plot([10000 * i for i in range(len(ecmc_ff_sf))], np.array(ecmc_ff_sf, dtype=complex), label="ecmc ff")

# plt.title("Structure Factor")

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



# mixing times
# eff_x = np.linspace(0, N_TRIALS, N_TRIALS)*np.mean(ecmc_ff.events)/N_PARTICLES
# eff_x_base = np.linspace(0, N_TRIALS, N_TRIALS)*np.mean(eff_base_events)/16
eff_x = np.cumsum(ecmc_ff_events)/N_PARTICLES
eff_x_base = np.cumsum(eff_base_events)/16


var_equil = (400-N_PARTICLES*DIAMETER)**2/(4*(N_PARTICLES+1))
var_equil_16 = (400-16*(SYSTEM_LENGTH/(2*16)))**2/(4*(16+1))

plt.plot(eff_x_base[:2000]/16, eff_base_mix[:2000]/var_equil_16, label="base")
plt.plot(eff_x[:2000]/N_PARTICLES, np.array(ecmc_ff_var_mix[:2000])/var_equil, label="test")

print(mixed_time, ecmc_ff_var_mix[mixed_time-1]/var_equil)

print(ecmc_ff.var_mix[0], np.mean(ecmc_ff_var_mix)/var_equil)
# plt.vlines((mixed_time-1)*np.mean(ecmc_ff_events)/N_PARTICLES**2, 0, 10, "black")
plt.hlines(1, 0, 10, "black")
#print(np.mean(np.array(ecmc_ff_var_mix[:N_PARTICLES**2])/var_equil))

plt.legend()
plt.yscale('log')
plt.xlim(0, 2)
plt.show()
