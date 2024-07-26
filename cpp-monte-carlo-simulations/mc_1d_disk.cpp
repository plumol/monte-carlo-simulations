#include <cmath>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <thread>
#include <fstream>
#include <iterator>
#include <complex>

class Particle {
    public:        
        // std::default_random_engine generator;
        // static std::uniform_real_distribution<double> distribution(-delta, delta);
        double radius;
        double bounding_box;
        double h_i;

        int accepted;
        int total;
        
        std::vector<double> pos;
        double v;
        double delta;

        void move(double pdx) {
            pos[0] += pdx;
        }

        bool is_collision(const Particle &other_particle, double tolerance = 1e-9) {
            //std::cout << "pdx " << fmod(pos[0] - other_particle.pos[0] + bounding_box/2, bounding_box) - 200.0 << std::endl;
            double distance = std::abs(fmod((std::abs(pos[0] - other_particle.pos[0]) + bounding_box/2), bounding_box) - 200.0);
            // if (distance < radius * 2.0) {
            //     std::cout << "dist " << distance << " radius " << radius*2 << std::endl;
            //     // std::cout << "collision spawn " << pos[0] << " " << other_particle.pos[0] << std::endl;
            //     std::cout << "distance - radius " << distance - radius*2 << std::endl;

            //     std::this_thread::sleep_for(std::chrono::seconds(1));
            // }
            // std::cout << "dist + tolerance " << distance + tolerance << std::endl;
            return distance < radius * 2.0 and distance > 0.0;
        }

        double propose_new_move(void) {
            std::mt19937 generator{std::random_device{}()};
            std::uniform_real_distribution<double> distribution(-delta, delta);
            total += 1;
            accepted += 1;
            double pdx = distribution(generator);
            return pdx;
        }

        std::vector<double> get_positions() {
            return pos;
        }

        Particle(double radius, std::vector<double> init_pos, std::string moveset, double bounding_box, double h_i){
            this->radius = radius;
            this->bounding_box = bounding_box;
            this->h_i = h_i;
            this->total = 0;
            this->accepted = 0;

            this->pos = init_pos;
            v = 0.0;
            delta = 0.1;
        }
};

class Simulation {

    // const int range_from = 0;
    // const int range_to = particle_list.size();

    public:
        double bounding_box;
        double mean;
        std::vector<Particle> particles;
        int trials;
        int active_idx;
        int n_particles;
        int diameter;
        
        std::vector<double> structure_factors;
        std::vector<double> x_pos;
        std::vector<int> event_count;
        std::vector<double> var_mix;

        void (*sampling_function) = nullptr;

        Simulation(std::string sampling_method, int n_trials, int bounding_box_size, int n_particles, double diameter, std::string spawning_protocol) {
            this->trials = n_trials;
            this->bounding_box = bounding_box_size;
            this->mean = (bounding_box_size - n_particles*diameter)/n_particles;
            this->n_particles = n_particles;
            this->diameter = diameter;

            event_count.reserve(trials);
            x_pos.reserve(trials * particles.size());
            structure_factors.reserve(trials);
            var_mix.reserve(trials);
            particles.reserve(n_particles);
            

            // init random idx for FF
            // if (sampling_method == "markov") {
            //     sampling_function = markov_chain_sequence;
            // }

            populate_spawning(n_particles, diameter/2, bounding_box_size, "random", spawning_protocol);
            // std::cout << "Initialized particle list" << std::endl;
            // std::cout << particles.size() << std::endl;
            for (Particle &particle : particles) {
                std::vector<double> positions = particle.get_positions();
                std::cout << positions[0] << std::endl;
            }
            std::cout << "NUM PARTICLES " << particles.size() << std::endl;

            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_int_distribution<> distribution(0, particles.size() - 1);
            this->active_idx = distribution(generator);


        }

        void populate_spawning(int n_particles, double radius, double bounding_box, std::string moveset, std::string spawning_protocol) {
            int max_particles_per_row = bounding_box / (radius*2);
            int current_row = 1;
            int current_column = 1;
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution<> distribution(radius + 0.1, bounding_box - radius - 0.1);

            for (int i = 0; i < n_particles; i++) {
                while (true) {
                    double init_x;
                    if (spawning_protocol == "random") {
                        init_x = distribution(generator);
                    }
                    if (spawning_protocol == "uniform") {
                        // this is for side by side spawning
                        init_x = (2.05 * radius * current_column) - radius;
                    }
                    std::vector<double> init_position({init_x, 0.0});
                    Particle particle(radius, init_position, moveset, bounding_box, (particles.size()+1)*radius*2);

                    bool add = true;
                    for (Particle &existing_particle : particles) {
                        if (particle.is_collision(existing_particle)) {
                            add = false;
                            break;
                        }

                    }

                    if (add) {
                        // std::cout << "Adding particle!" << particle.radius << std::endl;
                        particles.push_back(particle);
                        current_column += 1;
                        break;
                    }
            }
                }
            
        }

        void calculate_structure_factor() {
            std::complex<double> inner_sf(0.0, 0.0);
            double q = (2*M_PI)/bounding_box;
            std::complex<double> i(0.0, 1.0);

            for (const Particle &particle : particles) {
                inner_sf += exp(q*i * particle.pos[0]);
            }
            double structure = pow(std::abs(inner_sf), 2)/particles.size();
            structure_factors.push_back(structure);
        }

        void calculate_var_u() {
            std::vector<double> x_mix;
            std::vector<double> w;
            x_mix.reserve(n_particles);
            w.reserve(n_particles);

            for (Particle &particle : particles) {
                x_mix.push_back(particle.pos[0]);
            }

            for (int i = 0; i < n_particles; i++) {
                double distance = x_mix[(i+n_particles/2)%n_particles] - x_mix[i%n_particles] - n_particles/2 * diameter;
                if (distance < 0) {
                    distance += bounding_box;
                }
                w.push_back(distance);
            }

            // calculate mean and variance
            double mean = 0.0;
            double variance = 0.0;

            for (double &w_i: w) {
                mean += w_i;
            }
            mean /= n_particles;

            for (double &w_i : w) {
                variance += (w_i - mean) * (w_i - mean);
            }
            variance /= n_particles;

            var_mix.push_back(variance);
        }

        void save_structure_factors(std::string file_name) {
            std::ofstream output_file(file_name);
            output_file << "sf,events,var_mix\n";
            //std::ostream_iterator<double> output_iterator(output_file, ",\n");
            //std::copy(std::begin(structure_factors), std::end(structure_factors), output_iterator);
            for (int i = 0; i < structure_factors.size(); i++) {
                output_file << structure_factors[i] << "," << event_count[i] << "," << var_mix[i] << std::endl;
            }
        }

        void save_positions(std::string file_name) {
            std::ofstream output_file(file_name);
            output_file << "x,\n";
            std::ostream_iterator<double> output_iterator(output_file, ",\n");
            std::copy(std::begin(x_pos), std::end(x_pos), output_iterator);
        }

        void markov_chain_sequence() {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_int_distribution<> distribution(0, particles.size() - 1);

            int k = distribution(generator);//random

            double pdx = particles[k].propose_new_move();
            std::vector<double> init_pos = particles[k].get_positions();
            particles[k].move(pdx);

            // PBC conditions
            particles[k].pos[0] = fmod(particles[k].pos[0], bounding_box);
            for (int i = 0; i < particles.size(); i++) {
                if (i == k) {
                    continue;
                }
                else if (particles[k].is_collision(particles[i])) {
                    particles[k].accepted -= 1;
                    particles[k].pos[0] = init_pos[0];
                    break;
                }
            }
            x_pos.push_back(particles[k].pos[0]);
        }

        void event_chain_sequence() {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_int_distribution<> distribution(0, particles.size() - 1);
            std::exponential_distribution<> dist_tau(1.0/10.0);
            int k = distribution(generator);
            particles[k].v = 1;

            double tau_chain = dist_tau(generator);
            int events = 0;
            while (tau_chain > 0) {
                int next_idx = (k+1)%particles.size();
                const auto &particle = particles[next_idx];

                double dx = std::abs(fmod(std::abs(particles[k].pos[0] - particle.pos[0]) + bounding_box/2, bounding_box) - bounding_box/2) - 2 * particles[k].radius;

                // if (particles[k].pos[0] < particle.pos[0]) {
                //     dx = particle.pos[0] - particles[k].pos[0] - (2 * particles[k].radius);
                // }
                // else {
                //     dx = (particle.pos[0] + bounding_box) - particles[k].pos[0] - (2 * particles[k].radius);
                // }

                double colliding_times = dx/particles[k].v;

                colliding_times = std::min(colliding_times, tau_chain);

                double pdx = particles[k].v * colliding_times;
                particles[k].move(pdx);

                particles[k].pos[0] = fmod(particles[k].pos[0], bounding_box);

                particles[k].v = 0;
                k = next_idx;
                particles[k].v = 1;

                events += 1;
                tau_chain -= colliding_times;
                // std::cout << tau_chain << std::endl;

                // std::this_thread::sleep_for(std::chrono::seconds(1));
                
            }
            
            event_count.push_back(events);
            // for (int i = 0; i < particles.size(); i++) {
            //     if (k == i) {
            //         continue;
            //     }
            //     if (particles[k].is_collision(particles[i])) {
            //         std::cout << "collided invalid" << std::endl;
            //     }
            // }

            for (Particle &particle : particles) {
                // x_pos.push_back(particle.pos[0]);
                particle.accepted += 1;
                particle.total += 1;
                particle.v = 0;
            }
        }

        void event_chain_ff_sequence() {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::exponential_distribution<> dist_tau(1.0/10.0);

            int k = active_idx;
            particles[k].v = 1;
            int events = 0;
            double tau_chain = dist_tau(generator);

            while (tau_chain > 0) {
                std::exponential_distribution<> dist_ff(1/mean);
                double x_ff = dist_ff(generator);

                int next_idx = (k+1)%particles.size();
                int prev_idx = (k-1)%particles.size();

                const Particle &particle = particles[next_idx];

                double dx = std::abs(fmod(std::abs(particles[k].pos[0] - particle.pos[0]) + bounding_box/2, bounding_box) - bounding_box/2) - 2 * particles[k].radius;
                // double dx;
                // if (particles[k].pos[0] < particle.pos[0]) {
                //     dx = particle.pos[0] - particles[k].pos[0] - 2 * particles[k].radius;
                // }
                // else {
                //     dx = (particle.pos[0] + bounding_box) - particles[k].pos[0] - 2 * particles[k].radius;
                // }

                double colliding_times = dx/particles[k].v;
                double x_ff_t = x_ff/particles[k].v;

                // std::cout << colliding_times << " " << x_ff_t << std::endl;

                int lifted_particle;
                if (x_ff_t < colliding_times) { 
                    colliding_times = x_ff_t;
                    lifted_particle = prev_idx;
                }
                else {
                    lifted_particle = next_idx;
                }

                if (colliding_times < tau_chain) {
                    particles[k].move(particles[k].v * colliding_times);
                    particles[k].pos[0] = fmod(particles[k].pos[0], bounding_box);

                    particles[k].v = 0;
                    k = lifted_particle;
                    particles[k].v = 1;
                }
                else {
                    particles[k].move(particles[k].v * tau_chain);
                    particles[k].pos[0] = fmod(particles[k].pos[0], bounding_box);
                }

                events += 1;
                tau_chain -= colliding_times;

            }
            event_count.push_back(events);
            active_idx = k;
            // for (int i = 0; i < particles.size(); i++) {
            //     if (k == i) {
            //         continue;
            //     }
            //     if (particles[k].is_collision(particles[i])) {
            //         std::cout << "collided invalid" << std::endl;
            //     }
            // }

            for (Particle &particle : particles) {
                // x_pos.push_back(particle.pos[0]);
                particle.accepted += 1;
                particle.total += 1;
                particle.v = 0;
            }

        }

        void event_chain_ff_sequence_acc() {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::exponential_distribution<> dist_tau(1.0/400);

            int k = active_idx;
            particles[k].v = 1;
            int events = 0;
            double tau_chain = dist_tau(generator);

            while (tau_chain > 0) {
                std::exponential_distribution<> dist_ff(1/particles[k].h_i);
                double x_ff = dist_ff(generator);

                int next_idx = (k+1)%particles.size();
                int prev_idx = (k-1)%particles.size();

                const Particle &particle = particles[next_idx];

                double dx = std::abs(fmod(std::abs(particles[k].pos[0] - particle.pos[0]) + bounding_box/2, bounding_box) - bounding_box/2) - 2 * particles[k].radius;
                // double dx;
                // if (particles[k].pos[0] < particle.pos[0]) {
                //     dx = particle.pos[0] - particles[k].pos[0] - 2 * particles[k].radius;
                // }
                // else {
                //     dx = (particle.pos[0] + bounding_box) - particles[k].pos[0] - 2 * particles[k].radius;
                // }

                double colliding_times = dx/particles[k].v;
                double x_ff_t = x_ff/particles[k].v;

                // std::cout << colliding_times << " " << x_ff_t << std::endl;

                int lifted_particle;
                if (x_ff_t < colliding_times) { 
                    colliding_times = x_ff_t;
                    lifted_particle = prev_idx;
                }
                else {
                    lifted_particle = next_idx;
                }

                if (colliding_times < tau_chain) {
                    particles[k].move(particles[k].v * colliding_times);
                    particles[k].pos[0] = fmod(particles[k].pos[0], bounding_box);

                    particles[k].v = 0;
                    k = lifted_particle;
                    particles[k].v = 1;
                }
                else {
                    particles[k].move(particles[k].v * tau_chain);
                    particles[k].pos[0] = fmod(particles[k].pos[0], bounding_box);
                }

                events += 1;
                tau_chain -= colliding_times;

            }
            event_count.push_back(events);
            active_idx = k;
            // for (int i = 0; i < particles.size(); i++) {
            //     if (k == i) {
            //         continue;
            //     }
            //     if (particles[k].is_collision(particles[i])) {
            //         std::cout << "collided invalid" << std::endl;
            //     }
            // }

            for (Particle &particle : particles) {
                // x_pos.push_back(particle.pos[0]);
                particle.accepted += 1;
                particle.total += 1;
                particle.v = 0;
            }

        }

        void simulate() {
            std::cout << "Starting simulation." << std::endl;
            int count = 0;
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_int_distribution<> distribution(0, 20);

            auto t1 = std::chrono::high_resolution_clock::now();

            // for (Particle particle : particles) {
            //     std::cout << particle.radius << std::endl;
            // }
            while (count < trials) {
                // sampling_method(particles=particle_list);

                event_chain_ff_sequence();
                count += 1;

                if (count % 100000 == 0) {
                    std::cout << count << std::endl;

                    auto t2 = std::chrono::high_resolution_clock::now();
                    
                    int num_accepted = 0;
                    int num_total = 0;
                    for (Particle &particle : particles) {
                        num_accepted += particle.accepted;
                        num_total += particle.total;
                    }
                    std::cout << "Acceptance " << (double) num_accepted / num_total << std::endl;
                    auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1);
                    std::cout << "Time taken: " << duration.count() << std::endl;
                    t1 = std::chrono::high_resolution_clock::now();

                }
                if (count % 1 == 0) {
                    calculate_structure_factor();
                    calculate_var_u();
                }
            }

        }

};

int main(int argc, char *argv[]) {
    int N_TRIALS = 100000;
    int N_PARTICLES = std::stoi(argv[1]);
    double DIAMETER = 400.0/(2*N_PARTICLES);
    std::cout << DIAMETER << " " << DIAMETER/2 << std::endl;
    std::cout << argv[1] << std::endl;
    std::string arg1(argv[1]);

    Simulation markov("markov", N_TRIALS, 400, N_PARTICLES, DIAMETER, "uniform");
    markov.simulate();
    // markov.save_positions("ecmc_sf_10m-100.csv");
    markov.save_structure_factors("ecmc_ff_sf_10m-" + arg1 + ".csv");

}
