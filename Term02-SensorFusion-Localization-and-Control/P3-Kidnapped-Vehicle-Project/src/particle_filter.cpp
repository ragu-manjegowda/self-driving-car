/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "particle_filter.h"
#include "map.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 250;
    int i = 0; 
    
    while(i< num_particles)
    {
        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> gps_x(x, std[0]);
        normal_distribution<double> gps_y(y, std[1]);
        normal_distribution<double> gps_theta(theta, std[2]);

        double particle_x = gps_x(gen);
        double particle_y = gps_y(gen);
        double particle_theta = gps_theta(gen);
        double particle_weight = 1.0;

        Particle particle = {i, particle_x, particle_y, particle_theta, particle_weight};
        particles.push_back(particle);
        weights.push_back(particle_weight);

        i++;
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    
    int i=0; 
    
    while(i < num_particles)
    {

        if(yaw_rate == 0){
            particles[i].x = particles[i].x + (velocity * delta_t) * cos(particles[i].theta);
            particles[i].y = particles[i].y + (velocity * delta_t) * sin(particles[i].theta);
        }else{
            particles[i].x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta = particles[i].theta + (yaw_rate * delta_t);
        }


        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> pos_error_x(particles[i].x, std_pos[0]);
        normal_distribution<double> pos_error_y(particles[i].y, std_pos[1]);
        normal_distribution<double> pos_error_theta(particles[i].theta, std_pos[2]);

        particles[i].x = pos_error_x(gen);
        particles[i].y = pos_error_y(gen);
        particles[i].theta = pos_error_theta(gen);

        i++;
    }
	
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    
    int i = 0; 
    
    while(i < observations.size())
    {

        double x = observations[i].x;
        double y = observations[i].y;

        double delta_temp = 0.0;
        bool initialized = false;

        int j = 0; 
        while(j<predicted.size())
        {

            double delta_x = x - predicted[j].x;
            double delta_y = y - predicted[j].y;

            double delta = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));

            if((initialized == false) || (delta_temp > delta)) {
                delta_temp = delta;
                initialized = true;
                observations[i].id = j;
            }

            j++;
        }

        i++;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html
    
    int i = 0; 

    while(i < num_particles)
    {

        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        vector<LandmarkObs> predicted_landmarks;
        for(int j=0; j<map_landmarks.landmark_list.size(); j++)
        {
            int j_id = map_landmarks.landmark_list[j].id_i;
            double j_x = map_landmarks.landmark_list[j].x_f;
            double j_y = map_landmarks.landmark_list[j].y_f;

            double delta_x = j_x - x;
            double delta_y = j_y - y;

            double distance = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));
            
            if(distance <= sensor_range){
                j_x = delta_x * cos(theta) + delta_y * sin(theta);
                j_y = delta_y * cos(theta) - delta_x * sin(theta);
                LandmarkObs landmark_in_range = {j_id, j_x, j_y};
                predicted_landmarks.push_back(landmark_in_range);
            }
        }

        dataAssociation(predicted_landmarks, observations);

        double new_weight = 1.0;
        
        for(int k=0; k<observations.size(); k++) 
        {
            int k_id = observations[k].id;
            double k_x = observations[k].x;
            double k_y = observations[k].y;

            double delta_x = k_x - predicted_landmarks[k_id].x;
            double delta_y = k_y - predicted_landmarks[k_id].y;

            double a = exp(- 0.5 * (pow(delta_x,2.0)*std_landmark[0] + pow(delta_y,2.0)*std_landmark[1] ));
            double b = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
            new_weight = new_weight * (a/b);
        }
        weights[i] = new_weight;
        particles[i].weight = new_weight;

        i++;
    }
	
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    std::vector<Particle> resampled_particles;

    random_device rd;
    
    default_random_engine gen(rd());

    int i = 0; 
    
    while(i < particles.size())
    {

        discrete_distribution<int> index(weights.begin(), weights.end());
        resampled_particles.push_back(particles[index(gen)]);

        i++;
    }

    particles = resampled_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
