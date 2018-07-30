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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 100;
	
    // create a normal (Gaussian) distribution for x, y, theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	
	for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);	 
        p.weight = 1;
        p.id = i;
        particles.push_back(p);   
	}
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // This line creates a normal (Gaussian) distribution for x, y, theta
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {

        // predict new state
        if (fabs(yaw_rate) < 0.00001) {  
            particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
            particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
        } 
        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + dist_x(gen);
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + dist_y(gen);
            particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


    // implementation of a naive quadratic search for point matching. (we can do better)
    for (int i = 0; i < observations.size(); i++) {

        // set the min distance between the two landmarks
        double min_dist = numeric_limits<double>::max();

        // default id for unassociated landmark
        observations[i].id = -1;

        for (int j = 0; j < predicted.size(); j++) {

            // get pairwise distance
            double pairwise_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            //associate id 
            if (pairwise_dist < min_dist) {
                min_dist = pairwise_dist;
                observations[i].id = predicted[j].id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // iterate through particles
    for (int i = 0; i < num_particles; i++) {
        
        // 1. find landmarks within range
        vector<LandmarkObs> predicted;
        for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            
            float landmark_x = map_landmarks.landmark_list[j].x_f;
            float landmark_y= map_landmarks.landmark_list[j].y_f;
            int landmark_id = map_landmarks.landmark_list[j].id_i;
            
            if(pow(particles[i].x - landmark_x,2) + pow(particles[i].y - landmark_y,2) <= pow(sensor_range,2)){
                predicted.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
            } 
        }
        
        // 2. transform  observations coordinate to the particle i coordinate
        vector<LandmarkObs> t_observations;
        for(int j = 0; j < observations.size(); j++) {
            double t_x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
            double t_y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
            t_observations.push_back(LandmarkObs{observations[j].id, t_x, t_y });
        }

        // 3. Associate observations to landmarks
        dataAssociation(predicted, t_observations);


        // 4. calculate weights
        particles[i].weight = 1.0;
        
        for(int j = 0; j < t_observations.size(); j++) {
            // find matched landmark
            LandmarkObs matched_landmark;
            for(int k=0; k < predicted.size(); k++){
                if(predicted[k].id == t_observations[j].id){
                    matched_landmark = predicted[k];
                    break;
                }
            }

            // set weights
            double dX = t_observations[j].x - matched_landmark.x;
            double dY = t_observations[j].y - matched_landmark.y;
            particles[i].weight *= ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( pow(dX,2)/(2*pow(std_landmark[0],2)) + (pow(dY,2)/(2*pow(std_landmark[1],2))) ) );
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


    // get max weight
    double max_weight = numeric_limits<double>::min();
    for(int i = 0; i < num_particles; i++) {
        max_weight = particles[i].weight;
    }
    // create dists
    uniform_real_distribution<double> dist_weights(0.0, max_weight);
    uniform_int_distribution<int> dist_idx(0, num_particles - 1);

    // Generating index.
    int idx = dist_idx(gen);

    // wheel trick
    vector<Particle> new_particles;
    double beta = 0.0;
    for(int i = 0; i < num_particles; i++) {
        beta += dist_weights(gen) * 2.0;
        while( beta > particles[idx].weight) {
            beta -= particles[idx].weight;
            idx = (idx + 1) % num_particles;
        }
        new_particles.push_back(particles[idx]);
    }
    particles = new_particles;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{  
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
