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
#include <assert.h>     /* assert */

#include "map.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Number of particles to draw
	num_particles = 10;

	// set standard deviations for x, y and theta (GPS sensor noise)
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// define random number engine
	default_random_engine gen;

	// create normal (Gaussian) distributions for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// create initial weight for all particles
	double init_weight = 1. / num_particles;

	// create and initialize particles and weights
	for (unsigned int p = 0; p < num_particles; ++p) {

		// create new particle
		Particle particle;
		particle.id = p;
		// sample from these normal distributions
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	    while ( particle.theta >  2 * M_PI ) particle.theta -= 2. * M_PI;
	    while ( particle.theta < 0.0 ) particle.theta += 2. * M_PI;
		particle.weight = 1.;
		particles.push_back(particle);

		// initialize all weights uniformely
		weights.push_back(init_weight);
	}

	// Flag, if filter is initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// set standard deviations for x, y and theta (sensor noise)
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	// define random number engine
	default_random_engine gen;

	// create normal (Gaussian) distributions for x, y and theta
	normal_distribution<double> dist_x(0.0, std_x);
	normal_distribution<double> dist_y(0.0, std_y);
	normal_distribution<double> dist_theta(0.0, std_theta);

	// check if yaw rate is zero
    if (fabs(yaw_rate) >= 0.00001) {

		// make predictions for each particle
		for (unsigned int i = 0; i < num_particles; ++i) {

			Particle particle = particles[i];
			double theta_0 = particle.theta + dist_theta(gen);
			double theta_f = theta_0 + yaw_rate * delta_t;
			double temp1 = velocity / yaw_rate;
			particle.x = particle.x + dist_x(gen) + temp1 * (sin(theta_f) - sin(theta_0));
			particle.y = particle.y + dist_y(gen) + temp1 * (cos(theta_0) - cos(theta_f));
			particle.theta = theta_f;
		    while ( particle.theta >  2 * M_PI ) particle.theta -= 2. * M_PI;
		    while ( particle.theta < 0.0 ) particle.theta += 2. * M_PI;
			particles[i] = particle;
		}

	} else {

		// make predictions for each particle
		for (int i = 0; i < num_particles; ++i) {

			Particle particle = particles[i];
			double theta = particle.theta + dist_theta(gen);
			double distance = velocity * delta_t;
			particle.x = particle.x + dist_x(gen) + cos(theta) * distance;
			particle.y = particle.y + dist_y(gen) + sin(theta) * distance;
			particle.theta = theta;
		    while ( particle.theta >  2 * M_PI ) particle.theta -= 2. * M_PI;
		    while ( particle.theta < 0.0 ) particle.theta += 2. * M_PI;
			particles[i] = particle;
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// loop over observations
	for (unsigned int o = 0; o < observations.size(); ++o) {

		// set min distance to distance to the first landmark in the prediction vector
		int min_id = predicted[0].id;
		double min_dist = dist(observations[o].x, observations[o].y, predicted[0].x, predicted[0].y);

		// loop over predictions (landmarks within sensor range of the particle)
		for (unsigned int l = 1; l < predicted.size(); ++l) {

			// calculate distance between observation and predicted landmark
			double curr_dist = dist(observations[o].x, observations[o].y, predicted[l].x, predicted[l].y);
			// check if distance is smaller than the min distance so far
			if (curr_dist < min_dist) {
				min_id = predicted[l].id;
				min_dist = curr_dist;
			}
		}
		// assign the nearest (predicted) landmark to this observation
		observations[o].id = min_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a multivariate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// if no landmarks were observed, return
	if (observations.size() == 0) {
		cout << "**********************************  N O   L A N D M A R K S   O B S E R V E D  **********************************" << endl;
		return;
	}

	// set standard deviations for x and y (sensor noise)
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	long double weights_normalizer = 0.0;

	// loop over particles and weights
	for (unsigned int p = 0; p < num_particles; ++p) {

		Particle& particle = particles[p];
		long double weight = 1.0;

		// get all landmarks within sensor range of this particle
		std::vector<LandmarkObs> predicted;
		double max_dist = sensor_range + max(std_x, std_y);

		// loop over landmarks
		for (unsigned int l = 0; l < map_landmarks.landmark_list.size(); ++l) {

			LandmarkObs landmark;
			landmark.id = map_landmarks.landmark_list[l].id_i;
			landmark.x = map_landmarks.landmark_list[l].x_f;
			landmark.y = map_landmarks.landmark_list[l].y_f;
			assert (l == landmark.id - 1);

			// check distance to landmark
			double lm_dist = dist(particle.x, particle.y,
								  landmark.x, landmark.y);
			if (lm_dist <= max_dist) {
				predicted.push_back(landmark);
			}
		}

		// create vectors for associations
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;

		// consider case when no landmark is within sensor range of this particle
		if (predicted.size() == 0) {

			// set particle's weight to zero
			weight = 0.0;

			cout << "************  N O   L A N D M A R K S   F O U N D   W I T H I N   R A N G E   O F   P A R T I C L E  ************" << endl;

		} else {
			// transform observations from particles coordinate system to the map c/s
			std::vector<LandmarkObs> observations_map;

			// pre-calculations for homogeneous transformation
			double sin_theta = sin(particle.theta);
			double cos_theta = cos(particle.theta);

			// loop over observations
			for (unsigned int o = 0; o < observations.size(); ++o) {

				// apply homogeneous transformation from car to map coordinate system
				double x_map = particle.x + cos_theta * observations[o].x
									  	  - sin_theta * observations[o].y;
				double y_map = particle.y + sin_theta * observations[o].x
										  + cos_theta * observations[o].y;
				LandmarkObs observ_map;
				observ_map.x = x_map;
				observ_map.y = y_map;
				observations_map.push_back(observ_map);
			}

			// find for each observation the nearest (predicted) landmark
			dataAssociation(predicted, observations_map);

			// loop over observations
			for (unsigned int o = 0; o < observations_map.size(); ++o) {

				// get the nearest (predicted) landmark for this observation
				double x_map = observations_map[o].x;
				double y_map = observations_map[o].y;
				int lm_id = observations_map[o].id;
				assert (lm_id <= map_landmarks.landmark_list.size());
				Map::single_landmark_s nearest_lm = map_landmarks.landmark_list[lm_id-1];
				assert (lm_id == nearest_lm.id_i);

				// store associations
				associations.push_back(lm_id);
				sense_x.push_back(x_map);
				sense_y.push_back(y_map);

				// calculate multivariate Gaussian distribution density for this observation
				double gauss_norm = 2. * M_PI * std_x * std_y;
				double delta_x = x_map - nearest_lm.x_f;
				double delta_y = y_map - nearest_lm.y_f;
				long double exponent = - ( delta_x * delta_x / (2. * std_x * std_x)
								         + delta_y * delta_y / (2. * std_y * std_y));
				weight *= exp(exponent) / gauss_norm;
			}
		}

		// set list of associations for this particle
		SetAssociations(particle, associations, sense_x, sense_y);

		// update weight for this particle
		weights[p] = weight;
		weights_normalizer += weight;
	}

	// check, if sum of all weights is zero, e.g. if no landmark is within the range or weights are to small (underflow)
	if (weights_normalizer == 0.0) {
		// restore all weights
		cout << "**************************************  R E S T O R I N G   W E I G H T S  **************************************" << endl;
		// loop over particles and weights
		for (unsigned int p = 0; p < num_particles; ++p) {
			weights[p] = particles[p].weight;
		}
	} else {
		// normalize weights
		// loop over particles and weights
		for (unsigned int p = 0; p < num_particles; ++p) {
			weights[p] /= weights_normalizer;
			particles[p].weight = weights[p];
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// define random number engine
	default_random_engine gen;

	// create discrete distribution for particle IDs based on weights
	discrete_distribution<int> dist_weights (weights.begin(), weights.end());

	std::vector<Particle> new_particles;
	std::vector<long double> new_weights;

	long double weights_normalizer = 0.0;

	// loop over particles and weights
	for (unsigned int i = 0; i < num_particles; ++i) {

		// sample new particle
		int p = dist_weights(gen);
		Particle particle = particles[p];
		particle.id = p;
		new_particles.push_back(particle);

		// append weight
		new_weights.push_back(particle.weight);
		weights_normalizer += particle.weight;
	}

	// replace particles and weights
	particles = new_particles;
	weights = new_weights;

	// normalize weights
	// loop over particles and weights
	for (unsigned int p = 0; p < num_particles; ++p) {
		weights[p] /= weights_normalizer;
		particles[p].weight = weights[p];
	}
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
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
