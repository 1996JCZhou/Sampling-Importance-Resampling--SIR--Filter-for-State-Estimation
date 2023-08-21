from particle_filter_base import ParticleFilter
from resampling_algos import *


class ParticleFilterSIR(ParticleFilter):

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm):
        """
        Initialize a SIR particle filter using the 'ParticleFilter' class.

        :param resampling_algorithm: Define a resampling algorithm (based on the selected resampling scheme 'needs_resampling').
        """

        # Initialize particle filter base class.
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Set SIR specific properties.
        self.resampling_algorithm = resampling_algorithm

    def needs_resampling(self):
        """
        This method determines whether or not a resampling step is needed for the current time step.

        :return: Boolean indicating whether or not resampling is needed.
                 (The SIR particle filter resamples every time step, hence it always returns true.)
        """
        return True

    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process prediction, update, normalization and resampling for each particle.

        :param robot_forward_motion: Desired forward motion distance (m).
        :param robot_angular_motion: Desired rotation angle (rad) for the robot to perform.
        :param measurements:         Actual measurements of perturbed robot distance and angle in the new position 'z_k'.
        :param landmarks:            Landmark positions for calculating the expected measurements.
        """

        new_particles = []

        # Loop over all particles.
        for par in self.particles:

            """Prediction."""
            # 'par[1]':           A list of 3 state values ([x, y, heading angle]) of the particle before prediction.
            # 'propagated_state': A list of 3 state values ([x, y, heading angle]) of the particle after prediction.
            propagated_state = self.propagate_sample_Q(par[1], robot_forward_motion, robot_angular_motion, [0.07, 0.07, 0.1])

            """Update."""
            # 'par[0]': The weight of the particle before update.
            # 'weight': The weight of the particle after update.
            weight = par[0] * self.compute_likelihood(propagated_state, measurements, landmarks)

            # Store all new calculations in a list.
            new_particles.append([weight, propagated_state])

        """Particle weights normalization."""
        self.particles = self.normalize_weights(new_particles)

        """Check for resampling"""
        if self.needs_resampling():
            self.particles = resample(self.particles, self.n_particles, self.resampling_algorithm)
