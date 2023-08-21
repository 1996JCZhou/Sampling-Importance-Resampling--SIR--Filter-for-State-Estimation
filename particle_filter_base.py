from abc import ABC, abstractmethod
import copy
import numpy as np


class ParticleFilter(ABC):

    def __init__(self, number_of_particles, limits, process_noise, measurement_noise):
        """
        Initialize the basic particle filter.

        :param number_of_particles: Number of particles.
        :param limits:              List with maximum and minimum values for x and y dimension with [xmin (m), xmax (m), ymin (m), ymax (m)].
        :param process_noise:       A list of guessed standard deviation of additive zero mean Gaussian noise on [forward motion (m), rotation angles (rad)].
        :param measurement_noise:   A list of guessed standard deviation of additive zero mean Gaussian noise on [distance measurement (m), angle measurement (rad)].
        """

        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}!".format(number_of_particles))
        self.n_particles = number_of_particles

        self.state_dimension = 3  # The dimension of the state vector (which consists of x, y and heading angle).

        self.x_min = limits[0]
        self.x_max = limits[1]
        self.y_min = limits[2]
        self.y_max = limits[3]

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def particle_initialize_uniform_original_state_unknown(self):
        """
        Initialize each particle uniformly over the world with a 3D state vector (x, y, heading),
        when we do not know the original state of the target object.
        """

        self.particles = []
        weight = 1.0 / self.n_particles

        for _ in range(self.n_particles):

            # Add each particle into the list.
            self.particles.append([weight,
                                   [np.random.uniform(self.x_min, self.x_max, 1)[0],
                                    np.random.uniform(self.y_min, self.y_max, 1)[0],
                                    np.random.uniform(0, 2 * np.pi, 1)[0]]])

    def particle_initialize_uniform_original_state_known(self, robot):
        """
        Initialize each particle uniformly over the world with a 3D state vector (x, y, heading),
        when we know the original state of the target object.
        """

        self.particles = []
        weight = 1.0 / self.n_particles

        for _ in range(self.n_particles):

            # Add each particle into the list.
            self.particles.append([weight, [robot.x, robot.y, robot.theta]])

    def validate_state(self, state):
        """
        Validate the state to stay within the 'cyclic' world.

        :param state: Input particle state.
        :return: Validated particle state.
        """

        # Make sure the roboter position do not exceed allowed limits (assuming cyclic world).
        state[0] = np.mod(state[0], self.x_max)
        state[1] = np.mod(state[1], self.y_max)

        # Make sure the roboter heading angle stays within [-pi, pi].
        state[2] = np.mod(state[2], 2*(np.pi))

        return state

    def get_average_state(self):
        """
        Compute average state according to all normalized-weighted particle states.

        :return: Average x-position, y-position and heading angle.
        """

        # Compute weighted average of particle states.
        avg_x = 0.0
        avg_y = 0.0
        avg_theta = 0.0
        for weighted_sample in self.particles:
            avg_x     += weighted_sample[0] * weighted_sample[1][0]
            avg_y     += weighted_sample[0] * weighted_sample[1][1]
            avg_theta += weighted_sample[0] * weighted_sample[1][2]

        return [avg_x, avg_y, avg_theta]

    @staticmethod
    def normalize_weights(weighted_samples):
        """
        Particle weights normalization.

        :param weighted_samples: A list of (unnormalized weight, particle state)-lists, whose element is the unnormalized weight and the state of each particle.
        :return                  A list of (normalized weight, particle state)-lists, whose element is the normalized weight and the state of each particle.
        """

        sum_weights = 0.0
        for weighted_sample in weighted_samples:
            sum_weights += weighted_sample[0]

        """Check for reinitialization."""
        # Check if the sum of all the particle weights is too small,
        # which means that all the particles are far away from the target (very poor estimation).
        if sum_weights < 1e-7:
            print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized).".format(sum_weights))

            # Reinitialize weights of all the particles uniformly.
            return [[1.0 / len(weighted_samples), weighted_sample[1]] for weighted_sample in weighted_samples]

        # Return normalized weight for each particle.
        return [[weighted_sample[0] / sum_weights, weighted_sample[1]] for weighted_sample in weighted_samples]

    def propagate_sample(self, sample, desired_distance, desired_rotation):
        """
        Propagate an individual particle based on its process model that assumes
        the robot first moves 'desired_distance' (m) in the direction of its original heading
        and then rotates 'desired_rotation' (rad). Return the propagated particle state.
        (Old particle state → Prediction: process model + desired movements → Estimated particle state)

        A process model that encodes prior knowledge on how the state x_k is expected to evolve over time.

        :param sample:           A list of 3 state values ([x, y, heading angle]) of the particle before prediction.
        :param desired_distance: Desired forward motion distance (m).
        :param desired_rotation: Desired rotation angle (rad) for the robot to perform.
        :return:                 A list of 3 state values ([x, y, heading angle]) of the particle after prediction.
        """

        propagated_sample = copy.deepcopy(sample)

        # 1. move forward.
        # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
        forward_displacement = np.random.normal(desired_distance, self.process_noise[0], 1)[0]
        propagated_sample[0] += forward_displacement * np.cos(propagated_sample[2])
        propagated_sample[1] += forward_displacement * np.sin(propagated_sample[2])

        # 2. rotate by given amount plus additive noise sample (index 1 is angular noise standard deviation)
        propagated_sample[2] += np.random.normal(desired_rotation, self.process_noise[1], 1)[0]

        # Make sure we stay within cyclic world.
        return self.validate_state(propagated_sample)

    def propagate_sample_Q(self, sample, desired_distance, desired_rotation, Q):
        """
        Propagate an individual particle based on its process model that assumes
        the robot first moves 'desired_distance' (m) in the direction of its original heading
        and then rotates 'desired_rotation' (rad). Return the propagated particle state.
        (Old particle state → Prediction: process model + desired movements → Estimated particle state)

        A process model that encodes prior knowledge on how the state x_k is expected to evolve over time.

        :param sample:           A list of 3 state values ([x, y, heading angle]) of the particle before prediction.
        :param desired_distance: Desired forward motion distance (m).
        :param desired_rotation: Desired rotation angle (rad) for the robot to perform.
        :Q:                      A list of guessed standard deviations of zero mean Gaussian additive noise
                                 on [moving along x-axis (m), moving along y-axis (m), turning actions (rad)].
        :return:                 A list of 3 state values ([x, y, heading angle]) of the particle after prediction.
        """

        propagated_sample = []

        mean = [sample[0] + desired_distance * np.cos(sample[2]),
                sample[1] + desired_distance * np.sin(sample[2]),
                sample[2] + desired_rotation]

        cov = [[Q[0]**2, 0,       0      ],
               [0,       Q[1]**2, 0      ],
               [0,       0,       Q[2]**2]]

        multi_normal = np.random.multivariate_normal(mean, cov, 1)
        propagated_sample.append(multi_normal[0][0])
        propagated_sample.append(multi_normal[0][1])
        propagated_sample.append(multi_normal[0][2])

        return self.validate_state(propagated_sample)

    def compute_likelihood(self, sample, measurement, landmarks):
        """
        Compute likelihood for the current state.
        (Estimated particle state → Update: measurement model + measurements → Updated particle state)

        :param sample:      A list of 3 state values ([x, y, heading angle]) of the particle after prediction.
        :param measurement: Actual measurements of perturbed robot distance and angle in the new position 'z_k'.
        :param landmarks:   Absolute positions of landmarks in the world (m).
        :return             Likelihood based on all landmark measurements.
        """

        likelihood_sample = 1.0

        # Loop over each landmark.
        for i, lm in enumerate(landmarks):

            """Expected measurements using the measurement model for each independent dimension."""
            dx = sample[0] - lm[0]
            dy = sample[1] - lm[1]
            expected_distance = np.sqrt(dx*dx + dy*dy)
            expected_angle = np.arctan2(dy, dx)

            """Likelihood for each independent dimension."""
            p_z_given_expected_distance = \
                np.exp(-(measurement[i][0] - expected_distance) ** 2 / (2 * self.measurement_noise[0] ** 2))

            p_z_given_expected_angle = \
                np.exp(-(measurement[i][1] - expected_angle) ** 2 / (2 * self.measurement_noise[1] ** 2))

            likelihood_sample *= p_z_given_expected_distance * p_z_given_expected_angle

        return likelihood_sample

    @abstractmethod
    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement. Abstract method that must be implemented in derived
        class.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        pass
