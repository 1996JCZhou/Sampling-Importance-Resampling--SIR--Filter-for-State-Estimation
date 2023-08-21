import numpy as np
from .world import *


class Robot:

    def __init__(self, x, y, theta, process_noise, measurement_noise):
        """
        ---Initialize a robot with initial position and its heading angle.---
        :param x:                 Initial robot x-position    (m).
        :param y:                 Initial robot y-position    (m).
        :param theta:             Initial robot heading angle (rad).

        ---Define the process noise.---
        :param std_forward:       Actual standard deviation of additive zero mean Gaussian noise on moving forward (m).
        :param std_turn:          Actual standard deviation of additive zero mean Gaussian noise on turning actions (rad).

        ---Define the measurement noise.---
        :param std_meas_distance: Actual standard deviation of additive zero mean Gaussian noise on distance measurement (m).
        :param std_meas_angle:    Actual standard deviation of additive zero mean Gaussian noise on angle measurement (rad).
        """

        self.x = x
        self.y = y
        self.theta = theta

        self.std_forward = process_noise[0]
        self.std_turn = process_noise[1]

        self.std_meas_distance = measurement_noise[0]
        self.std_meas_angle = measurement_noise[1]

    def move(self, desired_distance, desired_rotation, world):
        """
        Move the robot according to given arguments and within the world of given dimensions.
        The true motion is the sum of the desired motion and additive Gaussian noise that represents the fact that
        the desired motion cannot exactly be realized, e.g., due to imperfect control and sensing.

        :param desired_distance: desired forward motion distance of the robot (m).
        :param desired_rotation: desired rotation angle (rad).
        :param world:            the cyclic world, where the robot executes its motion.
        """

        # Compute true forward distance of the robot.
        distance_driven = self._get_gaussian_noise_sample(desired_distance, self.std_forward)

        # First move the robot along its heading angle for the last time point.
        self.x += distance_driven * np.cos(self.theta)
        self.y += distance_driven * np.sin(self.theta)

        # Positions restricted in the world with the cyclic world assumption.
        self.x = np.mod(self.x, world.x_max)
        self.y = np.mod(self.y, world.y_max)
    ## -------------------------------------------------------------------------------------------
        # Compute true rotation angle of the robot.
        angle_rotated = self._get_gaussian_noise_sample(desired_rotation, self.std_turn)

        # Then update the heading angle for the current time point.
        self.theta += angle_rotated

        # Angles restricted in [0, 2*pi].
        self.theta = np.mod(self.theta, 2*(np.pi))

    def measure(self, world):
        """
        Perform a measurement.
        The robot is assumed to measure the distance to and angles
        with respect to all landmarks in meters and radians respectively.
        While doing so, the robot experiences zero mean additive Gaussian noise.

        :param world: World containing the landmark positions.
        :return: A list of lists: [[dist_to_landmark1, angle_wrt_landmark1], dist_to_landmark2, angle_wrt_landmark2], ...]
        """

        measurements = []

        # Loop over each landmark for measurements.
        for lm in world.landmarks:
            dx = self.x - lm[0]
            dy = self.y - lm[1]

            # Measured distance perturbed by zero mean additive Gaussian noise.
            z_distance = self._get_gaussian_noise_sample(np.sqrt(dx * dx + dy * dy), self.std_meas_distance)
        ## -------------------------------------------------------------------------------------------
            # Measured angle perturbed by zero mean additive Gaussian noise.
            z_angle = self._get_gaussian_noise_sample(np.arctan2(dy, dx), self.std_meas_angle)

            # Store measurement
            measurements.append([z_distance, z_angle])

        return measurements

    @staticmethod
    def _get_gaussian_noise_sample(mu, sigma):
        """
        Draw a random sample from an 1D Gaussian distribution with mean mu and standard deviation sigma.

        :param mu:    mean               of the 1D Gaussian distribution.
        :param sigma: standard deviation of the 1D Gaussian distribution.
        :return:      A random sample    from the 1D Gaussian distribution.
        """
        return np.random.normal(loc=mu, scale=sigma, size=1)[0]
