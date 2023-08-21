import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def __init__(self, radius_robot, landmark_size):
    
        self.circle_radius_robot = radius_robot

        self.landmark_size = landmark_size

        self.x_margin = 1
        self.y_margin = 1
        self.scale = 2

    def draw_world(self, world, robot, particles, average_state=None, i=None, hold_on=False, particle_color='r'):
        """
        Draw a world with landmarks, a robot with its pose (position x, y and orientation heading angle) and
        particles with their poses to represent the discrete probability distribution for estimation.

        :param world:          World object with its dimensions and landmarks.
        :param robot:          Robot object with its true (perturbed) pose (position x, y and orientation heading angle).
        :param particles:      Weighted particles (a list of [weight, [x, y, heading angle]]-lists).
        :param average_state:  The averaged position among all the weighted particles.
        :param i:              The current time step.
        :param hold_on:        Boolean indicating whether figure must be kept or not.
        :param particle_color: Color used for particles
        """

        """Begin drawing."""
        x_min = -self.x_margin
        x_max = self.x_margin + world.x_max
        y_min = -self.y_margin
        y_max = self.y_margin + world.y_max

        plt.figure(num=1, figsize=((x_max-x_min) / self.scale, (y_max-y_min) / self.scale), frameon=False)
        if not hold_on:
            plt.clf() # Clear the current drawings.

        ax=plt.gca()  #plt.gca(): get current axis.
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
## -----------------------------------------------------------------------------------------
        """Draw a world."""
        plt.plot([0, world.x_max], [0, 0], color='k', linewidth=1, linestyle='-')                      # lower line: (0, 0) → (world.x_max, 0).
        plt.plot([0, 0], [0, world.y_max], color='k', linewidth=1, linestyle='-')                      # left line:  (0, 0) → (0, world.y_max).
        plt.plot([0, world.x_max], [world.y_max, world.y_max], color='k', linewidth=1, linestyle='-')  # top line:   (0, world.y_max) → (world.x_max, world.y_max).
        plt.plot([world.x_max, world.x_max], [0, world.y_max], color='k', linewidth=1, linestyle='-')  # right line: (world.x_max, 0) → (world.x_max, world.y_max).

        # Set axes limits.
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        # No ticks on axes.
        plt.xticks([])
        plt.yticks([])

        # Set title.
        plt.title("Green: Randomly moving robot (line for heading).\nRed:     Average position over all {} particles.\nBlue:    {} landmarks.\nCurrent time step: {}.".\
            format(len(particles), len(world.landmarks), i), horizontalalignment='left')

        plt.text(20, 20.3, "A cyclic {}*{} (m) world.".\
            format(world.x_max, world.y_max), fontsize=10, horizontalalignment="right")
## -----------------------------------------------------------------------------------------
        """Add landmarks."""
        landmarks = np.array(world.landmarks)
        plt.plot(landmarks[:, 0], landmarks[:, 1], 'bs', markersize=self.landmark_size)
## -----------------------------------------------------------------------------------------
        """Add particles."""
        if average_state != None:
            plt.plot(average_state[0], average_state[1], particle_color+'.', markersize=14)
        else:
            states = np.array([np.array(particle[1]) for particle in particles])
            plt.plot(states[:, 0], states[:, 1], particle_color+'.', markersize=1)
## -----------------------------------------------------------------------------------------
        """Add robot."""
        self.add_pose(robot.x, robot.y, robot.theta, 'g', self.circle_radius_robot)
## -----------------------------------------------------------------------------------------
        print("Deviation distance between actual and estimated robot positions is {} .".\
            format(np.sqrt((robot.x-average_state[0])**2+(robot.y-average_state[1])**2)))
        print("Deviation in x is {} .".format(robot.x-average_state[0]))
        print("Deviation in y is {} .".format(robot.y-average_state[1]))
        print()
        plt.text(0, 20.3, "Distance deviation is {} (m).".\
            format(round(np.sqrt((robot.x-average_state[0])**2+(robot.y-average_state[1])**2), 2)), fontsize=10, horizontalalignment="left")

        plt.text(10, -0.6, "All the particles are initialized with uniform weight and random state (without prior robot knowledge).", horizontalalignment="center")

    def add_pose(self, x, y, theta, color, radius):
        """
        Plot a robot pose (position x, y and orientation heading angle) in the figure
        with given color and radius (circle with line indicating heading).

        :param x:      X-position (circle center).
        :param y:      Y-position (circle center).
        :param theta:  Heading angle (a line from circle center with this heading angle will be added).
        :param color:  Color of the lines.
        :param radius: Radius of the circle.
        """

        # Define a circle as the robot at the given position.
        circle = plt.Circle((x, y), radius, facecolor=color, edgecolor=color, alpha=0.5, zorder=100)
        plt.gca().add_patch(circle) # Apply the rendering function to display the circle.

        # Draw line indicating heading angle.
        plt.plot([x, x + radius * np.cos(theta)], [y, y + radius * np.sin(theta)], color=color, linewidth=1, linestyle='-')

    def display_world(self, world):
        """
        Draw a world with landmarks, a robot with its pose (position x, y and orientation heading angle) and
        particles with their poses to represent the discrete probability distribution for estimation.

        :param world:          World object with its dimensions and landmarks.
        """

        """Begin drawing."""
        x_min = -self.x_margin
        x_max = self.x_margin + world.x_max
        y_min = -self.y_margin
        y_max = self.y_margin + world.y_max

        plt.figure(num=1, figsize=((x_max-x_min) / self.scale, (y_max-y_min) / self.scale), frameon=False)

        ax=plt.gca()  #plt.gca(): get current axis.
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
## -----------------------------------------------------------------------------------------
        """Draw a world."""
        plt.plot([0, world.x_max], [0, 0], color='k', linewidth=1, linestyle='-')                      # lower line: (0, 0) → (world.x_max, 0).
        plt.plot([0, 0], [0, world.y_max], color='k', linewidth=1, linestyle='-')                      # left line:  (0, 0) → (0, world.y_max).
        plt.plot([0, world.x_max], [world.y_max, world.y_max], color='k', linewidth=1, linestyle='-')  # top line:   (0, world.y_max) → (world.x_max, world.y_max).
        plt.plot([world.x_max, world.x_max], [0, world.y_max], color='k', linewidth=1, linestyle='-')  # right line: (world.x_max, 0) → (world.x_max, world.y_max).

        # Set axes limits.
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        # No ticks on axes.
        plt.xticks([])
        plt.yticks([])

        plt.text(20, 20.3, "A cyclic {}*{} (m) world.".\
            format(world.x_max, world.y_max), fontsize=10, horizontalalignment="right")
## -----------------------------------------------------------------------------------------
        """Add landmarks."""
        landmarks = np.array(world.landmarks)
        plt.plot(landmarks[:, 0], landmarks[:, 1], 'bs', markersize=self.landmark_size)
