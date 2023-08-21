import matplotlib.pyplot as plt
import numpy as np
import time, random

from simulator import Robot, Visualizer, World

from particle_filter_nepr                  import ParticleFilterNEPR
from particle_filter_max_weight_resampling import ParticleFilterMWR

if __name__ == '__main__':

    print("Running particle filter demo.")

    """Initialize a world."""
    world = World(20.0, 20.0, [[1.0, 1.0], [1.0, 19.0], [19.0, 1.0], [19.0, 19.0]])
## ------------------------------------------------------------------------------------------------
    """Initialize visualization."""
    visualizer = Visualizer(radius_robot=0.6, landmark_size=10)
    visualizer.display_world(world)
    plt.pause(5)
## ------------------------------------------------------------------------------------------------
    """Initialize a robot."""
    robot_moving_std  = 0.005  # Actual standard deviation of additive zero mean Gaussian noise on moving forward along the heading angle (m).
    robot_turning_std = 0.002  # Actual standard deviation of additive zero mean Gaussian noise on turning angles (rad).
    actual_process_noise = [robot_moving_std, robot_turning_std]

    robot_meas_noise_distance_std = 0.2  # Actual standard deviation of additive zero mean Gaussian noise on distance measurement (m).
    robot_meas_noise_angle_std    = 0.05 # Actual standard deviation of additive zero mean Gaussian noise on angle measurement (rad).
    actual_measurement_noise = [robot_meas_noise_distance_std, robot_meas_noise_angle_std]

    robot = Robot(x=world.x_max/2.0, # Robot initial x position.
                  y=world.y_max/2.0, # Robot initial y position.
                  theta=0,           # Robot initial heading angle (facing right).
                  process_noise=actual_process_noise,
                  measurement_noise=actual_measurement_noise)
## ------------------------------------------------------------------------------------------------
    """Initialize a particle filter."""
    particle_moving_std  = 0.1  # Guessed standard deviation of additive zero mean Gaussian noise on moving forward along the heading angle (m).
    particle_turning_std = 0.2  # Guessed standard deviation of additive zero mean Gaussian noise on turning angles (rad).
    guessed_process_noise = [particle_moving_std, particle_turning_std]

    particle_meas_noise_distance_std = 0.2  # Guessed standard deviation of additive zero mean Gaussian noise on distance measurement (m).
    particle_meas_noise_angle_std    = 0.1  # Guessed standard deviation of additive zero mean Gaussian noise on angle measurement (rad).
    guessed_measurement_noise = [particle_meas_noise_distance_std, particle_meas_noise_angle_std]

    # particle_filter = ParticleFilterNEPR(number_of_particles=3000,
    #                                      limits=[0, world.x_max, 0, world.y_max],
    #                                      process_noise=guessed_process_noise,
    #                                      measurement_noise=guessed_measurement_noise,
    #                                      resampling_algorithm='STRATIFIED', # 'MULTINOMIAL', 'STRATIFIED'.
    #                                      number_of_effective_particles_threshold=3000/4.0)

    particle_filter = ParticleFilterMWR(number_of_particles=3000,
                                        limits=[0, world.x_max, 0, world.y_max],
                                        process_noise=guessed_process_noise,
                                        measurement_noise=guessed_measurement_noise,
                                        resampling_algorithm='STRATIFIED') # 'MULTINOMIAL', 'STRATIFIED'.

    # Initialize the particles uniformly over the world with a 3D state (x, y, heading).
    particle_filter.particle_initialize_uniform_original_state_unknown()
## ------------------------------------------------------------------------------------------------
    random.seed(time.localtime().tm_sec)

    n_time_steps = 30 # Number of simulated time steps.
    for i in range(1, n_time_steps+1):
    ## ------------------------------------------------------------------------------------------------
        """Feed new measurements for updating particles in the current time step."""
        # Generate random desired forward motion distance (m) and rotation angle (rad) for the robot to perform.
        robot_motion_distance = random.uniform(-3, 3)                 # 'u_fwd_k'.
        robot_rotation_angles = random.uniform(-(np.pi)/3, (np.pi)/3) # 'u_ang_k'.
        # robot_motion_distance = 1        # 'u_fwd_k'.
        # robot_rotation_angles = np.pi/18 # 'u_ang_k'.

        # First move and then turn the robot.
        robot.move(desired_distance=robot_motion_distance,
                   desired_rotation=robot_rotation_angles,
                   world=world)

        # Measure perturbed robot distance and angle in the new position.
        measurements = robot.measure(world) # 'z_k'.
    ## ------------------------------------------------------------------------------------------------
        """First predicting and then updating of all the particles."""
        particle_filter.update(robot_forward_motion=robot_motion_distance,
                               robot_angular_motion=robot_rotation_angles,
                               measurements=measurements,
                               landmarks=world.landmarks)
    ## ------------------------------------------------------------------------------------------------
        """Visualization."""
        visualizer.draw_world(world, robot, particle_filter.particles, particle_filter.get_average_state(), i, hold_on=False, particle_color='r')
        plt.pause(0.5)
    plt.pause(2)
