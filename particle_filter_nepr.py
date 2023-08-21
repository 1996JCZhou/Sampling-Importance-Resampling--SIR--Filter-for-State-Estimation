from particle_filter_sir import ParticleFilterSIR


class ParticleFilterNEPR(ParticleFilterSIR):
    """Apply approximate Number of Effective Particles Resampling (NEPR)."""

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm,
                 number_of_effective_particles_threshold):
        """
        Initialize a particle filter that performs resampling whenever the approximated number of effective particles
        falls below a user-specified threshold value.

        :param number_of_effective_particles_threshold: Define the user-specified threshold value.
        """

        # Initialize SIR particle filter class.
        ParticleFilterSIR.__init__(self, number_of_particles, limits, process_noise, measurement_noise, resampling_algorithm)

        self.resampling_threshold = number_of_effective_particles_threshold

    def needs_resampling(self):
        """
        This method determines whether or not a resampling is needed for the current particle filter state
        estimate. Resampling only occurs if the approximated number of effective particles falls below the
        user-specified threshold.

        :return: Boolean indicating whether or not resampling is needed.
        """

        sum_weights_squared = 0
        for par in self.particles:
            sum_weights_squared += par[0] ** 2

        return (1.0 / sum_weights_squared) < self.resampling_threshold
