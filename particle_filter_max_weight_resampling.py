from particle_filter_sir import ParticleFilterSIR


class ParticleFilterMWR(ParticleFilterSIR):
    """Apply Max Weight Resampling (MWR) if the reciprocal of the maximum weight drops below a specific value."""

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm,
                 resampling_threshold=1/0.005):
        """
        Initialize a particle filter that performs resampling whenever the reciprocal of the maximum particle weight
        among all the particle weights falls below a user-specified threshold value.

        :param resampling_threshold: Define the user-specified threshold value.
        """

        # Initialize SIR particle filter class.
        ParticleFilterSIR.__init__(self, number_of_particles, limits, process_noise, measurement_noise, resampling_algorithm)

        self.resampling_threshold = resampling_threshold

    def needs_resampling(self):
        """
        This method that determines whether or not a resampling step is needed for the current particle filter state
        estimate. Resampling only occurs if the reciprocal of the maximum particle weight falls below the user-specified
        threshold.

        :return: Boolean indicating whether or not resampling is needed.
        """

        max_weight = 0

        # Loop over all the normalized particle weights.
        for par in self.particles:
            max_weight = max(max_weight, par[0])

        return (1.0 / max_weight) < self.resampling_threshold
