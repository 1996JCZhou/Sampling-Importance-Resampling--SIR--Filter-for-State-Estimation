import numpy as np
import copy


def naive_search(cumulative_list, x):
    """
    Find the smallest index 'i' of the 'cumulative_list' list,
    for which 'x' <= 'cumulative_list[i]' holds.

    :param cumulative_list: List of elements that increase with increasing index, e.g. [0.1, 0.2, 0.9, 1.0].
    :param x:               Value to be checked.
    :return:                Index of the 'cumulative_list' list.
    """

    m = 0
    while cumulative_list[m] < x:
        m += 1
    return m


def resample(samples, N, algorithm):
    """
    Resampling interface, perform resampling using specified method.

    :param samples:   A list of (normalized weight, particle state)-lists before resampling.
    :param N:         Number of samples that must be resampled after resampling.
    :param algorithm: Preferred resampling method.
    :return:          A list of (uniform weight, particle state)-lists after resampling.
    """

    if algorithm == 'MULTINOMIAL':
        return multinomial(samples, N)
    elif algorithm == 'STRATIFIED':
        return stratified(samples, N)

def multinomial(samples, N):
    """
    Particles are sampled with replacement proportional to their weights and in arbitrary order.
    This leads to a maximum variance on the number of times a particle will be resampled,
    since any particle will be resampled between 0 and N times.

    :param samples: A list of (normalized weight, particle state)-lists before resampling.
    :param N:       Number of particles that must be resampled.
    :return:        A list of (normalized/uniform weight, particle state)-lists after resampling.
    """

    # Compute cumulative sum on normalized weights (which forms a discrete probability distribution).
    weights = [weighted_sample[0] for weighted_sample in samples]
    Q = np.cumsum(weights).tolist()

    # A list to store new particles with weights and states.
    new_samples = []

    for _ in range(N):

        # Draw a random sample 'u' from [0, 1).
        u = np.random.uniform(1e-10, 1, 1)[0]
    ## --------------------------------------------------------------------------------------
        # Find the smallest index 'm' of the cumulative list 'Q', for which
        # 'Q[m-1]' < 'u' <= 'Q[m]]' < 'Q[m+1]'.
        m = naive_search(Q, u)

        # Add copy of the state sample (uniform weights).
        new_samples.append([1.0/N, copy.deepcopy(samples[m][1])])

    return new_samples

def stratified(samples, N):
    """
    Stratified random sampling is a method of sampling,
    dividing a range of possibility [0, 1) into smaller strata
    with a range of [1e-10 + float(n) * 1 / N, 1.0 / N + float(n) * 1 / N) (integer n = 0, 1, 2, ...).

    :param samples: A list of (normalized weight, particle state)-lists before resampling.
    :param N:       Number of particles that must be resampled.
    :return:        A list of (normalized/uniform weight, particle state)-lists after resampling.
    """

    # Compute cumulative sum on normalized weights (which forms a discrete probability distribution).
    weights = [weighted_sample[0] for weighted_sample in samples]
    Q = np.cumsum(weights).tolist()

    n = 0
    new_samples = []
    while n < N:

        # Draw a random sample 'u0' from [0, 1.0/N].
        u0 = np.random.uniform(1e-10, 1.0 / N, 1)[0]

        # There is a random sample in every strata [1e-10 + float(n) * 1 / N, 1.0 / N + float(n) * 1 / N).
        # Integer n = 0, 1, 2, ...
        u = u0 + float(n) * 1 / N
    ## --------------------------------------------------------------------------------------
        # Find the smallest index 'm' of the cumulative list 'Q', for which
        # 'Q[m-1]' < 'u' <= 'Q[m]' < 'Q[m+1]'.
        m = naive_search(Q, u)

        # Add copy of the state sample (uniform weights).
        new_samples.append([1.0/N, copy.deepcopy(samples[m][1])])

        # Added another sample
        n += 1

    return new_samples
