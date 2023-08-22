# Sampling Importance Resampling (SIR) Filter for State Estimation

## Introduction

In the realm of state estimation, I've delved into recursivly tracking the evolving state of a dynamic system. This involves updating estimates with new measurements provided by sensors that measure state-related quantities. To tackle this challenge, I have used particle filtering, a technique that has proven its power since its emergence in the early 1990s. Particle filtering stands out due to its versatility in accommodating both linear and nonlinear process and measurement models, unlike other techniques such as the Kalman filter.

At its core, particle filtering operates from a Bayesian standpoint. The state vector is treated as a random variable, and a probability distribution, known as belief, encapsulates the estimate's uncertainty. Employing Bayes'theorem, belief is refined based on prior estimates and incoming measurements.

Effective state estimation hinges on two critical sources: a process model that contains prior knowledge about state evolution and a measurement model that correlates measurements with the state. These sources are formalized using mathematical models, often nonlinear in nature. To exemplify these concepts, I've chosen a robot localization example from [Particle Filters: A Hands-On Tutorial](https://www.mdpi.com/1424-8220/21/2/438) as my guiding problem. The scenario involves a robot navigating a 2D world, with the state vector encompassing the robot's 2D position in an x-y coordinate system measured in meters and its orientation in radians. This orientation, also referred to as the heading angle, guides the robot's movements between four detected landmarks, where it is assumed to advance a specific distance and then rotate an angle. The green circle represents the 2D robot position, the green line within the circle represents the robot’s heading, the blue rectangles represent the landmarks.

![image](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/images%20for%20Readme/world.PNG)

The state estimation problem can be addressed by the particle filter as a Bayesian estimation problem. This technique approximates the posterior probability distribution function (pdf) with a discrete pdf, ensuring minimal constraints on the underlying models. With prediction and filtering steps accomplished, the expected value vector from this discrete distribution over 3000 particles becomes the representative estimate of the robot's pose.

![image](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/images%20for%20Readme/result.PNG)

## Requirements

- python
- matplotlib
- time
- random
- numpy
- abc
- copy

## My learning process

Importance Sampling → [Sequence Importance Sampling Filter (SIS Filter) / Basic Particle Filter](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/particle_filter_base.py)

[Sequence Importance Sampling Filter (SIS Filter) / Basic Particle Filter](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/particle_filter_base.py) → [Basic Particle Filter + Resampling](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/particle_filter_sir.py): To solve particle degeneracy problem

[Basic Particle Filter + Resampling](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/particle_filter_sir.py) → Basic Particle Filter + Resampling:

1. When to resample → 3 Strategies: [Resampling every step](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/particle_filter_sir.py), [Max Weight Resampling](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/particle_filter_max_weight_resampling.py) and [Number of Effective Particles Resampling](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/particle_filter_nepr.py);
2. [How to resample](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/resampling_algos.py) → 3 Strategies: Multinomial Sampling, Stratified Sampling and Systematic Sampling.

Please read [the tutorial paper](https://www.mdpi.com/1424-8220/21/2/438) for more details and check [my notes](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/tree/master/Notes) as well. :)

## Results

1. Filter for state estimation:

   **SIR-Filter (Condensation Filter) + Max Weight Resampling (when to resample) + Stratified Sampling (how to resample) +
   Particle Initialization with initial robot state**.

   Please check my video for this filter under https://www.youtube.com/watch?v=DuiE1aSiS5E.

   [![Watch the video](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/images%20for%20Readme/youtube2.PNG)](https://www.youtube.com/watch?v=DuiE1aSiS5E)

3. Filter for state estimation:

   **SIR-Filter (Condensation Filter) + Max Weight Resampling (when to resample) + Stratified Sampling (how to resample) + Random particle Initialization without prior robot state knowledge**.

   Please check my video for this filter under https://www.youtube.com/watch?v=3K34nm6k5rg.

   [![Watch the video](https://github.com/1996JCZhou/Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation/blob/master/images%20for%20Readme/youtube2.PNG)](https://www.youtube.com/watch?v=3K34nm6k5rg)

## Next challenges and tasks

This repository is still running and my journey in researching the particle filter is also not finished. I want to face the following challenges in the future:
1. Introduce the SIR-Filter in the [Multiple Object Tracking project](https://github.com/1996JCZhou/Multiple-Objects-Tracking) for state estimation instead of the Kalman filter. With particle filter, we no longer need to limit ourselves to linear process and measurement models for pedestrains on the street.
2. Beside the Particle Degeneracy Problem, there are still challenges that must be addressed, like Sample Impoverishment, Particle Filter Divergence and Real Time Execution for real time usages.
