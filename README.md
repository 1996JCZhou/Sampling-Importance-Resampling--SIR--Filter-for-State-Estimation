# Sampling-Importance-Resampling--SIR--Filter-for-State-Estimation

## Introduction

In the realm of state estimation, I've delved into recursivly tracking the evolving state of a dynamic system. This involves updating estimates with new measurements provided by sensors that measure state-related quantities. To tackle this challenge, I have used particle filtering, a technique that has proven its power since its emergence in the early 1990s. Particle filtering stands out due to its versatility in accommodating both linear and nonlinear process and measurement models, unlike other techniques such as the Kalman filter.

At its core, particle filtering operates from a Bayesian standpoint. The state vector is treated as a random variable, and a probability distribution, known as belief, encapsulates the estimate's uncertainty. Employing Bayes'theorem, belief is refined based on prior estimates and incoming measurements.

Effective state estimation hinges on two critical sources: a process model that contains prior knowledge about state evolution and a measurement model that correlates measurements with the state. These sources are formalized using mathematical models, often nonlinear in nature. To exemplify these concepts, I've chosen a robot localization example from [Particle Filters: A Hands-On Tutorial](https://www.mdpi.com/1424-8220/21/2/438) as my guiding problem. The scenario involves a robot navigating a 2D world, with the state vector encompassing the robot's 2D position in an x-y coordinate system measured in meters and its orientation in radians. This orientation, also referred to as the heading angle, guides the robot's movements between four detected landmarks, where it is assumed to advance a specific distance and then rotate an angle.

The state estimation problem can be addressed by the particle filter as a Bayesian estimation problem. This technique approximates the posterior probability distribution function (pdf) with a discrete pdf, ensuring minimal constraints on the underlying models. With prediction and filtering steps accomplished, the expected value vector from this discrete distribution over 3000 particles becomes the representative estimate of the robot's pose.

## My learning process

Importance Sampling → Sequence Importance Sampling Filter (SIS Filter) / Basic Particle Filter

Basic Particle Filter → Basic Particle Filter + Resampling: To solve particle degeneracy problem

Basic Particle Filter + Resampling → Basic Particle Filter + Resampling: 1. When to resample; 2. How to resample.
