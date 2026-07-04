# Samplers

Available samplers:

- `DDPMSampler`: stochastic ancestral sampling over all training timesteps.
- `DDIMSampler`: deterministic or partially stochastic sampling over a timestep
  subsequence.

Additional samplers such as Euler, Heun, or DPM-Solver can call the same
parameterization adapter instead of assuming epsilon prediction.
