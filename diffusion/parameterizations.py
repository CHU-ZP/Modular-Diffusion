"""Prediction target adapters for epsilon, x0, and v parameterizations."""

from __future__ import annotations

from torch import Tensor

from .schedules import NoiseSchedule


class DiffusionParameterization:
    """Convert between model output semantics without changing samplers."""

    VALID_TARGETS = {"epsilon", "x0", "v"}

    def __init__(self, schedule: NoiseSchedule, prediction_target: str = "epsilon") -> None:
        prediction_target = prediction_target.lower()
        if prediction_target not in self.VALID_TARGETS:
            raise ValueError(
                f"prediction_target must be one of {sorted(self.VALID_TARGETS)}, "
                f"got {prediction_target!r}",
            )
        self.schedule = schedule
        self.prediction_target = prediction_target

    def _alpha_sigma(self, x_t: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        timesteps = timesteps.to(x_t.device)
        alpha = self.schedule.extract(self.schedule.sqrt_alpha_bars, timesteps, x_t.shape)
        sigma = self.schedule.extract(
            self.schedule.sqrt_one_minus_alpha_bars,
            timesteps,
            x_t.shape,
        )
        return alpha, sigma

    def target_from(self, x0: Tensor, noise: Tensor, x_t: Tensor, timesteps: Tensor) -> Tensor:
        if self.prediction_target == "epsilon":
            return noise
        if self.prediction_target == "x0":
            return x0
        alpha, sigma = self._alpha_sigma(x_t, timesteps)
        return alpha * noise - sigma * x0

    def model_output_to_epsilon(self, model_output: Tensor, x_t: Tensor, timesteps: Tensor) -> Tensor:
        if self.prediction_target == "epsilon":
            return model_output
        alpha, sigma = self._alpha_sigma(x_t, timesteps)
        if self.prediction_target == "x0":
            return (x_t - alpha * model_output) / sigma.clamp_min(1e-20)
        return sigma * x_t + alpha * model_output

    def model_output_to_x0(self, model_output: Tensor, x_t: Tensor, timesteps: Tensor) -> Tensor:
        if self.prediction_target == "x0":
            return model_output
        alpha, sigma = self._alpha_sigma(x_t, timesteps)
        if self.prediction_target == "epsilon":
            return (x_t - sigma * model_output) / alpha.clamp_min(1e-20)
        return alpha * x_t - sigma * model_output

    def model_output_to_v(self, model_output: Tensor, x_t: Tensor, timesteps: Tensor) -> Tensor:
        if self.prediction_target == "v":
            return model_output
        predicted_x0 = self.model_output_to_x0(model_output, x_t, timesteps)
        predicted_epsilon = self.model_output_to_epsilon(model_output, x_t, timesteps)
        alpha, sigma = self._alpha_sigma(x_t, timesteps)
        return alpha * predicted_epsilon - sigma * predicted_x0
