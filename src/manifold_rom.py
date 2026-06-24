from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .autoencoder import SnapshotAutoencoder, SnapshotScaler
from .newton import NewtonResult, solve_nonlinear_system


@dataclass
class ManifoldGalerkinROM:
    autoencoder: SnapshotAutoencoder
    scaler: SnapshotScaler
    problem: object
    solver_tol: float = 1e-8
    solver_max_iter: int = 25
    solver_line_search_steps: int = 8
    solver_min_step_scale: float = 1e-3

    @property
    def reduced_dim(self) -> int:
        return self.autoencoder.latent_dim

    def encode(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normalized = self.scaler.transform_tensor(torch.tensor(state, dtype=torch.float64))
            latent = self.autoencoder.encode(normalized)
        return latent.detach().cpu().numpy()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            decoded = self._decode_tensor(torch.tensor(latent, dtype=torch.float64))
        return decoded.detach().cpu().numpy()

    def _decode_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        normalized_state = self.autoencoder.decode(latent)
        return self.scaler.inverse_transform_tensor(normalized_state)

    def reduced_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)

        def decode_physical(z: torch.Tensor) -> torch.Tensor:
            return self._decode_tensor(z)

        state = decode_physical(latent)
        tangent = torch.autograd.functional.jacobian(decode_physical, latent, create_graph=True)
        ambient_residual = self.problem.residual_torch(state, mu)
        return tangent.T @ ambient_residual

    def solve(self, mu, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = torch.zeros(self.reduced_dim, dtype=torch.float64)
        else:
            x0 = torch.tensor(initial, dtype=torch.float64)
        return solve_nonlinear_system(
            lambda z: self.reduced_residual_torch(z, mu),
            x0,
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            line_search_steps=self.solver_line_search_steps,
            min_step_scale=self.solver_min_step_scale,
        )


@dataclass
class ManifoldLSPGROM:
    autoencoder: SnapshotAutoencoder
    scaler: SnapshotScaler
    problem: object
    solver_tol: float = 1e-8
    solver_max_iter: int = 25
    solver_line_search_steps: int = 8
    solver_min_step_scale: float = 1e-3

    @property
    def reduced_dim(self) -> int:
        return self.autoencoder.latent_dim

    def encode(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normalized = self.scaler.transform_tensor(torch.tensor(state, dtype=torch.float64))
            latent = self.autoencoder.encode(normalized)
        return latent.detach().cpu().numpy()

    def decode(self, latent: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            decoded = self._decode_tensor(torch.tensor(latent, dtype=torch.float64))
        return decoded.detach().cpu().numpy()

    def _decode_tensor(self, latent: torch.Tensor) -> torch.Tensor:
        normalized_state = self.autoencoder.decode(latent)
        return self.scaler.inverse_transform_tensor(normalized_state)

    def ambient_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)
        state = self._decode_tensor(latent)
        return self.problem.residual_torch(state, mu)

    def optimality_residual_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)

        def ambient_residual(local_latent: torch.Tensor) -> torch.Tensor:
            return self.ambient_residual_torch(local_latent, mu)

        jacobian = torch.autograd.functional.jacobian(ambient_residual, latent, create_graph=True)
        residual = ambient_residual(latent)
        return jacobian.T @ residual

    def ambient_residual_norm_torch(self, latent: torch.Tensor, mu) -> torch.Tensor:
        residual = self.ambient_residual_torch(latent, mu)
        return 0.5 * torch.dot(residual, residual)

    def solve(self, mu, *, initial: np.ndarray | None = None) -> NewtonResult:
        if initial is None:
            x0 = torch.zeros(self.reduced_dim, dtype=torch.float64)
        else:
            x0 = torch.tensor(initial, dtype=torch.float64)
        return solve_nonlinear_system(
            lambda z: self.optimality_residual_torch(z, mu),
            x0,
            merit_fn=lambda z: self.ambient_residual_norm_torch(z, mu),
            tol=self.solver_tol,
            max_iter=self.solver_max_iter,
            line_search_steps=self.solver_line_search_steps,
            min_step_scale=self.solver_min_step_scale,
        )
