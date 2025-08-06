# lse_hkr_loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import optax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import nnx


Reduction = Literal["none", "mean", "sum", "auto"]


@dataclass
class LseHKRMulticlassLoss:
    """JAX / Flax implementation of the LSE-HKR multiclass loss.

    Args
    ----
    alpha:        Interpolation between KR (0) and hinge (1).
    temperature:  Temperature scaling applied to the logits.
    penalty:      Controls the softness of the log-sum-exp approximation.
    min_margin:   Minimal margin used in both hinge and KR terms.
    reduction:    'none' | 'mean' | 'sum' | 'auto'  (auto == mean).
    """

    alpha: float = 1.0
    temperature: float = 1.0
    penalty: float = 1.0
    min_margin: float = 1.0
    reduction: Reduction = "mean"

    def __call__(self, logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        logits   : (B, C) array of raw, unnormalised scores.
        targets  : (B, C) one-hot or multi-hot array marking the positive class(es).

        Returns
        -------
        Either a scalar (if reduction != 'none') or a per-example vector (if 'none').
        """

        if jnp.ndim(targets) == 1:
            targets = jax.nn.one_hot(targets, logits.shape[1])

        logits = logits * self.temperature

        if self.alpha == 1.0:  # pure hinge
            loss_batch = self._lse_hinge(logits, targets)
        elif self.alpha == 0.0:  # pure KR
            loss_batch = self._lse_kr(logits, targets)
        else:  # interpolated HKR
            loss_batch = self._lse_hkr(logits, targets)

        return self._apply_reduction(loss_batch)

    def _get_pos(self, logits, targets):
        # Works for 1-hot (usual case) and multi-label (sums the true logits).
        return jnp.sum(logits * (targets > 0), axis=1)

    def _lse_neg(self, logits, targets):
        neg = jnp.where(targets > 0, -jnp.inf, logits)

        n_bins = logits.shape[1] - 1
        t = jnp.log(n_bins) / (self.min_margin * self.penalty / 2.0)

        return (1.0 / t) * logsumexp(t * neg, axis=1)  # shape (B,)

    # Hinge ------------------------------------------------------
    def _hinge_preproc(self, pos, lse_neg):
        half_m = self.min_margin / 2.0
        hinge_pos = jnp.maximum(half_m - pos, 0.0)
        hinge_neg = jnp.maximum(half_m + lse_neg, 0.0)
        return hinge_pos + hinge_neg  # shape (B,)

    def _lse_hinge(self, logits, targets):
        pos = self._get_pos(logits, targets)
        lse_neg = self._lse_neg(logits, targets)
        return self._hinge_preproc(pos, lse_neg)

    # Kantorovich–Rubinstein ------------------------------------
    def _kr_preproc(self, pos, lse_neg):
        return pos - lse_neg  # shape (B,)

    def _lse_kr(self, logits, targets):
        pos = self._get_pos(logits, targets)
        lse_neg = self._lse_neg(logits, targets)
        return -self._kr_preproc(pos, lse_neg)

    # Hybrid HKR -------------------------------------------------
    def _lse_hkr(self, logits, targets):
        pos = self._get_pos(logits, targets)
        lse_neg = self._lse_neg(logits, targets)

        lse_kr = -self._kr_preproc(pos, lse_neg)
        lse_hinge = self._hinge_preproc(pos, lse_neg)
        return (1.0 - self.alpha) * lse_kr + self.alpha * lse_hinge

    # Reduction --------------------------------------------------
    def _apply_reduction(self, loss_batch):
        if self.reduction in ("mean", "auto"):
            return jnp.mean(loss_batch)
        if self.reduction == "sum":
            return jnp.sum(loss_batch)
        if self.reduction == "none":
            return loss_batch
        raise ValueError(f"Unknown reduction '{self.reduction}'.")


class LossXEnt(nnx.Module):
    def __init__(self, offset, temperature):
        self.offset = offset
        self.temperature = temperature

    def __call__(self, logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        if jnp.ndim(targets) == 1:
            targets = jax.nn.one_hot(targets, logits.shape[1])
        offset_logits = logits - self.offset * targets
        offset_logits = offset_logits * self.temperature
        return optax.softmax_cross_entropy(logits, targets).mean() / self.temperature


class TauCCE(nnx.Module):
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
        if jnp.ndim(targets) == 1:
            targets = jax.nn.one_hot(targets, logits.shape[1])
        logits = logits * self.temperature
        return optax.softmax_cross_entropy(logits, targets).mean() / self.temperature
