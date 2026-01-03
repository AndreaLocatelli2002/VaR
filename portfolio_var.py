from __future__ import annotations

from statistics import NormalDist
from typing import Iterable

import numpy as np


def compute_returns(prices: Iterable[Iterable[float]]) -> np.ndarray:
    """Compute simple returns from price levels."""
    price_array = np.asarray(prices, dtype=float)
    if price_array.ndim != 2 or price_array.shape[0] < 2:
        raise ValueError("Prices must be a 2D array with at least two rows")
    return price_array[1:] / price_array[:-1] - 1.0


def compute_portfolio_returns(returns: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    weights_array = np.asarray(list(weights), dtype=float)
    if returns.shape[1] != weights_array.shape[0]:
        raise ValueError("Weights length must match number of assets")
    if not np.isclose(weights_array.sum(), 1.0):
        raise ValueError("Weights must sum to 1")
    return returns @ weights_array


def var_historical(portfolio_returns: np.ndarray, confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    percentile = np.percentile(portfolio_returns, (1 - confidence) * 100)
    return -float(percentile)


def var_parametric(portfolio_returns: np.ndarray, confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    mean = float(np.mean(portfolio_returns))
    std = float(np.std(portfolio_returns, ddof=1))
    if std == 0:
        return 0.0
    z_score = NormalDist().inv_cdf(1 - confidence)
    return -(mean + z_score * std)


def compute_portfolio_var(
    prices: Iterable[Iterable[float]],
    weights: Iterable[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Compute portfolio VaR from prices and weights.

    Args:
        prices: Price history with shape (time, assets).
        weights: Portfolio weights summing to 1.
        confidence: Confidence level (e.g., 0.95).
        method: "historical" or "parametric".
    """
    returns = compute_returns(prices)
    portfolio_returns = compute_portfolio_returns(returns, weights)

    if method == "historical":
        return var_historical(portfolio_returns, confidence)
    if method == "parametric":
        return var_parametric(portfolio_returns, confidence)

    raise ValueError("Method must be 'historical' or 'parametric'")
