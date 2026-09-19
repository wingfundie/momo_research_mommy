from __future__ import annotations

# Forecast scalars aligned with Robert Carver-style scaling.

CARVER_EWMAC_FORECAST_SCALARS: dict[tuple[int, int], float] = {
    (2, 8): 10.6,
    (4, 16): 7.5,
    (8, 32): 5.3,
    (16, 64): 3.75,
    (32, 128): 2.65,
    (64, 256): 1.87,
}

CARVER_BREAKOUT_FORECAST_SCALARS: dict[int, float] = {
    10: 0.60,
    20: 0.67,
    40: 0.70,
    80: 0.73,
    160: 0.74,
    320: 0.74,
}

