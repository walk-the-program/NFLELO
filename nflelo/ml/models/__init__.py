"""Game models. `logistic` holds the M3 walk-forward logistic regression (fit once per season, ties as half rows)."""
from .logistic import fit_logistic, season_weights, walk_forward  # noqa: F401
