"""
detect_logic.py — Phase 1 upgrade
Drop-in replacement. No changes needed in app.py or liveapp.py.
Automatically uses ensemble if lstm_trainer/ae_trainer are in the package.
Falls back to XGBoost-only with the old PKL.
"""

from ensemble import get_predictions  # noqa: F401
