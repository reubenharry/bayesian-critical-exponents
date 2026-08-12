"""Shared pytest fixtures for the probabilisticnumerics test suite."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seeded_rng():
    """Seed numpy at the start of every test for reproducibility."""
    np.random.seed(1234)
    yield
