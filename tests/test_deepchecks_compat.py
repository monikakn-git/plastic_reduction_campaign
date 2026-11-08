import numpy as np
from sklearn.metrics import get_scorer

from src.models.deepchecks_quality import prepare_deepchecks_compatibility


def test_deepchecks_legacy_numpy_and_scorer_aliases():
    prepare_deepchecks_compatibility()

    assert np.Inf == np.inf
    assert get_scorer("max_error") is not None
