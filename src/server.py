import numpy as np
from tqdm import tqdm

import client

from . import classifiers, data_helper


def run_tds(limit: int = None) -> tuple:
    """
    Starts the simulated Threat Detection System

    Args:
        limit (int, optional): Limit how many tests should be classified. Defaults to None.

    Returns:
        tuple: b_ct_arr, b_rf_arr, b_nrf_arr, is_threat_arr, results of threat detection.
    """
    training_set = data_helper.get_training_set()
    testing_set, y_test = data_helper.normalize_data(data_helper.get_testing_set())
    b_ct_arr = []
    b_rf_arr = []
    b_nrf_arr = []
    is_threat_arr = []
    classifier = classifiers.Classifier(training_set)
    encrypted_testing_set = client.encrypt_set(
        testing_set, classifier.get_hf(), limit=limit
    )
    if limit is None:
        limit = np.shape(testing_set)[0]
    for i in tqdm(range(0, limit), desc="Running threat detection..."):
        b_rf_arr.append(classifier.random_forest(testing_set[i]))
        b_nrf_arr.append(classifier.neural_random_forest(testing_set[i]))
        b_ct_arr.append(classifier.cryptotree(encrypted_testing_set[i]))
        is_threat_arr.append(bool(y_test[i]))
    return b_ct_arr, b_rf_arr, b_nrf_arr, is_threat_arr
