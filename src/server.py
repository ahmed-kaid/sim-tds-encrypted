import numpy as np
from tqdm import tqdm

from . import classifiers, data_helper


class Server:
    def __init__(self) -> None:
        self.training_set = data_helper.get_training_set()
        self.testing_set, self.y_test = data_helper.normalize_data(
            data_helper.get_testing_set()
        )
        self.b_ct_arr = []
        self.b_rf_arr = []
        self.b_nrf_arr = []
        self.is_threat_arr = []
        self.classifier = classifiers.Classifier(self.training_set)

    def get_training_set(self):
        return self.training_set

    def get_homomorphic_featurizer(self):
        return self.classifier.__homomorphic_featurizer

    def run_tds(self, limit: int = None) -> None:
        """
        Starts the simulated Threat Detection System

        Args:
            limit (int, optional): Limit how many tests should be classified. Defaults to None.

        Returns:
            tuple: b_ct_arr, b_rf_arr, b_nrf_arr, is_threat_arr, results of threat detection.
        """
        if limit is None:
            limit = np.shape(self.testing_set)[0]
        for i in tqdm(range(0, limit), desc="Running threat detection..."):
            self.run_rf(i)
            self.run_nrf(i)
            self.is_threat_arr.append(bool(self.y_test[i]))

    def run_ct(self, ctx):
        self.b_ct_arr.append(self.classifier.cryptotree(ctx))

    def run_nrf(self, i):
        self.b_nrf_arr.append(self.classifier.neural_random_forest(self.testing_set[i]))

    def run_rf(self, i: int):
        self.b_rf_arr.append(self.classifier.random_forest(self.testing_set[i]))
