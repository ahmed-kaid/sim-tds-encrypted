from tqdm import tqdm

from . import classifiers, data_helper


class Server:
    def __init__(self, limit: int = None) -> None:
        use_subset = limit is not None
        self.training_set = data_helper.get_training_set()
        self.testing_set, self.y_test = data_helper.normalize_data(
            data_helper.get_testing_set(), limit=limit, use_subset=use_subset
        )
        self.b_ct_arr = []
        self.b_rf_arr = []
        self.b_nrf_arr = []
        self.is_threat_arr = []
        self.classifier = classifiers.Classifier(self.training_set)

    def get_testing_set(self):
        return self.testing_set

    def get_homomorphic_featurizer(self):
        return self.classifier.get_hf()

    def get_b_ct_arr(self):
        return self.b_ct_arr

    def run_tds(self, limit: int = None) -> None:
        """
        Starts the simulated Threat Detection System

        Args:
            limit (int, optional): Limit how many tests should be classified. Defaults to None.

        Returns:
            tuple: b_ct_arr, b_rf_arr, b_nrf_arr, is_threat_arr, results of threat detection.
        """
        # self.y_test = self.y_test.to_numpy()
        for i in tqdm(range(0, limit), desc="Running threat detection..."):
            self.run_rf(i)
            self.run_nrf(i)
            self.is_threat_arr.append(bool(self.y_test[i]))
        return self.b_rf_arr, self.b_nrf_arr, self.is_threat_arr

    def run_ct(self, ctx):
        return self.classifier.cryptotree(ctx)

    def run_nrf(self, i):
        self.b_nrf_arr.append(self.classifier.neural_random_forest(self.testing_set[i]))

    def run_rf(self, i: int):
        self.b_rf_arr.append(self.classifier.random_forest(self.testing_set[i]))
