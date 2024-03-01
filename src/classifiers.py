import builtins
import pathlib
import subprocess
import sys
import time

import numpy as np
import tenseal.sealapi as seal
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from tqdm import tqdm

from cryptotree.cryptotree import (
    HomomorphicNeuralRandomForest,
    HomomorphicTreeEvaluator,
    HomomorphicTreeFeaturizer,
)
from cryptotree.polynomials import polyeval_tree
from cryptotree.seal_helper import append_globals_to_builtins
from cryptotree.tree import NeuralRandomForest, TanhTreeMaker

from . import data_helper


class Classifier:
    def __init__(self, training_set, dilatation_factor=16, polynomial_degree=16):
        self.__train_vecs, self.__target_vals = data_helper.normalize_data(training_set)
        self.__rf = self.__train_rf()
        self.__model = NeuralRandomForest(
            self.__rf.estimators_,
            tree_maker=TanhTreeMaker(
                use_polynomial=True,
                dilatation_factor=dilatation_factor,
                polynomial_degree=polynomial_degree,
            ),
        )
        self.__tree_evaluator, self.__homomorphic_featurizer = self.__train_h_rf()

    def __train_rf(self) -> RandomForestClassifier:
        """Trains a random forest

        Returns:
            RandomForestClassifier: Trained RF classifier
        """
        rf = RandomForestClassifier(max_depth=4, random_state=0).fit(
            self.__train_vecs, self.__target_vals
        )
        if "--export-tree-imgs" in sys.argv:
            # This requires graphviz installed
            # https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
            t = time.localtime()
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)
            path = "results/trees/" + current_time
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            for i in tqdm(
                range(0, len(rf.estimators_)),
                desc=f"Adding images of trees to {path}...",
                file=sys.stdout,
            ):
                export_graphviz(
                    rf.estimators_[i],
                    out_file=path + f"/{'%02d' % i}.dot",
                    feature_names=[
                        "dur",
                        "proto",
                        "service",
                        "state",
                        "spkts",
                        "dpkts",
                        "sbytes",
                        "dbytes",
                        "rate",
                        "sttl",
                        "dttl",
                        "sload",
                        "dload",
                        "sloss",
                        "dloss",
                        "sinpkt",
                        "dinpkt",
                        "sjit",
                        "djit",
                        "swin",
                        "stcpb",
                        "dtcpb",
                        "dwin",
                        "tcprtt",
                        "synack",
                        "ackdat",
                        "smean",
                        "dmean",
                        "trans_depth",
                        "response_body_len",
                        "ct_srv_src",
                        "ct_state_ttl",
                        "ct_dst_ltm",
                        "ct_src_dport_ltm",
                        "ct_dst_sport_ltm",
                        "ct_dst_src_ltm",
                        "is_ftp_login",
                        "ct_ftp_cmd",
                        "ct_flw_http_mthd",
                        "ct_src_ltm",
                        "ct_srv_dst",
                        "is_sm_ips_ports",
                        "label",
                    ],
                    class_names=["No Threat", "Threat"],
                    rounded=True,
                    proportion=False,
                    precision=2,
                    filled=True,
                )
                subprocess.call(
                    [
                        "dot",
                        "-Tpng",
                        path + f"/{'%02d' % i}.dot",
                        "-o",
                        path + f"/{'%02d' % i}.png",
                        "-Gdpi=600",
                    ]
                )
        return rf

    def __train_h_rf(self) -> tuple:
        """Trains a homomorphic Random Forest (HRF)

        Returns:
            tuple: tree_evaluator, homomorphic_featurizer
        """
        dilatation_factor = 16
        polynomial_degree = dilatation_factor

        dilatation_factor = 16
        degree = dilatation_factor

        model = self.__model
        model.freeze_layer("comparator")
        model.freeze_layer("matcher")

        PRECISION_BITS = 28
        UPPER_BITS = 9

        polynomial_multiplications = int(np.ceil(np.log2(degree))) + 1
        n_polynomials = 2
        matrix_multiplications = 3

        depth = matrix_multiplications + polynomial_multiplications * n_polynomials

        poly_modulus_degree = 16384

        moduli = (
            [PRECISION_BITS + UPPER_BITS]
            + (depth) * [PRECISION_BITS]
            + [PRECISION_BITS + UPPER_BITS]
        )
        data_helper.create_seal_globals(
            globals(),
            poly_modulus_degree,
            moduli,
            PRECISION_BITS,
            use_symmetric_key=False,
        )
        append_globals_to_builtins(globals(), builtins)
        h_rf = HomomorphicNeuralRandomForest(model)
        tree_evaluator = HomomorphicTreeEvaluator.from_model(
            h_rf,
            TanhTreeMaker(
                use_polynomial=True,
                dilatation_factor=dilatation_factor,
                polynomial_degree=polynomial_degree,
            ).coeffs,
            polyeval_tree,
            evaluator,  # noqa: F821 # type: ignore
            encoder,  # noqa: F821 # type: ignore
            relin_keys,  # noqa: F821 # type: ignore
            galois_keys,  # noqa: F821 # type: ignore
            scale,  # noqa: F821 # type: ignore
        )

        homomorphic_featurizer = HomomorphicTreeFeaturizer(
            h_rf.return_comparator(),
            encoder,  # noqa: F821 # type: ignore
            encryptor,  # noqa: F821 # type: ignore
            scale,  # noqa: F821 # type: ignore
        )
        return tree_evaluator, homomorphic_featurizer

    def get_hf(self):
        return self.__homomorphic_featurizer

    def random_forest(self, vec: np.ndarray) -> bool:
        """Applies Random Forest to a data point

        Args:
            vec (np.ndarray): Training set entry to be classified

        Returns:
            bool: True, if threat. False, if no threat.
        """
        return bool(self.__rf.predict([vec]))

    def neural_random_forest(self, vec) -> torch.Tensor:
        """Applies Random Forest to a data point

        Args:
            vec (np.ndarray): Training set entry to be classified

        Returns:
            torch.Tensor: Result of classification
        """
        return self.__model(torch.tensor(vec).float().unsqueeze(0))

    def cryptotree(self, ctx: seal.Ciphertext) -> bool:
        """Applies Random Forest to a data point

        Args:
            ctx (seal.Ciphertext): Encrypted training set entry to be classified

        Returns:
            Ciphertext: Encrypted result of classification
        """
        return self.__tree_evaluator(ctx)
