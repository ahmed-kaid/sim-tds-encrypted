import sys
from typing import Union

import numpy as np
import tenseal.sealapi as seal
from tabulate import tabulate
from tqdm import tqdm

from cryptotree.cryptotree import HomomorphicTreeFeaturizer
from src import data_helper, server


def run_encrypted_tests(
    X_test: np.ndarray,
    homomorphic_featurizer: HomomorphicTreeFeaturizer,
    limit: int,
    s: server.Server,
) -> list:
    """Encrypts a set of testing data, to be used by Cryptotree.

    Args:
        X_test (np.ndarray): The testing set
        homomorphic_featurizer (HomomorphicTreeFeaturizer): Will encrypt entries
        limit (int): Limit how many entries of the testing set should be encrypted.

    Returns:
        np.ndarray: Encrypted training set
    """
    b_ct_arr = []
    if limit is None or limit > np.shape(X_test)[0]:
        limit = np.shape(X_test)[0]
    for i in tqdm(
        range(0, limit), desc="Running encrypted threat detection...", file=sys.stdout
    ):
        pred = encrypt_and_classify(X_test, homomorphic_featurizer, s, i)
        b_ct_arr.append(pred)
    return b_ct_arr


def encrypt_and_classify(X_test, homomorphic_featurizer, s, i):
    time_functions = "--time_functions" in sys.argv
    if time_functions:
        import time

        encr_start = time.time()
    ctx = homomorphic_featurizer.encrypt(X_test[i])
    if time_functions:
        encr_end = time.time()
        pred_start = time.time()
    pred = s.run_ct(ctx)
    if time_functions:
        pred_end = time.time()
        print("Time taken for encryption:", encr_end - encr_start, "seconds")
        print("Time taken for classification:", pred_end - pred_start, "seconds")
    return pred


def evaluate_results(
    b_ct_arr: list, b_rf_arr: list, b_nrf_arr: list, is_threat_arr: list
) -> tuple:
    """Evaluates the results of three classifiers based on the actual testing data.

    Args:
        b_ct_arr (list): Results from Cryptotree.
        b_rf_arr (list): Results from Random Forest.
        b_nrf_arr (list): Results from Neural Random Forest.
        is_threat_arr (list): Acutal testing data.

    Returns:
        tuple: Returns a pretty table, as well as all three scores, containg TPs, FPs, TNs and FNs.
    """
    if not (len(b_ct_arr) == len(b_rf_arr) == len(b_nrf_arr) == len(is_threat_arr)):
        print("Error: Lists are not same length")
        print(len(b_ct_arr))
        print(len(b_rf_arr))
        print(len(b_nrf_arr))
        print(len(is_threat_arr))
        return
    b_ct_decrypted_arr = []
    for ctxt in b_ct_arr:
        ptx = seal.Plaintext()
        decryptor.decrypt(ctxt, ptx)  # noqa: F821 # type: ignore
        # [:2], because these are the scores of the classes
        b_ct_decrypted_arr.append(encoder.decode_double(ptx)[:2])  # noqa: F821 # type: ignore
    ct_score = [0, 0, 0, 0]
    rf_score = [0, 0, 0, 0]
    nrf_score = [0, 0, 0, 0]
    for i in tqdm(
        range(0, len(is_threat_arr)), desc="Scoring results...", file=sys.stdout
    ):
        rf_score = data_helper.score(rf_score, b_rf_arr[i], is_threat_arr[i])
        nrf_score = data_helper.score_normalized(
            nrf_score, b_nrf_arr[i][0], is_threat_arr[i]
        )
        ct_score = data_helper.score_normalized(
            ct_score, b_ct_decrypted_arr[i], is_threat_arr[i]
        )
    return (
        data_helper.scores_to_table(ct_score, rf_score, nrf_score, len(is_threat_arr)),
        ct_score,
        rf_score,
        nrf_score,
    )


def print_metrics(score: list) -> None:
    """Print out more metrics using the number of results.

    Args:
        score (list): Results from classifaction
    """
    tp = score[0]
    fp = score[1]
    tn = score[2]
    fn = score[3]
    acc = (tp + tn) / (tp + fn + tn + fp)
    err = (fp + fn) / (tp + fn + tn + fp)
    precision = (tp) / (tp + fp) if (tp + fp) > 0 else 0
    recall = (tp) / (tp + fn) if (tp + fn) > 0 else 0
    f_1 = (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    beta = 0.5
    f_beta_1 = (
        ((beta**2 + 1) * precision * recall) / ((beta**2) * precision + recall)
        if ((beta**2) * precision + recall) > 0
        else 0
    )
    beta = 2
    f_beta_2 = (
        ((beta**2 + 1) * precision * recall) / ((beta**2) * precision + recall)
        if ((beta**2) * precision + recall) > 0
        else 0
    )

    print("Accuracy    " + str(acc))
    print("Error       " + str(err))
    print("Precision   " + str(precision))
    print("Recall      " + str(recall))
    print("F_1-Score   " + str(f_1))
    print("F_0.5-Score " + str(f_beta_1))
    print("F_2-Score   " + str(f_beta_2) + "\n")


def cli_helper(args: list) -> Union[int | None, str | None]:
    """Parses through CLI arguments and sets limit and attack_cat variables

    Args:
        args (list): CLI arguments

    Raises:
        ValueError: If wrong arguments are given

    Returns:
        Union[int | None, str | None]: limit, attack_cat
    """
    possible_values = [
        "Analysis",
        "Exploits",
        "Normal",
        "DoS",
        "Reconnaissance",
        "Fuzzers",
        "Backdoor",
        "Generic",
        "Shellcode",
        "Worms",
    ]
    help_text = (
        "Usage: python client.py [options...]\n"
        "     --limit <int | None>\tLimits how many entries should be processed. Defaults to 200\n"
        "     --attack_cat <str | None>\tFilter by specific attack category\n"
        "\nPossible attack categories:\n"
        "     " + ", ".join(possible_values)
    )
    if "--help" in args:
        print(help_text)
        sys.exit(0)
    limit = 200  # default value
    attack_cat = None
    try:
        if "--limit" in args:
            limit = args[args.index("--limit") + 1]
            if limit == "None":
                limit = None
            elif limit.isnumeric():
                limit = int(limit)
            else:
                raise ValueError("Limit is neither a number nor 'None'.")
        if "--attack_cat" in args:
            attack_cat = args[args.index("--attack_cat") + 1]
            if attack_cat == "None":
                attack_cat = None
            elif attack_cat not in possible_values:
                raise ValueError(
                    f"Attack Category {attack_cat} not in list of possible values."
                )
    except ValueError as e:
        print(e)
        print(help_text)
        sys.exit(1)
    return limit, attack_cat


def main(limit: int = None, attack_cat: str = None) -> tuple:
    """Asks the server to start the TDS

    Args:
        limit (int, optional): Limit how many tests should be classified. Defaults to None.

    Returns:
        tuple: Returns a pretty table, as well as all three scores, containg TPs, FPs, TNs and FNs.
    """
    s = server.Server(limit, attack_cat)
    b_rf_arr, b_nrf_arr, is_threat_arr = s.run_tds(limit)
    b_ct_arr = run_encrypted_tests(
        s.get_testing_set(), s.get_homomorphic_featurizer(), limit, s
    )
    return evaluate_results(b_ct_arr, b_rf_arr, b_nrf_arr, is_threat_arr)


if __name__ == "__main__":
    limit, attack_cat = cli_helper(sys.argv)
    tables, ct_score, rf_score, nrf_score = main(limit, attack_cat)
    if attack_cat is not None:
        print(f"Filtered results by attack category: {attack_cat}")
    print("\n\n" + "-" * 72)
    print("\nRandom Forest (unencrypted performance)\n")
    print(tabulate(tables[0], headers="firstrow"))
    print_metrics(rf_score)
    print("-" * 72)
    print("\nNeural Random Forest (unencrypted performance)\n")
    print(tabulate(tables[1], headers="firstrow"))
    print_metrics(nrf_score)
    print("-" * 72)
    print("Cryptotree (encrypted performance)\n")
    print(tabulate(tables[2], headers="firstrow"))
    print_metrics(ct_score)
    print("-" * 72)
