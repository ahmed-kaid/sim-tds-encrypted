import numpy as np
import tenseal.sealapi as seal
from tabulate import tabulate
from tqdm import tqdm

from cryptotree.cryptotree import HomomorphicTreeFeaturizer
from src import data_helper, server


def encrypt_set(
    X_test: np.ndarray,
    homomorphic_featurizer: HomomorphicTreeFeaturizer,
    limit: int,
) -> np.ndarray:
    """Encrypts a set of testing data, to be used by Cryptotree.

    Args:
        X_test (np.ndarray): The testing set
        homomorphic_featurizer (HomomorphicTreeFeaturizer): Will encrypt entries
        limit (int): Limit how many entries of the testing set should be encrypted.

    Returns:
        np.ndarray: Encrypted training set
    """
    encrypted_set = []
    i = 0
    if limit is None:
        limit = np.shape(X_test)[0]
    progress = tqdm(range(0, limit), desc="Encrypting entries...")
    for entry in X_test:
        ctx = homomorphic_featurizer.encrypt(entry)
        encrypted_set.append(ctx)
        progress.update(1)
        i += 1
        if i > limit:
            break
    progress.close()
    return encrypted_set


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
        b_ct_decrypted_arr.append(encoder.decode_double(ptx)[:2])  # noqa: F821 # type: ignore
    ct_score = [0, 0, 0, 0]
    rf_score = [0, 0, 0, 0]
    nrf_score = [0, 0, 0, 0]
    for i in tqdm(range(0, len(is_threat_arr)), desc="Scoring results..."):
        rf_score = data_helper.score(rf_score, b_rf_arr[i], is_threat_arr[i])
        nrf_score = data_helper.score_normalized(
            nrf_score, b_nrf_arr[i], is_threat_arr[i]
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


def main(limit: int = None) -> tuple:
    """Asks the server to start the TDS

    Args:
        limit (int, optional): Limit how many tests should be classified. Defaults to None.

    Returns:
        tuple: Returns a pretty table, as well as all three scores, containg TPs, FPs, TNs and FNs.
    """
    b_ct_arr, b_rf_arr, b_nrf_arr, is_threat_arr = server.run_tds(limit)
    return evaluate_results(b_ct_arr, b_rf_arr, b_nrf_arr, is_threat_arr)


if __name__ == "__main__":
    tables, ct_score, rf_score, nrf_score = main(limit=20)
    print("\n\n" + "-" * 72)
    print("\nRandom Forest (unencrypted performance)\n")
    print(tabulate(tables[0], headers="firstrow"))
    print("-" * 72)
    print("\nNeural Random Forest (unencrypted performance)\n")
    print(tabulate(tables[1], headers="firstrow"))
    print("-" * 72)
    print("Cryptotree (encrypted performance)\n")
    print(tabulate(tables[2], headers="firstrow"))
    print("-" * 72)
