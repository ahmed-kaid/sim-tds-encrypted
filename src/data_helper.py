from typing import List

import numpy as np
import pandas as pd
import tenseal.sealapi as seal
from tqdm import tqdm

from cryptotree.preprocessing import Featurizer


def get_training_set(
    data_folder_path: str = "unsw-nb15/",
    file_path: str = "UNSW_NB15_training-set.csv",
    attack_cat: str = None,
) -> pd.DataFrame:
    """Reads a UNSW-NB15 CSV file and converts its entries into vectors.

    Args:
        data_folder_path (str, optional): Relative path of the python data folder.
        Defaults to "unsw-nb15/".
        file_path (str. optional): Path of the file, after the data folder.
        Defaults to "UNSW_NB15_training-set.csv".
        attack_cat (str, optional): Limit set to only include entries with category.

    Returns:
        pd.DataFrame: An dataframe of all entries in the CSV file.
    """
    df = pd.read_csv(data_folder_path + file_path, header=0)
    df.drop(columns=df.columns[0], axis=1, inplace=True)  # remove ID column
    if attack_cat is not None:
        df.drop(df[df.attack_cat != attack_cat].index, inplace=True)
    return df


def normalize_data(
    df: pd.DataFrame,
    limit: int = None,
    use_subset: bool = False,
) -> np.ndarray:
    """Normalizes Data, so it can be used for Neural Random Forest and Cryptotree

    Args:
        df (pd.DataFrame): Data to be normalized

    Returns:
        np.ndarray: Normalized data
    """
    categorical_columns = [
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
    ]
    df.columns = df.columns = [
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
        "attack_cat",
        "label",
    ]
    if use_subset:  # Get a random subset of elements
        if limit > df.shape[0]:
            limit = df.shape[0]
        df = df.sample(limit)
    X = df[categorical_columns]
    y = df["attack_cat"].apply(lambda x: 0 if x == "Normal" else 1)
    pipe = Featurizer(categorical_columns)
    X, y = pipe.fit_transform(X), y
    return X, y


def value_to_float(
    value_type: str,
    value: float | str | pd.core.series.Series,
    step_count: float = 0.2,
    reverse=False,
) -> float:
    """Converts the string-values of UNSW-NB15 data parameters to float.

    Args:
        value_type (str): Possible values: "proto", "service", "state", "attack_cat"
        value (str or float or pd.core.series.Series): The value(s) with the associated value type
        step_count (float, optional): While encoding,
        the float will be a multiple of the step_count.
        Defaults to 0.2.
        reverse (bool, optional): If the function should take in a float as value and
        return the corresponding string

    Returns:
        float: The converted/encoded value as a float.
    """
    if type(value) == pd.core.series.Series:
        if value_type in [
            "proto",
            "service",
            "state",
            "attack_cat",
        ]:
            i = 0
            progress = tqdm(total=len(value), desc=f"Encoding {value_type} values...")
            while i < len(value):
                value[i] = value_to_float(value_type, value[i])
                i += 1
                progress.update(1)
            return value
        else:
            return value

    match value_type:
        case "proto":
            possible_values = [
                "ipcomp",
                "icmp",
                "rtp",
                "trunk-1",
                "xns-idp",
                "bbn-rcc",
                "sdrp",
                "cftp",
                "xnet",
                "iplt",
                "ipcv",
                "chaos",
                "ptp",
                "sctp",
                "stp",
                "ipnip",
                "br-sat-mon",
                "nsfnet-igp",
                "ggp",
                "sep",
                "larp",
                "mfe-nsp",
                "argus",
                "ipip",
                "sccopmce",
                "crudp",
                "iatp",
                "vrrp",
                "ospf",
                "secure-vmtp",
                "ippc",
                "st2",
                "pnni",
                "pim",
                "ib",
                "bna",
                "ttp",
                "pri-enc",
                "micp",
                "vmtp",
                "a/n",
                "iso-tp4",
                "idpr-cmtp",
                "sat-mon",
                "vines",
                "nvp",
                "rsvp",
                "sat-expak",
                "rdp",
                "aris",
                "ipv6-route",
                "xtp",
                "narp",
                "ifmp",
                "merit-inp",
                "i-nlsp",
                "ipx-n-ip",
                "sps",
                "pipe",
                "idrp",
                "pvp",
                "dgp",
                "igmp",
                "eigrp",
                "pup",
                "uti",
                "l2tp",
                "irtp",
                "wb-expak",
                "egp",
                "ipv6-frag",
                "any",
                "smp",
                "isis",
                "ddx",
                "cpnx",
                "mtp",
                "scps",
                "fc",
                "sprite-rpc",
                "ipv6",
                "skip",
                "3pc",
                "wsn",
                "ip",
                "compaq-peer",
                "dcn",
                "wb-mon",
                "qnx",
                "netblt",
                "igp",
                "sm",
                "cphb",
                "etherip",
                "pgm",
                "tcf",
                "mobile",
                "swipe",
                "udp",
                "kryptolan",
                "ax.25",
                "arp",
                "visa",
                "ddp",
                "unas",
                "idpr",
                "il",
                "leaf-1",
                "hmp",
                "trunk-2",
                "mux",
                "leaf-2",
                "prm",
                "ipv6-opts",
                "zero",
                "aes-sp3-d",
                "encap",
                "gmtp",
                "gre",
                "crtp",
                "tlsp",
                "srp",
                "sun-nd",
                "ipv6-no",
                "iso-ip",
                "mhrp",
                "snp",
                "tcp",
                "tp++",
                "rvd",
                "fire",
                "emcon",
                "cbt",
            ]
        case "service":
            possible_values = [
                "-",
                "ftp-data",
                "pop3",
                "dns",
                "radius",
                "irc",
                "ftp",
                "dhcp",
                "http",
                "ssl",
                "snmp",
                "smtp",
                "ssh",
            ]
        case "state":
            possible_values = [
                "INT",
                "RST",
                "FIN",
                "CON",
                "CLO",
                "ACC",
                "REQ",
                "ECO",
                "PAR",
                "URN",
                "no",
            ]
        case "attack_cat":
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
        case _:
            return value
    floats = np.arange(0, (len(possible_values) * step_count), step_count).tolist()
    if not reverse:
        return floats[possible_values.index(value)]
    else:
        return possible_values[floats.index(value)]


def get_testing_set(attack_cat: str = None) -> pd.DataFrame:
    """Obtain UNSW-NB15 testing set.
    Args:
        attack_cat (str, optional): Limit set to only include entries with category.

    Returns:
        DataFrame: Testing set
    """
    return get_training_set(
        file_path="UNSW_NB15_testing-set.csv", attack_cat=attack_cat
    )


def score(curr_score: list, classification: bool, result: bool) -> list:
    """Checks whether a false/true positive/negative occured,
    and adds scores to the tally

    Args:
        score (list): The current score. Format: [true positives, false positives,
        true negatives, false negatives]
        classification (bool): The result from a classifier (RF or CryptoTree (CT)).
        result (bool): The true result, if the vector was an actual threat or not.

    Returns:
        list: New score.
    """
    new_score = curr_score
    if classification and result:  # True positive
        new_score[0] += 1
    elif classification and not result:  # False positive
        new_score[1] += 1
    elif not result:  # True negative
        new_score[2] += 1
    else:  # False negative
        new_score[3] += 1
    return new_score


def score_normalized(curr_score: list, classification: list, result: bool) -> list:
    """Checks whether a false/true positive/negative occured,
    and adds scores to the tally (for normalized values)

    Args:
        curr_score (list): _description_
        classification (float): _description_
        result (bool): _description_

    Returns:
        list: _description_
    """
    # TODO: Understand classification and evaluate results
    return score(curr_score, classification[1] >= classification[0], result)


def scores_to_table(ct: list, rf: list, nrf: list, total: int, acc: str = ".2") -> list:
    """Takes the scores from the classification and puts them into two tables, to be
    printed in main()

    Args:
        ct (list): Cryptotree (encrypted) scores
        rf (list): Random Forest (unencrypted) scores
        total (int): Total amount of tests in the testing set
        acc (str, optional): How accurate the percentages should be. Defaults to ".2".

    Returns:
        list: Two tables, one for each score
    """
    ct_table = (
        ["True Positives", "False Positives", "True Negatives", "False Negatives"],
        [
            format(ct[0] / total, acc + "%"),
            format(ct[1] / total, acc + "%"),
            format(ct[2] / total, acc + "%"),
            format(ct[3] / total, acc + "%"),
        ],
    )

    rf_table = (
        ["True Positives", "False Positives", "True Negatives", "False Negatives"],
        [
            format(rf[0] / total, acc + "%"),
            format(rf[1] / total, acc + "%"),
            format(rf[2] / total, acc + "%"),
            format(rf[3] / total, acc + "%"),
        ],
    )
    nrf_table = (
        ["True Positives", "False Positives", "True Negatives", "False Negatives"],
        [
            format(nrf[0] / total, acc + "%"),
            format(nrf[1] / total, acc + "%"),
            format(nrf[2] / total, acc + "%"),
            format(nrf[3] / total, acc + "%"),
        ],
    )
    return [rf_table, nrf_table, ct_table]


def create_seal_globals(
    globals: dict,
    poly_modulus_degree: int,
    moduli: List[int],
    PRECISION_BITS: int,
    use_local=True,
    use_symmetric_key=False,
):
    """Creates SEAL context variables and populates the globals with it.
    Modified version from Cryptotree package:
    https://github.com/dhuynh95/cryptotree/blob/master/cryptotree/seal_helper.py#L71"""
    parms = seal.EncryptionParameters(seal.SCHEME_TYPE.CKKS)
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(seal.CoeffModulus.Create(poly_modulus_degree, moduli))

    context = seal.SEALContext(parms, True, seal.SEC_LEVEL_TYPE.TC128)

    keygen = seal.KeyGenerator(context)

    globals["parms"] = parms
    globals["context"] = context
    globals["scale"] = pow(2.0, PRECISION_BITS)

    globals["public_key"] = seal.PublicKey()
    keygen.create_public_key(globals["public_key"])
    globals["secret_key"] = keygen.secret_key()

    if use_local:
        globals["relin_keys"] = seal.RelinKeys()
        keygen.create_relin_keys(globals["relin_keys"])
        globals["galois_keys"] = seal.GaloisKeys()
        keygen.create_galois_keys(globals["galois_keys"])
    else:
        globals["relin_keys"] = keygen.relin_keys()
        globals["galois_keys"] = keygen.galois_keys()

    if use_symmetric_key:
        globals["encryptor"] = seal.Encryptor(context, globals["secret_key"])
    else:
        globals["encryptor"] = seal.Encryptor(context, globals["public_key"])

    globals["evaluator"] = seal.Evaluator(context)
    globals["decryptor"] = seal.Decryptor(context, globals["secret_key"])
    globals["encoder"] = seal.CKKSEncoder(context)
