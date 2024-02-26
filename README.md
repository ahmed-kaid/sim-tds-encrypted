# Implementation and Simulation of a Threat Detection System (TDS) for Encrypted Data

> Using the [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset), [CKKS Encryption scheme](https://eprint.iacr.org/2016/421.pdf), [TenSEAL](https://github.com/OpenMined/TenSEAL) and [Cryptotree](https://github.com/dhuynh95/cryptotree/blob/master/README.md?plain=1)

## Installation

#### 1. Install Cryptotree

```
git clone https://github.com/dhuynh95/cryptotree.git
```

- Before installing:
  1. Change line 24 of `cryptotree/settings.ini` to `requirements = torch nbdev scikit-learn matplotlib fastcore fastai` because `sklearn` is deprecated.
  2. Change line 49 in `cryptotree/cryptotree/activations.py` to
     `d = tree.n_features_in_`.
- Follow https://github.com/dhuynh95/cryptotree#install

#### 2. Clone this project

```
git clone https://github.com/ahmed-kaid/sim-tds-encrypted.git
cd sim-tds-encrypted
```

#### 3. Install required packages

```
pip install -r requirements.txt
```

## Usage

To run the project, run the `client.py`.
On line 88 of `client.py`, you can adjust, how many entries of the testing set should be classified. This is done, by modifying the `limit` paramter, which defaults to `20` and can be set to `None`.
