# Implementation and Simulation of a Threat Detection System (TDS) for Encrypted Data

> Using the [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset), [CKKS Encryption scheme](https://eprint.iacr.org/2016/421.pdf), [TenSEAL](https://github.com/OpenMined/TenSEAL) and [Cryptotree](https://github.com/dhuynh95/cryptotree/)

## Installation

#### Optionally create [virutal environment](https://docs.python.org/3/library/venv.html)

```
python -m venv .venv
source .venv/bin/activate
```

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

#### 4. Download data set

- Download the [UNSW-NB15 dataset](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15?utm_medium=social&utm_campaign=kaggle-dataset-share&utm_source=twitter) and put the CSV files into a folder called `unsw-nb15` at the root of the project.

## Usage

To run the project, run the `client.py`.
On line 102 of `client.py`, you can adjust, how many entries should be processed. This is done, by modifying the `limit` parameter, which defaults to `200` and can be set to `None`. This will pick a random subset of the training set, that will then be classified.
