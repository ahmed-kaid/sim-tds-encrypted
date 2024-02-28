# Implementation and Simulation of a Threat Detection System (TDS) for Encrypted Data

> Using the [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset), [CKKS Encryption scheme](https://eprint.iacr.org/2016/421.pdf), [TenSEAL](https://github.com/OpenMined/TenSEAL) and [Cryptotree](https://github.com/dhuynh95/cryptotree/)

## Installation

#### Optionally create [virtual environment](https://docs.python.org/3/library/venv.html)

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

## Results

The results of running this script multiple times with different parameters can be found in the `results/` folder.

## Usage

To run the project for yourself, run the `client.py`.

```
python client.py
```

You can specify how many testing set entries should be processed, by using the `--limit` parameter. The default value is 200. This will pick a random subset of the training set, that will then be classified.

```
python client.py --limit <number | None>
```

You can also specify, if you want to take a look at a specific attack category. The default value is None.

```
python client.py --attack_cat <str | None>
```

Attack categories can be found via the help command.

```
python client.py --help
```
