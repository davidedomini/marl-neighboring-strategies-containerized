import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from hashlib import sha512
from itertools import product


def get_hyperparameters():
    hyperparams = os.environ[HYPERPARAMETERS_NAME]
    print(hyperparams)
    hyperparams = yaml.safe_load(hyperparams)
    experiment_name, hyperparams = list(hyperparams.items())[0]
    return experiment_name, hyperparams


def get_neighboring_strategy(module_name, agents, neighbors, visible_items):
    import importlib
    class_name = module_name.rsplit('.')[-1]
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    return klass(agents, neighbors, visible_items)


def check_experiment(id, data_directory):
    for file in data_directory.glob('**/*.yaml'):
        if file.is_file() and file.stem == id:
            print(f'Experiment with ID {id} already exists in {file.parent}', file=sys.stderr)
            exit(0)


HYPERPARAMETERS_NAME = 'LEARNING_HYPERPARAMETERS'
experiment_name, hyperparameters = get_hyperparameters()

SEED           = set(range(hyperparameters['seed']['min'], hyperparameters['seed']['max'], hyperparameters['seed']['step']))
AGENTS         = set(hyperparameters['agents'])
NEIGHBORS      = set(hyperparameters['neighbors'])
VISIBLE_ITEMS  = set(hyperparameters['visible_items'])
FOOTPRINT_KEYS = {'experiment_name', 'seed', 'agents', 'neighbors', 'visible_items'}
DATA_DIR       = Path(os.getenv('DATA_DIR', './data'))
OUTPUT_DIR     = Path(os.getenv('OUTPUT_DIR', str(DATA_DIR)))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_experiments = list(product(AGENTS, NEIGHBORS, VISIBLE_ITEMS))

for agents, neighbors, visible_items in all_experiments:
    for seed in SEED:
        EXPERIMENT_FOOTPRINT = {k: v for k, v in locals().items() if k in FOOTPRINT_KEYS}
        EXPERIMENT_FOOTPRINT_YAML = yaml.dump(EXPERIMENT_FOOTPRINT, sort_keys=True)
        EXPERIMENT_ID = sha512(EXPERIMENT_FOOTPRINT_YAML.encode()).hexdigest()
        check_experiment(EXPERIMENT_ID, DATA_DIR)
        trainer = get_neighboring_strategy(f'training.{experiment_name}', agents, neighbors, visible_items)
        trainer.train()
