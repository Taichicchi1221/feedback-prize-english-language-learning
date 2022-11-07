import os
import subprocess
import sys
import glob
import typing
import argparse
import shutil
import joblib
from itertools import product

from omegaconf import OmegaConf

import mlflow


MLFLOW_DIR = "../mlruns"


########################## source directory and experiment name ##########################
SRC_DIR = "language_model"
EXPERIMENT_NAME = "deberta-v3-base_for_ensemble"

# MODEL = ["microsoft/deberta-v3-base"]
# POOLING = ["MeanPooling", "CLSPooling"]
# BATCH_SIZE =

OVERWRITE_PARAMS = [None]


# SRC_DIR = "pretrain"
# EXPERIMENT_NAME = "pretrain"
# OVERWRITE_PARAMS = [None]

########################## language model ##########################
if SRC_DIR == "language_model":
    WORKFILE_NAME = "main.py"
    DEPENDENT_FILES = [
        "utils.py",
        "train.py",
        "dataset.py",
        "inference.py",
        "loss.py",
        "optimizer.py",
        "tokenizer.py",
        "model.py",
        "preprocess.py",
    ]
    CONFIGFILE_NAME = "language_model_config.yaml"
    WORKFILE_TO_CLEAR = ["__pycache__", "main.log", "lightning_logs", ".hydra"]
    REPORT_RESULTS = True
########################## language model ##########################


########################## pretrain ##########################
if SRC_DIR == "pretrain":
    WORKFILE_NAME = "main.py"
    DEPENDENT_FILES = ["pretrain.py"]
    CONFIGFILE_NAME = "pretrain_config.yaml"
    WORKFILE_TO_CLEAR = ["__pycache__", "main.log", ".hydra"]
    REPORT_RESULTS = False
########################## pretrain ##########################


# ====================================================
# util
# ====================================================
def flatten_dict(params: typing.Dict[typing.Any, typing.Any], delimiter: str = "/") -> typing.Dict[str, typing.Any]:
    """
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    Examples:
        >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> LightningLoggerBase._flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, typing.MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (typing.MutableMapping, argparse.Namespace)):
                    value = vars(value) if isinstance(value, argparse.Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [
                        key,
                        value if value is not None else str(None),
                    ]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


# ====================================================
# work
# ====================================================
def premain(directory):
    shutil.rmtree(directory)
    os.makedirs(directory)
    os.chdir(directory)
    sys.path.append(directory)


def copy_files_work():
    # copy src -> work
    shutil.copy(f"../src/{SRC_DIR}/{WORKFILE_NAME}", "main.py")

    for dependent_file in DEPENDENT_FILES:
        shutil.copy(f"../src/{SRC_DIR}/{dependent_file}", dependent_file)

    # copy config -> work
    shutil.copy(f"../config/{CONFIGFILE_NAME}", "config.yaml")


# ====================================================
# save results
# ====================================================
def save_results_main():
    results = joblib.load("results.pkl")

    print(results["params"])

    # identify experiment
    client = mlflow.tracking.MlflowClient(MLFLOW_DIR)
    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
    except:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id)

    # params
    for key, value in flatten_dict(results["params"]).items():
        client.log_param(run.info.run_id, key, value)

    # metric
    for key, value in results["metrics"].items():
        client.log_metric(run.info.run_id, key, value)

    # artifacts
    for filename in glob.glob("./*"):
        client.log_artifact(run.info.run_id, filename)


def remove_work_files(paths):
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


def parse_params(params):
    if params is None:
        return ""

    params_command = []
    for k, v in params.items():
        params_command.append(f"{k}='{v}'")

    return " ".join(params_command)


def main(overwrite_params: dict):
    cfg = OmegaConf.load(os.path.join("../config", CONFIGFILE_NAME))

    # prepare
    premain("/workspaces/feedback-prize-english-language-learning/work")
    copy_files_work()

    # call work.main
    print("=" * 25, "PROCESS", "=" * 25)
    overwrite_params_command = parse_params(overwrite_params)
    command = "python -u main.py" + " " + overwrite_params_command
    print("=" * 5, command)
    ret = subprocess.run(command, shell=True)
    print("=" * 25, "PROCESS", "=" * 25)

    if ret.returncode:
        return ret.returncode

    remove_work_files(WORKFILE_TO_CLEAR)

    # save results
    if cfg.globals.debug:
        return ret.returncode
    if REPORT_RESULTS:
        save_results_main()
    return ret.returncode


if __name__ == "__main__":
    for overwrite_params in OVERWRITE_PARAMS:
        returncode = main(overwrite_params)
        if returncode:
            break
