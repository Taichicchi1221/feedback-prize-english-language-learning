from genericpath import isdir
import os
import subprocess
import sys
import glob
import typing
import argparse
import shutil
import joblib

from omegaconf import OmegaConf

import mlflow


MLFLOW_DIR = "../mlruns"

########################## language model ##########################
SRC_DIR = "language_model"
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
]
CONFIGFILE_NAME = "language_model_config.yaml"
EXPERIMENT_NAME = "baseline_tuning"
WORKFILE_TO_CLEAR = ["__pycache__", "main.log", "lightning_logs", ".hydra"]
REPORT_RESULTS = True
########################## language model ##########################


########################## pretrain ##########################
# SRC_DIR = "pretrain"
# WORKFILE_NAME = "main.py"
# DEPENDENT_FILES = ["pretrain.py"]
# CONFIGFILE_NAME = "pretrain_config.yaml"
# EXPERIMENT_NAME = "pretrain"
# WORKFILE_TO_CLEAR = ["__pycache__", "main.log", ".hydra"]
# REPORT_RESULTS = False
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


def copy_files_exp():
    # copy src -> exp
    run_paths = glob.glob(os.path.join(f"../src/experiments/{EXPERIMENT_NAME}", "run*"))
    run_number = len(run_paths)
    dir_ = f"../src/experiments/{EXPERIMENT_NAME}/run{str(run_number).zfill(3)}"

    os.makedirs(dir_, exist_ok=False)
    shutil.copy(WORKFILE_NAME, f"{dir_}/work.py")

    for dependent_file in DEPENDENT_FILES:
        shutil.copy(dependent_file, f"{dir_}/{dependent_file}")

    # copy config -> exp
    shutil.copy("config.yaml", f"{dir_}/config.yaml")


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


def main():
    cfg = OmegaConf.load(os.path.join("../config", CONFIGFILE_NAME))

    # prepare
    premain("/workspaces/feedback-prize-english-language-learning/work")
    copy_files_work()

    # call work.main
    print("=" * 25, "PROCESS", "=" * 25)
    command = "python -u main.py"
    ret = subprocess.run(command, shell=True)
    print("=" * 25, "PROCESS", "=" * 25)

    if ret.returncode:
        return

    remove_work_files(WORKFILE_TO_CLEAR)

    # save results
    if cfg.globals.debug:
        return
    if REPORT_RESULTS:
        copy_files_exp()
        save_results_main()


if __name__ == "__main__":
    main()
