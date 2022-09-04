import os
import subprocess
import sys
import glob
import typing
import argparse
import shutil
import joblib


import mlflow


MLFLOW_DIR = "../mlruns"

########################## work ##########################
WORKFILE_NAME = "work.py"
DEPENDENT_FILES = ["utils.py"]
EXPERIMENT_NAME = "test"
WORKFILE_TO_CLEAR = []
########################## work ##########################


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


def copy_files():
    # copy src -> exp
    run_paths = glob.glob(os.path.join(f"../src/{EXPERIMENT_NAME}", "run*"))
    run_number = len(run_paths)
    dir_ = f"../src/{EXPERIMENT_NAME}/run{run_number}"

    os.makedirs(dir_, exist_ok=False)
    shutil.copy(f"../src/{WORKFILE_NAME}", f"{dir_}/work.py")

    for dependent_file in DEPENDENT_FILES:
        shutil.copy(f"../src/{dependent_file}", f"{dir_}/{dependent_file}")

    # copy src -> work
    shutil.copy(f"../src/{WORKFILE_NAME}", "work.py")

    for dependent_file in DEPENDENT_FILES:
        shutil.copy(f"../src/{dependent_file}", dependent_file)


# ====================================================
# save results
# ====================================================
def save_results_main():
    results = joblib.load("results.pkl")

    print(results.params)

    # identify experiment
    client = mlflow.tracking.MlflowClient(MLFLOW_DIR)
    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
    except:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id)

    # params
    for key, value in flatten_dict(results.params).items():
        client.log_param(run.info.run_id, key, value)

    # metric
    for key, value in results.metrics.items():
        client.log_metric(run.info.run_id, key, value)

    # artifacts
    for filename in glob.glob("./*"):
        client.log_artifact(run.info.run_id, filename)

    return results


def remove_work_files(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":

    # prepare
    premain("/workspaces/feedback-prize-english-language-learning/work")
    copy_files()

    # call work.main
    print("=" * 25, "PROCESS", "=" * 25)
    subprocess.run("python -u work.py", shell=True)
    print("=" * 25, "PROCESS", "=" * 25)

    remove_work_files(WORKFILE_TO_CLEAR)

    # save results
    save_results_main()
