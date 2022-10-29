import os
import glob
import subprocess
import shutil

UPLOAD = False

DATASET_DIR = "../datasets"
DATASET_NAME = "feedback3-deberta-v3-xsmall"

EXPERIMENT_ID = "2"
RUN_ID = "e837a0acb01e433baa4a9dce0e32b001"


MLFLOW_DIR = f"../mlruns/{EXPERIMENT_ID}"


if os.path.exists(os.path.join(DATASET_DIR, DATASET_NAME)):
    shutil.rmtree(os.path.join(DATASET_DIR, DATASET_NAME))

# copy files
print("#" * 30, RUN_ID, "#" * 30)

shutil.copytree(
    os.path.join(MLFLOW_DIR, RUN_ID, "artifacts"),
    os.path.join(DATASET_DIR, DATASET_NAME),
)


# kaggle api
if UPLOAD:
    os.chdir(os.path.join(DATASET_DIR, DATASET_NAME))
    subprocess.run(f"kaggle datasets metadata hutch1221/{DATASET_NAME}", shell=True)
    subprocess.run(f"kaggle datasets version -r zip -m '{DATASET_NAME}'", shell=True)
