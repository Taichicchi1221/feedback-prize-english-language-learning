import os
import glob
import json
import subprocess
import shutil

UPLOAD = True
INIT = True

DATASET_DIR = "../datasets"
DATASET_NAME = "deberta-v3-base-for-ensemble4"

EXPERIMENT_ID = "19"
RUN_ID = "be5c33f74c4f40e384dd4a375a740afc"


MLFLOW_DIR = f"../mlruns/{EXPERIMENT_ID}"


if os.path.exists(os.path.join(DATASET_DIR, DATASET_NAME)):
    shutil.rmtree(os.path.join(DATASET_DIR, DATASET_NAME))

# copy files
print("#" * 30, RUN_ID, "#" * 30)

shutil.copytree(
    os.path.join(MLFLOW_DIR, RUN_ID, "artifacts"),
    os.path.join(DATASET_DIR, DATASET_NAME),
)


# make dataset metadata
METADATA = {
    "title": DATASET_NAME,
    "id": "hutch1221/" + DATASET_NAME,
    "licenses": [
        {
            "name": "CC0-1.0",
        }
    ],
}
with open(os.path.join(DATASET_DIR, DATASET_NAME, "dataset-metadata.json"), "w") as f:
    json.dump(METADATA, f)

# kaggle api
if UPLOAD:
    os.chdir(os.path.join(DATASET_DIR, DATASET_NAME))
    subprocess.run(f"kaggle datasets metadata hutch1221/{DATASET_NAME}", shell=True)
    if INIT:
        subprocess.run(f"kaggle datasets create -r zip", shell=True)
    else:
        subprocess.run(f"kaggle datasets version -r zip -m '{DATASET_NAME}'", shell=True)
