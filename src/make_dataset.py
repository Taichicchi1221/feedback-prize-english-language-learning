import os
import glob
import json
import subprocess
import shutil

UPLOAD = True
INIT = True

USERNAME = "hutch1221"

DATASET_DIR = "../datasets"
DATASET_NAME = "feedback3-longformer-base-attention-fold10"

EXPERIMENT_ID = "48"
RUN_ID = "307296e54a424c48810c61c6e32c8ad9"


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
    "id": USERNAME + "/" + DATASET_NAME,
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
    if INIT:
        command = "kaggle datasets create -r zip"
        print("###", command)
        subprocess.run(command, shell=True)
    else:
        command = f"kaggle datasets metadata {USERNAME}/{DATASET_NAME}"
        print("###", command)
        subprocess.run(command, shell=True)

        command = f"kaggle datasets version -r zip -m '{DATASET_NAME}'"
        print("###", command)
        subprocess.run(command, shell=True)
