import os
import shutil
import glob
import subprocess

shutil.rmtree("../work")
os.makedirs("../work")
os.chdir("../work")

print(os.getcwd())

for path in glob.glob("../src/*"):
    shutil.copy(path, os.path.basename(path))

subprocess.run("python main.py", shell=True)
