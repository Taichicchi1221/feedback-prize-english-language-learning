import os
import hydra


import train
import inference


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg):
    train.main(cfg)
    inference.main()


if __name__ == "__main__":
    main()
