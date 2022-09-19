import os
import hydra


import train
import inference


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    train.main(cfg)
    inference.main()


if __name__ == "__main__":
    main()
