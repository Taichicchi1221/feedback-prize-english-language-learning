import os
import hydra

import pretrain


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg):
    pretrain.main(cfg)


if __name__ == "__main__":
    main()
