from pprint import pprint

from src.train_pipeline import run_training


if __name__ == "__main__":
    output = run_training("configs/config.yaml")
    pprint(output)
