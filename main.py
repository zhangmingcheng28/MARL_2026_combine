import os
import random

import numpy as np
import torch
from arguments import build_config, get_args
from train import main as train_main
from evaluate import main as evaluate_main


def _set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def main():
    args = get_args()
    config = build_config(args)
    _set_global_seed(config.get("seed", 777))

    num_devices = torch.cuda.device_count()
    print("Number of GPUs:", num_devices)
    # Get the names of the available GPUs
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_devices)]
    print("GPU Names:", gpu_names)

    if config["device"] == "auto":
        if torch.cuda.is_available():
            config["device"] = "cuda"
            print("Using GPU")
        else:
            config["device"] = "cpu"
            print("Using CPU")
    elif config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("Device is set to cuda but CUDA is not available.")

    mode = getattr(args, "mode", "train").lower()

    if mode == "train":
        train_main(config)
    elif mode in ["eval", "evaluate", "test"]:
        evaluate_main(config)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Supported modes are: train, evaluate"
        )

if __name__ == "__main__":
    main()
