import numpy as np
import psutil
import os
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


def get_current_ram_usage():
    ram = psutil.virtual_memory()
    return ram.available / 1024 / 1024 / 1024, ram.total / 1024 / 1024 / 1024


def download_models(models):
    for model in tqdm(models, desc="Downloading models"):
        logger.info(f"Downloading {model}")
        for i in range(0, 5):
            curr_dir = f"{model}/train_{i}/best_model/"
            os.makedirs(curr_dir)
            os.system(
                f"wget -q https://huggingface.co/researchaccount/{model}/resolve/main/train_{i}/best_model/config.json -P {curr_dir}"
            )
            os.system(
                f"wget -q https://huggingface.co/researchaccount/{model}/resolve/main/train_{i}/best_model/pytorch_model.bin -P {curr_dir}"
            )
            os.system(
                f"wget -q https://huggingface.co/researchaccount/{model}/resolve/main/train_{i}/best_model/special_tokens_map.json -P {curr_dir}"
            )
            os.system(
                f"wget -q https://huggingface.co/researchaccount/{model}/resolve/main/train_{i}/best_model/tokenizer_config.json -P {curr_dir}"
            )
            os.system(
                f"wget -q https://huggingface.co/researchaccount/{model}/resolve/main/train_{i}/best_model/training_args.bin -P {curr_dir}"
            )
            os.system(
                f"wget -q https://huggingface.co/researchaccount/{model}/resolve/main/train_{i}/best_model/vocab.txt -P {curr_dir}"
            )


def softmax(x):
    return np.exp(x) / sum(np.exp(x))
