import re
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


def ga(file):
    code = """
    <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-NH9HWCW08F"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-NH9HWCW08F');
    </script>
    """

    a = os.path.dirname(file) + "/static/index.html"
    with open(a, "r") as f:
        data = f.read()
        if len(re.findall("G-", data)) == 0:
            with open(a, "w") as ff:
                newdata = re.sub("<head>", "<head>" + code, data)
                ff.write(newdata)
