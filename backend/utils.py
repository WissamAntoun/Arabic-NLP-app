import psutil
from huggingface_hub import Repository
import os


def get_current_ram_usage():
    ram = psutil.virtual_memory()
    return ram.available / 1024 / 1024 / 1024, ram.total / 1024 / 1024 / 1024


def download_models(models):
    model_dirs = {}
    for model in models:
        model_dirs[model] = Repository(
            model, clone_from=f"https://huggingface.co/researchaccount/{model}"
        )
    return model_dirs


def install_git_lfs():
    os.system("apt-get install git-lfs")
    os.system("git lfs install")
