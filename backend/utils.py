import psutil


def get_current_ram_usage():
    ram = psutil.virtual_memory()
    return ram.available / 1024 / 1024 / 1024, ram.total / 1024 / 1024 / 1024
