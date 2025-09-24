import math


from .io import safe_open_write, json_load, json_dump, json_get, text_get, try_run_cmd
from .visualization import get_camera_geometry


def create_bound_function(self, func):
    # Binds an external function to a class instance as if it was defined in the class
    def new_func(*args, **kwargs): return func(self, *args, **kwargs)
    return new_func


def format_big_number(num):
    if num < 1:
        return f'{num}'

    suffixes = ['', 'K', 'M', 'B', 'T']
    i = min(len(suffixes) - 1, int(math.floor(math.log10(num) / 3)))
    unit = suffixes[i]
    num = num / (10 ** (3 * i))

    return f"{num:.2f}{unit}"


def get_model_stats(model, print_stats=True):
    # TODO change to return model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    res = f'Total params: {format_big_number(total_params)}; Trainable params: {format_big_number(trainable_params)}'
    if print_stats:
        print(res)
    else:
        return res


def get_path_size(path):
    # Neither os or shutil give accurate results
    # TODO add windows equivalent
    return int(try_run_cmd(f'du -sb "{path}"', raise_err=True).split()[0])
