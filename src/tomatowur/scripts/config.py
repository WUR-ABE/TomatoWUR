# Config.py

from pathlib import Path

# import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize_config_dir, compose

from itertools import product
import argparse
import json

latest_version = 1.0


## TODO
class Config(DictConfig):
    pass


def convert_paths(cfg):
    """
    Recursively convert string values that look like paths into pathlib.Path objects.
    Operates in-place on OmegaConf DictConfig.
    """
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            convert_paths(value)
        elif isinstance(value, str) and ("/" in value or "\\" in value):
            try:
                # Optionally check if it's a valid path before converting
                path = Path(value)
                cfg[key] = path
            except Exception as e:
                pass  # Leave it as string if Path conversion fails


def flatten_cfg(cfg, parent_key='', sep='.'):
    """Flatten nested dictionary (with dot-separated keys)."""
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_cfg(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(flat_dict, sep='.'):
    nested_dict = {}
    for compound_key, value in flat_dict.items():
        keys = compound_key.split(sep)
        d = nested_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    return nested_dict

def create_config_list(train_cfg: DictConfig):
    """
    Generates a list of configuration dictionaries by creating all possible
    combinations of sweepable fields (list-like values) in the input configuration.
    Args:
        train_cfg (DictConfig): Input configuration, potentially containing
                                sweepable fields (lists).
    Returns:
        List[dict]: A list of dictionaries, each representing a unique combination
                    of the input configuration's sweepable fields.
    """
        #     # If value is a dict, recursively expand its sweepable fields
        # if isinstance(v, dict):
        #     # sub_combos =OmegaConf.create(v)
        #     # Each sub_combo is a dict; flatten keys
        #     for k2, v2 in v.items():
        #         if isinstance(v2, dict):
        #             raise NotImplementedError("dict of dict currently not supported")
        #         elif isinstance(v2, list):
        #             sweep_keys.append(k2)
        #             sweep_values.append(v2)
        #         else:
        #             sweep_keys.append(k2)
        #             sweep_values.append([v2])  # Wrap single value to keep shape

        #         # flat_combo = {f"{k}.{sub_k}": sub_v for sub_k, sub_v in sub_combo.items()}
        #         # sweep_keys.extend(flat_combo.keys())
        #         # sweep_values.extend([[val] for val in flat_combo.values()])

    # Convert to plain dict (if needed)
    cfg = OmegaConf.to_container(train_cfg, resolve=True)

    flat_cfg = flatten_cfg(cfg)


    # Identify sweepable fields (lists)
    sweep_keys = []
    sweep_values = []

    for k, v in flat_cfg.items():

        # Treat anything list-like as a sweep dimension
        if isinstance(v, list):
            sweep_keys.append(k)
            sweep_values.append(v)
        else:
            sweep_keys.append(k)
            sweep_values.append([v])  # Wrap single value to keep shape

    # Cartesian product of all sweep values
    combos = []
    for values in product(*sweep_values):
        combo_flat = dict(zip(sweep_keys, values))
        combo_nested = unflatten_dict(combo_flat)
        combos.append(combo_nested)
        

    return combos


def init_config(cfg_filename, overrides=[]):
    """
    Initializes and loads a configuration file.
    Args:
        cfg_filename (str or Path): Path to the configuration file.
        overrides (list, optional): List of override parameters to modify the configuration. Defaults to an empty list.
    Returns:
        OmegaConf.DictConfig: The loaded and processed configuration object.
    """

    cfg_filename = Path(cfg_filename)
    config_dir = str(cfg_filename.parent.resolve())
    config_name = cfg_filename.stem

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
        # Disable "struct mode" so we can add new keys
        OmegaConf.set_struct(cfg, False)
        cfg["cfg_filename"] = cfg_filename

    if cfg.best_settings is not None:
        with open(cfg.best_settings,"r") as f:
            data = json.load(f)
        overriden = [x.split("=")[0] for x in overrides]
        data_checked = data.copy()
        for key, value in data.items():
            new_key = f"{cfg.skeleton_method}.{key}"
            if new_key in overriden:
                data_checked.pop(key)
        cfg[cfg.skeleton_method].update(data_checked)
        print("Succesfully loaded optimised model")

        # Apply overrides to cfg (if any)
        # if overrides:
        #     for override in overrides:
        #         if "=" in override:
        #             key, value = override.split("=", 1)
        #             # Try to interpret value as JSON (for lists, dicts, bools, etc.)
        #             try:
        #                 value = json.loads(value)
        #             except Exception:
        #                 pass  # Keep as string if not JSON
        #             OmegaConf.update(cfg, key, value, merge=True)

    if cfg.config_version != latest_version:
        assert (
            cfg.config_version < latest_version
        ), f"Config version {cfg.config_version} is outdated. Latest version is {latest_version}."

    # cfg["cfg_filename"] = cfg_filename
    # Resolve interpolations
    OmegaConf.resolve(cfg)

    # Convert path-like strings to Path objects
    convert_paths(cfg)

    # create training thing
    # create_config_list(cfg.train)

    return cfg


def init_config_from_args():
    """
    Parses command-line arguments to initialize a configuration.
    This function uses argparse to parse the '--config' argument, which specifies
    the path to a YAML configuration file. Additional arguments are treated as
    overrides for the configuration.

    Example:
    python3 config.py --config config/sample_insseg run_modes=[train,evaluate]

    Returns:
        cfg: The initialized configuration object.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Name of the config YAML file",
    )
    args, overrides = parser.parse_known_args()
    cfg = init_config(args.config, overrides)
    return cfg



def save_config(cfg, save_path: Path):
    """
    Saves the configuration object to a file in a JSON-serializable format.
    Args:
        cfg (DictConfig): The configuration object to save.
        save_path (Path): The path where the configuration should be saved.
    """
    # Convert DictConfig to a plain dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # Convert any Path objects to strings for JSON serialization
    for key, value in cfg_dict.items():
        if isinstance(value, Path):
            cfg_dict[key] = str(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, Path):
                    value[sub_key] = str(sub_value)

    # Save the dictionary as a YAML file
    with open(save_path, "w") as f:
        OmegaConf.save(cfg_dict, f)

if __name__ == "__main__":
    # cfg = init_config_from_args()
    cfg = init_config("config.yaml")
