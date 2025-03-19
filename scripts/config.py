# Config.py

from omegaconf import OmegaConf
from pathlib import Path

class Config():
    def __init__(self, config_name= "config.yaml"):

        self.cfg = OmegaConf.load('config.yaml')
        self.cfg = OmegaConf.to_container(self.cfg, resolve=True)  # Resolve ${} variables
        self._set_attributes(self.cfg)  # Recursively set attributes

    def _set_attributes(self, dictionary, parent_key=""):
        """ Recursively set dictionary keys as attributes, converting paths when needed """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                setattr(self, key, Config._dict_to_object(value))
            elif isinstance(value, str) and ('/' in value or '\\' in value): #("dir" in key or "path" in key):#('/' in value or '\\' in value):
                setattr(self, key, Path(value))  # Convert potential paths
            else:
                setattr(self, key, value)

    @staticmethod
    def _dict_to_object(d):
        """ Convert a dictionary to an object with attributes """
        obj = lambda: None  # Create an empty object
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(obj, key, Config._dict_to_object(value))  # Recursively convert dict
            elif isinstance(value, str) and ('/' in value or '\\' in value):
                setattr(obj, key, Path(value))  # Convert paths
            else:
                setattr(obj, key, value)
        return obj
    

if __name__=="__main__":
    obj = Config(config_name="config.yaml")
    print(obj.project_dir)