'''
MIT License

Copyright (c) 2022 Jia-Qi Yang

https://github.com/ThyrixYang/config_tool

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import re
import yaml
import pprint
import inspect


def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def deep_filter(mapping, selector):
    new_mapping = {}
    for k, v in selector.items():
        if k in mapping and isinstance(mapping[k], dict) and isinstance(v, dict):
            new_mapping[k] = deep_filter(mapping[k], v)
        else:
            new_mapping[k] = mapping[k]
    return new_mapping


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def config_to_dict(x):
    if isinstance(x, Config):
        res = {}
        for k, v in x._config_dict.items():
            res[k] = config_to_dict(v)
        return res
    else:
        return x


def config_usage_to_dict(x, key):
    if isinstance(x, Config):
        res = {}
        for k, v in x._config_dict.items():
            res[k] = config_usage_to_dict(v, key)
        if hasattr(x, "_usage_state"):
            for kk, vv in x._usage_state.items():
                res[kk] = vv[key]
        return res
    else:
        return {}


def flatten_config(c, prefix=""):
    res = {}
    for k, v in c._config_dict.items():
        if prefix == "":
            name = k
        else:
            name = prefix + "." + k
        if isinstance(v, Config):
            res.update(flatten_config(v, name))
        else:
            res[name] = v
    return res


class Config:

    def __init__(self,
                 config_dict={},
                 usage_state_level="count"):
        self.reset_config(config_dict=config_dict,
                          usage_state_level=usage_state_level)
        
    def update(self, updates, usage_state_level="count"):
        config_dict = config_to_dict(self)
        for k, v in updates.items():
            if k in config_dict:
                config_dict[k] = type(config_dict[k])(v)
            else:
                config_dict[k] = v
        self.reset_config(config_dict=config_dict,
                          usage_state_level=usage_state_level)
        return self

    def reset_config(self, config_dict, usage_state_level):
        assert usage_state_level in ["none", "count", "hist"]
        self._config_dict = {}
        self._usage_state = {}
        self._usage_state_level = usage_state_level
        for k in config_dict.keys():
            if isinstance(config_dict[k], dict):
                self._config_dict[k] = Config(config_dict[k],
                                              usage_state_level=usage_state_level)
            else:
                self._config_dict[k] = config_dict[k]
                self._usage_state[k] = {"count": 0, "hist": []}

    def __getattr__(self, key):
        if not isinstance(self._config_dict[key], Config):
            if self._usage_state_level == "none":
                pass
            elif self._usage_state_level == "count":
                self._usage_state[key]["count"] += 1
            elif self._usage_state_level == "hist":
                self._usage_state[key]["count"] += 1
                s = [(f"filename: {x.filename}, line: {x.lineno}, code: {x.code_context}")
                     for x in inspect.stack()[1:]]
                self._usage_state[key]["hist"].append(s)
            else:
                raise ValueError()
        return self._config_dict[key]

    def __str__(self):
        d = config_to_dict(self)
        return pprint.pformat(d)
        
    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        state = config_to_dict(self)
        return state

    def __setstate__(self, state):
        self.reset_config(state, usage_state_level="none")

    def to_file(self, path):
        with open(path, "w") as f:
            yaml.dump(
                config_to_dict(self),
                f,
                default_flow_style=False)
            
    def to_dict(self):
        return config_to_dict(self)

def _load_config(file_path):
    with open(file_path + ".yaml", "r") as f:
        return yaml.load(f, Loader=loader)


def check_config_path(file_path):
    num_sub = file_path.count("-")
    assert num_sub <= 1
    if num_sub == 1:
        pos_sub = file_path.rfind("-")
        pos_add = file_path.rfind("+")
        assert pos_add < pos_sub
    return True


def load_config(file_path, usage_state_level="hist"):
    if file_path == "":
        return Config({}, usage_state_level=usage_state_level)
    prefix = "/".join(file_path.split("/")[:-1])
    file_path = file_path.split("/")[-1]

    check_config_path(file_path)
    if file_path.endswith(".yaml"):
        file_path = file_path.replace(".yaml", "")
    file_paths = file_path.split("-")
    if len(file_paths) > 1:
        assert len(file_paths) == 2
        sub_path = file_paths[1]
    else:
        sub_path = None
    file_path = file_paths[0]
    file_paths = file_path.split("+")
    config = {}
    for i in range(len(file_paths)):
        _config = _load_config(os.path.join(prefix, file_paths[i]))
        if _config is not None:
            config = deep_update(config, _config)
    if sub_path is not None:
        _config = _load_config(os.path.join(prefix, sub_path))
        if _config is not None:
            config = deep_filter(config, _config)
    return Config(config, usage_state_level=usage_state_level)
