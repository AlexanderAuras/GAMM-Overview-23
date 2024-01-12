import argparse
import re
from functools import lru_cache
from pathlib import Path
from types import NoneType
from typing import Any, Dict, List, Optional, Union, cast

import torch
import yaml


class _Reference:
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path.split(".")


@lru_cache
def load_config(path: Union[str, Path], _toplevel: bool = True) -> Any:
    path = Path(path)

    class ConfigLoader(yaml.SafeLoader):
        pass

    ConfigLoader.add_constructor("!device", lambda l, n: torch.device(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))))
    ConfigLoader.add_implicit_resolver("!device", re.compile(r"^(?:cuda(?::\d+)?)|(?:cpu)$"), None)
    ConfigLoader.add_constructor("!path", lambda l, n: Path(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))))
    ConfigLoader.add_implicit_resolver("!path", re.compile(r"^file://"), None)
    ConfigLoader.add_constructor("!include", lambda l, n: load_config(Path(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))).resolve(), False) if Path(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))).is_absolute() else load_config(path.parent.joinpath(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))).resolve(), False))
    ConfigLoader.add_constructor("!eval", lambda l, n: eval(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))))
    ConfigLoader.add_constructor("!ref", lambda l, n: _Reference(cast(str, l.construct_scalar(cast(yaml.ScalarNode, n)))))
    # BUG pyyaml does not recognize all float literals correctly
    ConfigLoader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile("""^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)|\\.[0-9_]+(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*|[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$""", re.X), list("-+0123456789."))

    def resolve_references(element: Any, _full_config: Optional[Dict[str, Any]] = None) -> Any:
        if _full_config is None:
            _full_config = element
        if isinstance(element, _Reference):
            value = _full_config
            for path_part in element.path:
                value = cast(Dict[str, Any], value)[path_part]
            return value
        elif isinstance(element, Dict):
            return {k: resolve_references(v, _full_config) for k, v in element.items()}
        elif isinstance(element, List):
            return [resolve_references(x, _full_config) for x in element]
        return element

    with open(path, "r", encoding="UTF-8") as config_file:
        config = yaml.load(config_file, ConfigLoader)
        if not isinstance(config, Dict):
            raise RuntimeError("Configuration root element must be a dictionary/mapping")
        if _toplevel:
            config = resolve_references(config)
        return _to_obj(config)


def save_config(path: Union[str, Path], config: Any) -> None:
    path = Path(path)

    class ConfigDumper(yaml.SafeDumper):
        pass

    ConfigDumper.add_representer(torch.device, lambda d, o: d.represent_str(str(o)))
    ConfigDumper.add_representer(Path, lambda d, o: d.represent_str("file://" + str(o)))

    with open(path, "w", encoding="UTF-8") as config_file:
        yaml.dump(_to_dict(config), config_file, ConfigDumper)


def _to_obj(any: Any) -> Any:
    if isinstance(any, List):
        return [_to_obj(x) for x in any]
    elif isinstance(any, Dict):
        return type("__AnonymousConfig__", (object,), {k: _to_obj(v) for k, v in any.items()})
    return any


def _to_dict(obj: Any) -> Dict[str, Any]:
    result = {}
    for attr_name, attr in vars(obj).items():
        if attr_name.startswith("_"):
            continue
        if isinstance(attr, List):
            result[attr_name] = [_to_dict(x) for x in attr]
        elif isinstance(attr, (NoneType, bool, int, float, str, Path, torch.device)):
            result[attr_name] = attr
        else:
            result[attr_name] = _to_dict(attr)
    return result


def add_arpgarse_arguments(parser: argparse.ArgumentParser, config: Any) -> None:
    for attr_name, attr in vars(config).items():
        if attr_name.startswith("_"):
            continue
        if isinstance(attr, bool):
            parser.add_argument(f"--{attr_name}", type=bool, action="store_true", default=attr)
        if isinstance(attr, int):
            parser.add_argument(f"--{attr_name}", type=int, default=attr)
        if isinstance(attr, float):
            parser.add_argument(f"--{attr_name}", type=float, default=attr)
        if isinstance(attr, str):
            parser.add_argument(f"--{attr_name}", default=attr)
        if isinstance(attr, Path):
            parser.add_argument(f"--{attr_name}", type=Path, default=attr)
        if isinstance(attr, torch.device):
            parser.add_argument(f"--{attr_name}", type=torch.device, default=attr)


def process_arpgarse_arguments(args: argparse.Namespace, config: Any) -> None:
    for attr_name in vars(config).keys():
        if attr_name.startswith("_"):
            continue
        if attr_name in vars(args).keys():
            setattr(config, attr_name, getattr(args, attr_name))
