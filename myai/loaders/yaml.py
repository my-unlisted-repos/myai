from collections.abc import Mapping

import ruamel.yaml

def yamlread(file: str) -> dict:
    yaml = ruamel.yaml.YAML(typ='safe')
    with open(file, 'r', encoding = 'utf8') as f:
        return yaml.load(f)

def _ensure_dict(d: Mapping):
    d = dict(d)
    for k,v in d.copy().items():
        if isinstance(v, Mapping):
            d[k] = _ensure_dict(v)
    return d

def yamlwrite(d: Mapping, file:str, sequence = 4, offset = 2):
    # ensure all numpy scalars and such are converted to python numbers
    d = _ensure_dict(d)
    for k, v in d.copy().items():
        if isinstance(v, bool): d[k] = bool(v)
        elif isinstance(v, int): d[k] = int(v)
        elif isinstance(v, float): d[k] = float(v)
        elif isinstance(v, str): d[k] = str(v)

    yaml = ruamel.yaml.YAML()
    yaml.indent(sequence=sequence, offset=offset)
    with open(file, 'w', encoding = 'utf8') as f:
        yaml.dump(d, f, )

