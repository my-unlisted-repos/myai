import sys
import types

if sys.version_info[1] < 12:
    from .p311 import *
else:
    from .classes import *
    from .f2f import *
    from .files import *
    from .functions import *
    from .identity import *
    from .iterables import *
    from .objects import *
    from .performance import *
    from .printing import *
    from .profiling import *
    from .progress import Progress
    from .relaxed_multikey_dict import RelaxedMultikeyDict, normalize_string
    from .serialization import *
    from .threading_ import *
    from .time import *
    from .types_ import *

    __all__ = [name for name, thing in globals().items() # type:ignore
            if not (name.startswith('_') or isinstance(thing, types.ModuleType))]
    del types

