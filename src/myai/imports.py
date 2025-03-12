#pylint:disable=C0413
import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import copy
import csv
import functools
import itertools
import math
import operator
import random
import shutil
import sys
import time
import typing
from collections import abc
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from functools import partial
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchzero as tz
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchinfo import summary as __
from torchvision.transforms import v2
from tqdm import tqdm

from . import nn as mynn
from . import python_tools
from .data import DS
from .event_model.callback import Callback
from .event_model.conditional_callback import ConditionalCallback
from .logger import DictLogger

if sys.version_info[1] >= 12: from .learner import *
from .loaders.csv_ import csvread, csvwrite
from .loaders.image import imread, imreadtensor, imwrite
from .loaders.text import txtread, txtwrite
from .loaders.yaml import yamlread, yamlwrite
from .plt_tools import Fig, imshow, imshow_grid, linechart, scatter
from .python_tools import (
                           Compose,
                           ShutUp,
                           Split,
                           clean_mem,
                           compose,
                           find_file_containing,
                           flatten,
                           get0,
                           get1,
                           get__name__,
                           get_all_files,
                           get_first_file,
                           getlast,
                           identity,
                           identity_kwargs,
                           listdir_fullpaths,
                           maybe_compose,
                           path_zoom,
                           perf_counter_context,
                           pretty_print_dict,
                           print_callable_defaults,
                           reduce_dim,
                           time_context,
)
from .python_tools import SaveSignature as sig
from .python_tools import printargs as printa
from .torch_tools import (
                           count_params,
                           pad,
                           pad_dim,
                           pad_like,
                           pad_to_shape,
                           performance_tweaks,
)
from .transforms import normalize, tonumpy, totensor, znormalize

CUDA = torch.device('cuda')
CPU = torch.device('cpu')


summary = partial(__,  depth=100, col_names = ("input_size", "output_size", "kernel_size", "num_params", "trainable"), mode='train', row_settings =["depth", "var_names",])

