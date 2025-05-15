import inspect
import os
import time
import typing as T
import warnings
from collections import abc
from datetime import datetime

import numpy as np
import torch
import torchzero as tz

from ..event_model import Callback, EventModel
from ..loaders.text import txtwrite
from ..loaders.yaml import yamlread, yamlwrite
from ..logger.base_logger import BaseLogger
from ..logger.dict_logger import DictLogger
from ..python_tools import (
    SaveSignature,
    epoch_to_datetime,
    get__name__,
    get_extra_signature,
    make_dict_serializeable,
    to_valid_fname,
)
from ..torch_tools import CUDA_IF_AVAILABLE, maybe_ensure_pynumber
from .callbacks.default import Default
from .callbacks.scheduler_ import scheduler as _scheduler_cb

DEFAULT_CALLBACKS = ()


if T.TYPE_CHECKING:
    from accelerate import Accelerator

def _tz_module_handler(v: "tz.core.Module"):
    if len(v.children) == 0: return v.__class__.__name__
    return f"{v.__class__.__name__}({'-'.join(m.__class__.__name__ for m in v.children.values())}"

def _tz_modular_handler(v: "tz.Modular"):
    if v.__class__.__name__ != "Modular": return v.__class__.__name__
    return f'M({"-".join(_tz_module_handler(m) for m in v.unrolled_modules)})'

_extra_type_handlers = {
    tz.Modular: _tz_modular_handler,
}

def _maybe_add_repr_(d: dict, attr):
    """maybe adds __repr__ to d if attr has a handler"""
    if type(attr) in _extra_type_handlers:
        d['__repr__'] = _extra_type_handlers[type(attr)](attr)

class Learner(EventModel):
    model: torch.nn.Module | T.Any
    loss_fn: abc.Callable
    optimizer: torch.optim.Optimizer | T.Any # type:ignore
    scheduler: torch.optim.lr_scheduler.LRScheduler | T.Any # type:ignore

    inputs: torch.Tensor | T.Any
    """Inputs object that gets passed to the model, gets assigned before running `forward`."""
    preds: torch.Tensor | T.Any
    """Outputs of the model, gets assigned after running `forward`. Callbacks can make changes or overwrite this on `after_forward`."""
    targets: torch.Tensor | T.Any
    """Targets object that gets passed to the loss function, gets assigned before running `get_loss`."""
    loss: torch.Tensor | T.Any
    """Output of the loss function, gets assigned after running `get_loss`."""
    batch: tuple[torch.Tensor | T.Any, torch.Tensor | T.Any] | T.Any
    """Batch, gets assigned at the beginning of `one_batch`. Doesn't get assigned on `inference`."""
    dltrain: abc.Iterable
    """Train dataloader, gets assigned before starting `fit` context."""
    dl: abc.Iterable
    """Current dataloader, gets assigned before starting `one_epoch` context."""
    def __init__(
        self,
        callbacks: Callback | abc.Iterable[Callback] = (),
        model: torch.nn.Module | SaveSignature | abc.Callable | None = None,
        loss_fn: abc.Callable | SaveSignature | None = None,
        optimizer: torch.optim.Optimizer | SaveSignature | T.Any | None = None, # type:ignore
        scheduler: torch.optim.lr_scheduler.LRScheduler | SaveSignature | T.Any | None = None,

        logger: BaseLogger | None = None,

        name: str = '{prefix} {model} {loss_fn} {optimizer}{optimizer.lr} {scheduler} {cbtext} {postfix} - {date_created}',
        main_metric: str = 'test accuracy',

        default_callbacks: Callback | abc.Iterable[Callback] = Default(),
    ):
        super().__init__()
        self.info: dict[str, dict | T.Any] = {}
        """Info dictionary, which is `{'model': {'__class__': 'ResNet', init_filters: 32, ...}}`"""
        self.prefix = ''
        """Prefix for the name"""
        self.postfix = ''
        """Postfix for the name"""
        self.accelerator: "Accelerator | T.Any" = None
        if logger is None: logger = DictLogger()
        self.logger: BaseLogger = logger
        self.name_template = name
        """By default this is `'{prefix} {model} {loss_fn} {optimizer}{optimizer.lr} {scheduler} {cbtext} {postfix} - {date_created}'`."""
        self.main_metric = main_metric
        """Main metric to be used for epoch dir name, by default `test accuracy`."""
        self.backward_kwargs = {}
        """Kwargs to pass to backward"""

        self.creation_time = time.time()
        self._dirs: dict[tuple[str | None, str], str] = {}
        """Directories for each root."""

        # counters
        self.cur_epoch = 0
        """Current epoch in fit."""
        self.cur_batch = 0
        """Current batch in train or test epoch."""
        self.num_forwards = 0
        """Total number of forward passes during training."""
        self.num_backwards = 0
        """Total number of backward passes during training."""

        self.total_epochs = 0
        """Total train epochs"""
        self.total_batches = 0
        """Total train batches"""

        self.status: T.Literal['init', 'train', 'test',] = 'init'
        """Current status, gets assigned at the beginning of each epoch and batch."""

        # set all attributes
        # some of those may be SaveSignature
        # which is an easy way to save all kwargs that stuff like optimizer uses
        # all kwargs are stored in `self.info`
        for attr, cls in (('model', model), ('loss_fn', loss_fn), ('optimizer', optimizer)):
            if cls is not None: self._set_x(attr, cls)
            else: setattr(self, attr, None)
        self.scheduler = scheduler

        # add all callbacks
        if isinstance(callbacks, Callback): callbacks = [callbacks]
        if isinstance(default_callbacks, Callback): default_callbacks = [default_callbacks]
        for c in default_callbacks: self.add_callback(c, default=True)
        for c in callbacks: self.add_callback(c)
        if scheduler is not None:
            self.add_callback(_scheduler_cb(scheduler))

    @property
    def device(self):
        return next(iter(self.model.parameters())).device

    def _set_x_cls[**P](self, attr: str, x: abc.Callable[P, T.Any], *args: P.args, **kwargs: P.kwargs):
        """sets `self.attr = x(*args, **kwargs)` and saves args and kwargs into `self.info`."""
        arglist = list(args)
        for i,v in enumerate(arglist):
            if isinstance(v, SaveSignature): arglist[i] = v.resolve()

        for k,v in kwargs.items():
            if isinstance(v, SaveSignature): kwargs[k] = v.resolve()

        setattr(self, attr, x(*arglist, **kwargs)) # type:ignore
        self.info[attr] = make_dict_serializeable(
            get_extra_signature(x, *args, **kwargs), raw_strings=False, recursive=True, type_handlers=_extra_type_handlers
        )
        self.info[attr]['__class__'] = get__name__(getattr(self, attr))
        _maybe_add_repr_(self.info[attr], getattr(self, attr))

        return self

    def _set_x(self, attr: str, x, params: abc.Mapping[str, T.Any] | None = None):
        """sets `self.attr = x`, and saves params into `self.info`"""
        # SaveSignature contains x(*args, **kwargs) as well as signature of x
        # we set x and save signature by using _set_x_cls
        if isinstance(x, SaveSignature):
            setattr(self, attr, x.resolve())
            self.info[attr] = make_dict_serializeable(x.extra_signature(), raw_strings=False, recursive=True, type_handlers=_extra_type_handlers)
            _maybe_add_repr_(self.info[attr], x.resolve())
            return self

        # else just set the attribute
        setattr(self, attr, x)
        self.info[attr] = {"__class__": get__name__(x)}
        _maybe_add_repr_(self.info[attr], x)
        if params is not None: self.info[attr].update(make_dict_serializeable(params, raw_strings=False, recursive=True, type_handlers=_extra_type_handlers))
        return self

    def set_model_cls[**P](self, cls: abc.Callable[P, torch.nn.Module | abc.Callable], *args: P.args, **kwargs: P.kwargs):
        return self._set_x_cls('model', cls, *args, **kwargs)
    def set_model(self, model: torch.nn.Module | abc.Callable): return self._set_x('model', model)

    def set_loss_cls[**P](self, cls: abc.Callable[P, abc.Callable], *args: P.args, **kwargs: P.kwargs):
        return self._set_x_cls('loss_fn', cls, *args, **kwargs)
    def set_loss(self, loss_fn: abc.Callable): return self._set_x('loss_fn', loss_fn)

    def set_optimizer_cls[**P](self, cls: abc.Callable[P, torch.optim.Optimizer | T.Any], *args: P.args, **kwargs: P.kwargs): # type:ignore
        return self._set_x_cls('optimizer', cls, *args, **kwargs)
    def set_optimizer(self, optimizer: torch.optim.Optimizer | T.Any): return self._set_x('optimizer', optimizer) # type:ignore

    def set_scheduler_cls[**P](self, cls: abc.Callable[P, torch.optim.lr_scheduler.LRScheduler | T.Any], *args: P.args, **kwargs: P.kwargs):
        return self._set_x_cls('scheduler', cls, *args, **kwargs)
    def set_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler | T.Any):
        return self._set_x('scheduler', scheduler)

    def add_named_info(self, name: str, **info: T.Any):
        """Add named misc. info, for example transforms."""
        if name not in self.info: self.info[name] = {}
        self.info[name].update(make_dict_serializeable(info, raw_strings=False, recursive=True, type_handlers=_extra_type_handlers))
        return self

    def add_info(self, **info):
        """Add misc. info"""
        return self.add_named_info('info', **info)

    def set_name(self, name: str = '{model} {loss_fn} {optimizer}{optimizer.lr} {scheduler} {postfix}'):
        """Sets name which is mainly used as directory name for saving stuff. Can use `{}`"""
        self.name_template = name
        return self

    def set_postfix(self, postfix: str):
        """Set postfix which may be used in the name. Can use `{}`."""
        self.postfix = postfix
        return self

    def set_prefix(self, prefix: str):
        """Set postfix which may be used in the name. Can use `{}`."""
        self.prefix = prefix
        return self

    def get_main_metric(self):
        if self.main_metric in self.logger:
            return self.logger.last(self.main_metric)
        return ""

    def set_main_metric(self, metric:str):
        self.main_metric = metric
        return self

    def _process_interp_template(self, s: str) -> str:
        """Preprocesses a template inside {} brackets."""
        parts = s.rsplit('.', 1)
        base = parts[0]
        if len(parts) == 2: attr = parts[1]
        else: attr = None

        # get from self.info
        if base in self.info:
            if attr is None:
                # print(f'{s = }')
                # print(f'{base = }')
                # print(f'{self.info = }')
                # print(f'{self.info[base] = }')
                if '__repr__' in self.info[base]: return str(self.info[base]['__repr__'])
                if '__class__' in self.info[base]: return str(self.info[base]['__class__'])
                if '__constructor__' in self.info[base]: return str(self.info[base]['__constructor__'])
                return ''
            if attr in self.info[base]: return str(self.info[base][attr])
            if attr in self.info['info']: return str(self.info['info'][attr])
            return ''

        # get from logger
        if base == 'logger':
            if attr is None: raise ValueError(f'Invalid template: {s}, {attr} not found in logger')
            if attr in self.logger:
                v = maybe_ensure_pynumber(self.logger.last(attr))
                if isinstance(v, float): v = f'{v:.4f}'
                return str(v)
            return ''

        # get some other attribute
        if base == 'date_created': return epoch_to_datetime(self.creation_time).strftime("%Y.%m.%d %H-%M-%S")
        if base == 'datetime': return datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        if base == 'main_metric':
            v = maybe_ensure_pynumber(self.get_main_metric())
            if isinstance(v, float): v = f'{v:.4f}'
            return str(v)
        if base == 'prefix':
            return self.process_template(self.prefix)
        if base == 'postfix':
            return self.process_template(self.postfix)
        if base == 'cbtext':
            return ' '.join(cb._learner_text for cb in self.callbacks)
        if base in dir(self):
            v = getattr(self, base)
            if v is not None: return str(v)
            return ''
        raise ValueError(f'Invalid template: {s}, {base} not found')

    def process_template(self, template: str) -> str:
        """Processes a template like `'{total_epochs} {logger.test loss} {main_metric}'`"""
        template = template.replace('{{', '__BRACEOPEN__').replace('}}', '__BRACECLOSE__')
        # "stuff {attr1} {attr2}"
        if template.count('{') != 0:
            starts = template.split('{')

            for i, s in enumerate(starts.copy()):
                if '}' in s:
                    interp = s[:s.find('}')]
                    starts[i] = starts[i].replace(f'{interp}}}', self._process_interp_template(interp))

            string = ''.join(starts).replace('__BRACEOPEN__', '{').replace('__BRACECLOSE__', '}')
        else:
            string = template
        while '  ' in string: string = string.replace('  ', ' ')
        return string.strip()

    def get_learner_dir(self, root: str = 'runs', prefix = None, postfix=None):
        """Creates a directory if it doesn't exist and returns the path. The path is `root(/prefix)/name(/postfix)`"""
        if (prefix, root) in self._dirs:
            dir = self._dirs[(prefix, root)]

            # maybe add postfix
            if postfix is not None:
                dir = os.path.join(dir, self.process_template(postfix))
                if not os.path.exists(dir): os.mkdir(dir)

            return dir

        if not os.path.exists(root): os.mkdir(root)

        # maybe add prefix
        if prefix is not None:
            root = os.path.join(root, self.process_template(prefix))
            if not os.path.exists(root): os.mkdir(root)

        dir = os.path.join(root, to_valid_fname(self.name))
        if os.path.exists(dir):
            dir = f'{dir} 2'
            c = 2
            while os.path.exists(dir):
                c += 1
                dir = f'{dir[:-2]} {c}'

        os.mkdir(dir)

        self._dirs[(prefix, root)] = dir

        # maybe add postfix
        if postfix is not None:
            dir = os.path.join(dir, self.process_template(postfix))
            if not os.path.exists(dir): os.mkdir(dir)

        return dir


    def get_epoch_dir(self, root: str = 'runs', epoch_template = '{total_epochs} {cur_batch} {logger.test loss} {main_metric}', prefix = None, postfix=None):
        """Creates a directory if it doesn't exist and returns the path. The path is `root/name(/prefix)/epoch(/postfix)`"""
        # get root dir
        dir = self.get_learner_dir(root)

        # maybe add prefix
        if prefix is not None:
            dir = os.path.join(dir, self.process_template(prefix))
            if not os.path.exists(dir): os.mkdir(dir)

        # add epoch dir
        dir = os.path.join(dir, self.process_template(epoch_template))
        if not os.path.exists(dir): os.mkdir(dir)

        # maybe add postfix
        if postfix is not None:
            dir = os.path.join(dir, self.process_template(postfix))
            if not os.path.exists(dir): os.mkdir(dir)

        return dir

    @property
    def name(self):
        """By default this is `{model} {loss_fn} {optimizer}{optimizer.lr} {scheduler} {postfix} {date_created}`"""
        n = self.process_template(self.name_template)
        if len(n) == 0 or n == ' ': n = 'empty-name'
        return n

    def set_use_closure(self, use_closure: bool):
        """Whether to pass closure to optimizer. When Learner is created, this is set to True by default."""
        cb: Default = self.get_callback('Default') # type:ignore
        cb._use_closure = use_closure
        return self

    def state_dict(self):
        """State dict. Saves the following attributes: ones that have `state_dict`,
        and ones that are `int, float, str, bool, None, np.ndarray, torch.Tensor`.
        This includes attributes like total_batches, etc."""
        state_dict = {}
        for attr in dir(self):

            # skip private methods
            if attr.startswith('_') or attr == 'callbacks': continue

            # skip properties
            if attr in dir(type(self)) and isinstance(getattr(type(self), attr), property): continue

            # get the atttribute
            x = getattr(self, attr)

            # skip methods
            if inspect.ismethod(x): continue

            # if it has a state_dict, save a dictionary with the state dict and the type
            if hasattr(x, 'state_dict'):
                try:
                    state_dict[attr] = {"state_dict": x.state_dict(), "type": get__name__(x)}
                except Exception as e:
                    warnings.warn(f'Failed to save {attr}:\n{e!r}')
            # if it is a serializeable object,
            elif isinstance(x, (int, float, str, bool, np.ndarray, torch.Tensor)) or x is None:
                state_dict[attr] = {"object": x}

            # else we store names and types of those objects so that they can be restored
            else:
                state_dict[attr] = {"type": get__name__(x)}
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Load a state dict."""
        for attr, value in state_dict.items():
            # load state_dict
            if 'state_dict' in value:
                self_attr = getattr(self, attr)
                if get__name__(self_attr) != value['type']:
                    warnings.warn(
                        f"Loading state dict for {attr} but type is different. \
                        Self.{attr} is {get__name__(self_attr)}, state_dict['{attr}'] is {value['type']}"
                    )
                getattr(self, attr).load_state_dict(value['state_dict'])

            # or just set object to its value
            elif 'object' in value:
                setattr(self, attr, value['object'])


    def save(self, dir: str, mkdir = True, state_dict=True, logger=True, info=True, text=True):
        """Saves this learner to a directory, creates files in that directory."""
        if not os.path.exists(dir) and mkdir: os.mkdir(dir)

        if info:
            # save info
            info = self.info.copy()
            info['cur_batch'] = self.cur_batch; info['cur_epoch'] = self.cur_epoch
            info['total_batches'] = self.total_batches; info['total_epochs'] = self.total_epochs
            info['num_forwards'] = self.num_forwards; info['num_backwards'] = self.num_backwards
            yamlwrite(info, os.path.join(dir, 'info.yaml'))

        # save state_dicts
        if state_dict:
            learner_attrs = {} # learner state_dict for stuff like cur_batch
            for k, v in self.state_dict().items():
                if k == 'logger': continue # logger is saved using `save`
                if 'state_dict' in v:
                    try:
                        torch.save(v['state_dict'], os.path.join(dir, f'{k}.state_dict'))
                    except Exception as e:
                        warnings.warn(f'Failed to save {k}:\n{e!r}')
                else: learner_attrs[k] = v
            torch.save(learner_attrs, os.path.join(dir, 'learner_attrs.state_dict'))

        # save logger
        if logger:
            self.logger.save(os.path.join(dir, 'logger.npz'))

        # save model and optimizer as strings
        if text:
            txtwrite(os.path.join(dir, 'model.txt'), str(self.model))
            txtwrite(os.path.join(dir, 'optimizer.txt'), str(self.optimizer))
            txtwrite(os.path.join(dir, 'logger.yaml'), self.logger.as_yaml_string(), )

    def load(self, dir: str):
        files = set(os.listdir(dir))
        if 'info.yaml' in files:
            self.info = yamlread(os.path.join(dir, 'info.yaml'))
            for k in ('cur_batch', 'cur_epoch', 'total_batches', 'total_epochs', 'num_forwards', 'num_backwards'):
                del self.info[k]

        if 'logger.npz' in files: self.logger.load(os.path.join(dir, 'logger.npz'))

        # load attrs like cur_batch
        if 'learner_attrs.state_dict' in files:
            learner_attrs: dict[str, T.Any] = torch.load(os.path.join(dir, 'learner_attrs.state_dict'), weights_only = False)
            for k, v in learner_attrs.items():
                if 'object' in v:
                    if k != 'device': # TODO TEMPORARY BECAUSE I HCANGED DEVICE INTO PROPERTY
                        setattr(self, k, v['object'])

        # load state_dicts
        for file in files:
            if file.endswith('.state_dict') and file != 'learner_attrs.state_dict':
                attr_name = file.replace('.state_dict', '')
                try:
                    attr = getattr(self, attr_name)
                    attr.load_state_dict(torch.load(os.path.join(dir, file), weights_only = False), )
                except Exception as e: warnings.warn(f"Failed to load state dict for {attr_name}: {e!r}")

        return self

    @property
    def training(self):
        return self.model.training
    # ---------------------------------------------------------------------------- #
    #                            callback based methods                            #
    # ---------------------------------------------------------------------------- #

    def log(self, metric: str, value: T.Any):
        self.fire_event('log', metric, value)

    def train(self):
        self.fire_event('train',)

    def eval(self):
        self.fire_event('eval',)

    def forward(self, inputs,):
        """Pass inputs through model and return predictions."""
        self.inputs = inputs
        self.preds = self.fire_event('forward', self.inputs)
        if self.status == 'train': self.num_forwards += 1
        self.fire_event('after_forward')
        return self.preds

    def get_loss(self, *args):
        """Evaluate loss value between preds and targets."""
        if len(args) == 2: self.targets = args[1]
        self.loss = self.fire_event('get_loss', *args)
        return self.loss

    def backward(self, loss: torch.Tensor, **kwargs):
        """Call backward on loss"""
        self.fire_event('backward', loss, **kwargs, **self.backward_kwargs)
        self.fire_event('after_backward')
        if self.status == 'train': self.num_backwards += 1

    def zero_grad(self, set_to_none: bool = True):
        """Zero grad"""
        self.fire_event('zero_grad', set_to_none = set_to_none)

    def closure(self, batch, backward=True) -> torch.Tensor:
        self.fire_event('before_any_step')
        self.fire_event(f'before_{self.status}_step')
        loss = self.fire_event('closure', batch = batch, backward=backward)
        self.fire_event('after_any_step')
        self.fire_event(f'after_{self.status}_step')
        return loss

    def make_closure(self, batch) -> abc.Callable[[], torch.Tensor]:
        return self.fire_event('make_closure', batch = batch)

    def inference(self, inputs, enable_grad = False):
        return self.fire_event('inference', inputs, enable_grad)

    def optimizer_step(self, *args, **kwargs):
        self.fire_event('optimizer_step', *args, **kwargs)
        self.fire_event('after_optimizer_step')

    def one_batch(self, batch, train: bool):
        self.batch = batch

        self.status = 'train' if train else 'test'

        self.fire_event('before_any_batch')
        self.fire_event(f'before_{self.status}_batch')

        self.fire_event('one_batch', batch = self.batch, train = train)
        if train: self.total_batches += 1

        self.fire_event(f'after_{self.status}_batch')
        self.fire_event('after_any_batch')

    def one_epoch(self, dl: abc.Iterable, train: bool):
        self.dl = dl
        self.status = 'train' if train else 'test'

        self.fire_event('before_any_epoch')
        self.fire_event(f'before_{self.status}_epoch')

        # run epoch context which catches CancelContext("epoch")
        with self.context('epoch', after = ('after_any_epoch', f'after_{self.status}_epoch')):
            self.fire_event('one_epoch', dl = self.dl, train = train)
            if train: self.total_epochs += 1
        # context runs `after_x` events on exit there

    def fit(self, dltrain: abc.Iterable, n_epochs: int, catch_kb_interrupt = True, ):
        # attrs
        self.dltrain = dltrain
        self.n_epochs = n_epochs
        self.epochs_iterator: abc.Iterable[int] = range(n_epochs)
        self.catch = [KeyboardInterrupt, ] if catch_kb_interrupt else []

        self.fire_event('before_fit')

        # run fit context which catches CancelContext("fit")
        try:
            with self.context('fit', after = ('after_fit', ), catch = tuple(self.catch)):
            # we pass dltrain and epochs_iterator to the event because
            # they can be replaced by other callbacks that are used
            # in the middle of training, and we don't want fit callback to break
                self.fire_event('fit', dltrain = self.dltrain, epochs_iterator = self.epochs_iterator, )
        except Exception as e:
            self.fire_event('on_fit_exception')
            raise e

        # context runs `after_fit` event on exit, CancelContext('fit') and optionally KeyboardInterrupt
