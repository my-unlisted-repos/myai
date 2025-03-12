import bisect
import inspect
import typing as T
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
import logging
import functools
from .callback import Callback, _CallbackMethod, _get_order, _get_valid_methods
from .conditional_callback import ConditionalCallback

class CancelContext(Exception):
    """Raising `CancelContext('name')` ends an `event_model.context('name')` context."""

class EventModel:
    def __init__(self):
        self.callbacks: set[Callback] = set()
        """Set of callbacks."""
        self._events: dict[str, list[_CallbackMethod]] = {}
        """Dictionary of events."""
        self._default_events: dict[str, list[_CallbackMethod]] = {}
        """Dictionary of default events. When a key is not found in self.events, it is then accessed in default_events.
        This is used for all default methods that can be overridden by other callbacks."""

    def __getattr__(self, attr:str):
        if attr not in self._events and attr not in self._default_events:
            raise AttributeError(
                f'{attr} is not found in any of the callbacks, available attributes are: {tuple(self._events.keys())}, {tuple(self._default_events.keys())}'
            )
        return functools.partial(self.fire_event, attr)

    def _add_callback(self, callback: Callback, default):
        # add each method
        for name, method in _get_valid_methods(callback):

            # choose the events dict
            if default: events = self._default_events
            else: events = self._events

            # insort callback into the events dict
            if name not in events: events[name] = []
            bisect.insort_right(events[name], _CallbackMethod(callback, method), key = _get_order)

        self.callbacks.add(callback)
        callback.enter(self)

    def _add_conditional_callback(self, callback: ConditionalCallback):
        if len(callback.events) == 0:
            logging.warning("%s has no conditions specified, it will never run.", callback.__class__.__name__ )

        for name in callback.events:
            if name not in self._events: self._events[name] = []

            bisect.insort(
                self._events[name],
                # _conditional_run checks if any of the conditions for this event are satisfied
                _CallbackMethod(callback, functools.partial(callback._conditional_run, event=name)),
                key = _get_order
            )

        # add each method
        for name, method in _get_valid_methods(callback):

            # insort callback into the events dict
            if name not in self._events: self._events[name] = []
            bisect.insort(self._events[name], _CallbackMethod(callback, method), key = _get_order)

        self.callbacks.add(callback)
        callback.enter(self)

    def add_callback(self, callback: Callback | ConditionalCallback, default=False):
        if isinstance(callback, ConditionalCallback):
            if default: raise ValueError("Conditional callback can't be default")
            self._add_conditional_callback(callback)
        else:
            self._add_callback(callback, default)

    def get_callback(self, name: str):
        for cb in self.callbacks:
            if name == cb.__class__.__name__:
                return cb
        raise ValueError(f'No callback with name {name} found.')

    def remove_callbacks(self, callbacks: Callback | str | Iterable[Callback | str]) -> list[Callback]:
        # since there can be multiple of some type of callback
        # if it is accessed as a string multiple callbacks will be removed
        # so this function must return a list of callbacks
        # therefore it accepts a list of callbacks for consistency

        if isinstance(callbacks, (Callback, str)): callbacks = [callbacks]
        else: callbacks = list(callbacks)

        # find callbacks by their names
        for cb_name in callbacks.copy():
            if isinstance(cb_name, str):
                # iterate over all callbacks and compare names
                for self_cb in self.callbacks:
                    if cb_name.__class__.__name__ == cb_name:
                        callbacks.append(self_cb)

        removed: list[Callback] = []

        for cb in callbacks:
            if isinstance(cb, Callback):

                cb.exit(self)

                # remove callback from all events
                for name, method in _get_valid_methods(cb):

                    cb_method = _CallbackMethod(cb, method)

                    # remove works here because _CallbackMethod defines __eq__
                    if name in self._events:
                        self._events[name].remove(cb_method)
                    if name in self._default_events:
                        self._default_events[name].remove(cb_method)

                self.callbacks.remove(cb)

                removed.append(cb)

        return removed # actual callbacks returned in case it was a string

    def fire_event(self, event: str, *args, **kwargs) -> T.Any:
        """Fires an event under `name`.
        This calls `event` method in all callbacks that have it in order of their `order` attribute,
        passing *args and **kwargs to them.
        If those methods return something, this returns last returned value."""
        ret = None # returned value

        if event in self._events:
            for cb_method in self._events[event]:
                ret = cb_method(self, *args, **kwargs)

        elif event in self._default_events:
            for cb_method in self._default_events[event]:
                ret = cb_method(self, *args, **kwargs)

        return ret

    @contextmanager
    def context(self, name: str, after: tuple | tuple[str]=(), catch = (), on_cancel: tuple | tuple[str] = (), on_catch: tuple | tuple[str] = (), on_success: tuple | tuple[str] = ()):
        """Run a context that can be cancelled by raising CancelContext(name).

        :param name: Name of the context.
        :param after: Events to run after the context, including if context is cancelled or `catch` is catched.
        :param catch: Exceptions to catch.
        """
        try: yield
        except CancelContext as e:
            if str(e) != name: raise e
            for event in after: self.fire_event(event)
            for event in on_cancel: self.fire_event(event)
        except catch:
            for event in after: self.fire_event(event)
            for event in on_catch: self.fire_event(event)
        else:
            for event in after: self.fire_event(event)
            for event in on_success: self.fire_event(event)

    @contextmanager
    def with_extra_callbacks(self, callbacks: Callback | Iterable[Callback]):
        """Temporarily add extra callbacks"""
        if isinstance(callbacks, Callback): callbacks = [callbacks]
        try:
            for cb in callbacks: self.add_callback(cb)
            yield
        finally:
            for cb in callbacks: self.remove_callbacks(cb)

    @contextmanager
    def without_callbacks(self, callbacks: str | Callback | Iterable[str | Callback]):
        """Temporarily remove some callbacks"""
        removed = self.remove_callbacks(callbacks)
        try: yield
        finally:
            for cb in removed: self.add_callback(cb)