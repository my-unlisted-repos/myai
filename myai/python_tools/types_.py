from typing import TypeVar, Protocol

_T_co = TypeVar("_T_co", covariant=True)

class SupportsNext(Protocol[_T_co]):
    def __next__(self) -> _T_co: ...

class HasIterDunder(Protocol[_T_co]):
    def __iter__(self) -> _T_co: ...

class SupportsGetitem(Protocol[_T_co]):
    def __getitem__(self, __k: int, /) -> _T_co: ...

class SupportsLenAndGetitem(Protocol[_T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, __k: int, /) -> _T_co: ...

type SupportsIter[_T_co] = HasIterDunder[_T_co] | SupportsLenAndGetitem[_T_co]
