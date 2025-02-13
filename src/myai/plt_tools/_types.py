import typing as T
from collections import abc
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_Linestyles = T.Literal["solid", "dashed", "dashdot", "dotted", "-", "--", "-.", ":"]
_JoinStyles = T.Literal["butt", "projecting", "round"]
_CapStyles = T.Literal["miter", "round", "bevel"]
_FillStyles = T.Literal["full", "left", "right", "bottom", "top", "none"]
_DrawStyles = T.Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"]
_Hatch = T.Literal["/", "", "|", "-", "+", "x", "o", "O", ".", "*"]
_FontFamilies = T.Literal['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
_FontSizes = T.Literal['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']
_FontStretches = T.Literal['ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded']
_FontStyles = T.Literal['normal', 'italic', 'oblique']
_FontVariants = T.Literal['normal', 'small-caps']
_FontWeigths = T.Literal['ultralight', 'light', 'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black']
_HorizonalAlignment = T.Literal['left', 'center', 'right']
_Rotations = T.Literal['vertical', 'horizontal']
_VerticalAlignment = T.Literal['baseline', 'bottom', 'center', 'center_baseline', 'top']

class _K_Figure(T.TypedDict, total=False):
    figure: Figure | None
    ax: Axes | None
    figsize: float | abc.Sequence[float] | None
    dpi: float | None
    facecolor: T.Any | None
    edgecolor: T.Any | None
    frameon: bool
    layout: T.Literal["constrained", "compressed", "tight", "none"] | None
    projection: (
        T.Literal["aitoff", "hammer", "lambert", "mollweide", "polar", "rectilinear"]
        | str
        | None
    )
    polar: bool
    label: str


class _K_Line2D(T.TypedDict, total=False):
    alpha: float | None
    antialiased: bool
    aa: bool
    color: T.Any
    c: T.Any
    dash_capstyle: _CapStyles
    dash_joinstyle: _JoinStyles
    drawstyle: _DrawStyles
    ds: _DrawStyles
    fillstyle: _FillStyles
    gapcolor: T.Any | None
    label: str | None
    linestyle: _Linestyles
    ls: _Linestyles
    linewidth: float | None
    lw: float
    marker: str
    markeredgecolor: T.Any
    mec: T.Any
    markeredgewidth: T.Any
    mew: T.Any
    markerfacecolor: T.Any
    mfc: T.Any
    markerfacecoloralt: T.Any
    mfcalt: T.Any
    markersize: float
    ms: float
    markevery: int | tuple[int, int] | slice | list[int] | float | tuple[float, float] | list[bool]
    mouseover: bool
    solid_capstyle: _CapStyles
    solid_joinstyle: _JoinStyles
    zorder: float



class _K_Collection(T.TypedDict, total=False):
    """https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.Collection"""
    edgecolors: T.Any
    facecolors: T.Any
    linewidths: float | abc.Sequence[float] | None
    linestyles: _Linestyles | abc.Sequence[_Linestyles]
    capstyle: _CapStyles
    joinstyle: _JoinStyles
    antialiaseds: bool | abc.Sequence[bool]
    offsets: abc.Sequence[float] | abc.Sequence[abc.Sequence[float]]
    cmap: T.Any
    norm: T.Any
    hatch: _Hatch
    pickradius: float
    zorder: float


class _K_Text(T.TypedDict, total=False):
    alpha: float | None
    antialiased: bool
    backgroundcolor: T.Any
    bbox: T.Any
    color: T.Any
    c: T.Any
    fontfamily: str | _FontFamilies
    fontname: str | _FontFamilies
    family: str | _FontFamilies
    fontproperties: str
    font: str
    font_properties: str
    fontize: float | _FontSizes
    size: float | _FontSizes
    fontstretch: float | _FontStretches
    stretch: float | _FontStretches
    fontstyle: _FontStyles
    style: _FontStyles
    fontvariant: _FontVariants
    variant:  _FontVariants
    fontweight: float | _FontWeigths
    weight: float |  _FontWeigths
    horizontalalignment: _HorizonalAlignment
    ha: _HorizonalAlignment
    linespacing: float
    math_fontfamily: str
    mouseover: bool
    multialignment: _HorizonalAlignment
    ma: _HorizonalAlignment
    parse_math: bool
    position: tuple[float, float]
    rasterized: bool
    rotation: float | _Rotations
    verticalalignment: _VerticalAlignment
    va: _VerticalAlignment
    visible: bool
    wrap: bool
    x: float
    y: float
    zorder: float
