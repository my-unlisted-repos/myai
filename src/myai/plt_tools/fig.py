import typing as T
from collections.abc import Callable
from itertools import zip_longest
from operator import attrgetter, methodcaller
from types import NoneType

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..python_tools import (
    Compose,
    method2func,
    method2func_return_override,
    method2method_return_override,
)
from ._plot import _Plot
from ._types import _K_Text

_Fig: T.TypeAlias = "Fig"

class Fig:
    def __init__(self):
        self.plots: list[list[Callable[[_Plot], T.Any]]] = []
        self.titles: list[str | None] = []
        self.cur = 0
        self.fig_fns: list[Callable[[Fig], T.Any]] = []

    def __len__(self):
        return len(self.plots)

    def add(self, title: str | T.Any | None = None):
        self.plots.append([])
        self.titles.append(str(title)[:10000] if title is not None else None)
        self.cur = len(self.plots) - 1
        return self

    def get(self, i: int):
        self.cur = i
        return self

    def _add_plot_func(self, name: str, *args, **kwargs):
        if len(self.plots) == self.cur: self.add()
        self.plots[self.cur].append(methodcaller(name, *args, **kwargs))
        return self

    def _add_figure_func(self, name: str, *args, **kwargs):
        self.fig_fns.append(Compose(attrgetter('figure'), methodcaller(name, *args, **kwargs)))
        return self

    @method2method_return_override(_Plot.linechart, _Fig)
    def linechart(self, *args, **kwargs): return self._add_plot_func("linechart", *args, **kwargs)

    @method2method_return_override(_Plot.scatter, _Fig)
    def scatter(self, *args, **kwargs): return self._add_plot_func("scatter", *args, **kwargs)

    @method2method_return_override(_Plot.imshow, _Fig)
    def imshow(self, *args, **kwargs): return self._add_plot_func("imshow", *args, **kwargs)

    @method2method_return_override(_Plot.imshow_grid, _Fig)
    def imshow_grid(self, *args, **kwargs): return self._add_plot_func("imshow_grid", *args, **kwargs)

    @method2method_return_override(_Plot.grid, _Fig)
    def grid(self, *args, **kwargs): return self._add_plot_func("grid", *args, **kwargs)

    @method2method_return_override(_Plot.axtitle, _Fig)
    def axtitle(self, *args, **kwargs): return self._add_plot_func("axtitle", *args, **kwargs)

    @method2method_return_override(_Plot.figtitle, _Fig)
    def figtitle(self, *args, **kwargs): return self._add_plot_func("figtitle", *args, **kwargs)

    @method2method_return_override(_Plot.figsize, _Fig)
    def figsize(self, *args, **kwargs): return self._add_plot_func("figsize", *args, **kwargs)

    @method2method_return_override(_Plot.xlabel, _Fig)
    def xlabel(self, *args, **kwargs): return self._add_plot_func("xlabel", *args, **kwargs)

    @method2method_return_override(_Plot.ylabel, _Fig)
    def ylabel(self, *args, **kwargs): return self._add_plot_func("ylabel", *args, **kwargs)

    @method2method_return_override(_Plot.axlabels, _Fig)
    def axlabels(self, *args, **kwargs): return self._add_plot_func("axlabels", *args, **kwargs)

    @method2method_return_override(_Plot.legend, _Fig)
    def legend(self, *args, **kwargs): return self._add_plot_func("legend", *args, **kwargs)

    @method2method_return_override(_Plot.xlim, _Fig)
    def xlim(self, *args, **kwargs): return self._add_plot_func("xlim", *args, **kwargs)

    @method2method_return_override(_Plot.ylim, _Fig)
    def ylim(self, *args, **kwargs): return self._add_plot_func("ylim", *args, **kwargs)

    @method2method_return_override(_Plot.axis, _Fig)
    def axis(self, *args, **kwargs): return self._add_plot_func("axis", *args, **kwargs)

    @method2method_return_override(_Plot.ticks, _Fig)
    def ticks(self, *args, **kwargs): return self._add_plot_func("ticks", *args, **kwargs)

    @method2method_return_override(_Plot.tick_params, _Fig)
    def tick_params(self, *args, **kwargs): return self._add_plot_func("tick_params", *args, **kwargs)

    @method2method_return_override(_Plot.set_axis_off, _Fig)
    def set_axis_off(self, *args, **kwargs): return self._add_plot_func("set_axis_off", *args, **kwargs)

    @method2method_return_override(_Plot.axoff, _Fig)
    def axoff(self, *args, **kwargs): return self._add_plot_func("axoff", *args, **kwargs)

    @method2method_return_override(_Plot.xscale, _Fig)
    def xscale(self, *args, **kwargs): return self._add_plot_func("xscale", *args, **kwargs)

    @method2method_return_override(_Plot.yscale, _Fig)
    def yscale(self, *args, **kwargs): return self._add_plot_func("yscale", *args, **kwargs)

    @method2method_return_override(_Plot.segmentation, _Fig)
    def segmentation(self, *args, **kwargs): return self._add_plot_func("segmentation", *args, **kwargs)

    @method2method_return_override(_Plot.preset, _Fig)
    def preset(self, *args, **kwargs): return self._add_plot_func("preset", *args, **kwargs)

    @method2method_return_override(_Plot.funcplot1d, _Fig)
    def funcplot1d(self, *args, **kwargs): return self._add_plot_func("funcplot1d", *args, **kwargs)

    @method2method_return_override(_Plot.contourf, _Fig)
    def contour(self, *args, **kwargs): return self._add_plot_func("contour", *args, **kwargs)

    @method2method_return_override(_Plot.contourf, _Fig)
    def contourf(self, *args, **kwargs): return self._add_plot_func("contourf", *args, **kwargs)

    @method2method_return_override(_Plot.pcolormesh, _Fig)
    def pcolormesh(self, *args, **kwargs): return self._add_plot_func("pcolormesh", *args, **kwargs)

    @method2method_return_override(_Plot.colorbar, _Fig)
    def colorbar(self, *args, **kwargs): return self._add_plot_func("colorbar", *args, **kwargs)

    @method2method_return_override(_Plot.funcplot2d, _Fig)
    def funcplot2d(self, *args, **kwargs): return self._add_plot_func("funcplot2d", *args, **kwargs)

    @method2method_return_override(_Plot.path, _Fig)
    def path(self, *args, **kwargs): return self._add_plot_func("path", *args, **kwargs)

    @method2method_return_override(_Plot.vline, _Fig)
    def vline(self, *args, **kwargs): return self._add_plot_func("vline", *args, **kwargs)

    @method2method_return_override(_Plot.hline, _Fig)
    def hline(self, *args, **kwargs): return self._add_plot_func("hline", *args, **kwargs)

    @method2method_return_override(_Plot.line, _Fig)
    def line(self, *args, **kwargs): return self._add_plot_func("line", *args, **kwargs)

    @method2method_return_override(_Plot.origin_lines, _Fig)
    def origin_lines(self, *args, **kwargs): return self._add_plot_func("origin_lines", *args, **kwargs)

    @method2method_return_override(_Plot.point, _Fig)
    def point(self, *args, **kwargs): return self._add_plot_func("point", *args, **kwargs)

    @method2method_return_override(_Plot.quiver, _Fig)
    def quiver(self, *args, **kwargs): return self._add_plot_func("quiver", *args, **kwargs)

    @method2method_return_override(_Plot.funcplot2d_quiver, _Fig)
    def funcplot2d_quiver(self, *args, **kwargs): return self._add_plot_func("funcplot2d_quiver", *args, **kwargs)

    def show(
        self,
        nrows: int | float | None = None,
        ncols: int | float | None = None,
        figure: "Figure | _Plot | Fig | None" = None,
        figsize: float | tuple[float, float] | None = None,
        axsize: float | tuple[float, float] | None = None,
        dpi: float | None = None,
        facecolor: T.Any | None = None,
        edgecolor: T.Any | None = None,
        frameon: bool = True,
        layout: T.Literal["constrained", "compressed", "tight", "none"] | None = 'compressed',
        **label_kwargs: T.Unpack[_K_Text],
    ):
        # distribute rows and cols
        if ncols is None:
            if nrows is None:
                nrows = len(self.plots) ** 0.45
            ncols = len(self.plots) / nrows # type:ignore
        else:
            nrows = len(self.plots) / ncols # type:ignore

        # ensure rows and cols are correct
        if nrows is None or ncols is None: raise ValueError('shut up pylance')
        nrows = round(nrows)
        ncols = round(ncols)
        nrows = max(nrows, 1)
        ncols = max(ncols, 1)
        r = True
        while nrows * ncols < len(self.plots):
            if r: ncols += 1
            else: ncols += 1
            r = not r

        nrows = min(nrows, len(self.plots))
        ncols = min(ncols, len(self.plots))

        # create the figure if it is None
        if isinstance(figsize, (int,float)): figsize = (figsize, figsize)
        if isinstance(figure, (_Plot | Fig)): figure = figure.figure

        if axsize is not None:
            if isinstance(axsize, (int,float)): axsize = (axsize, axsize)
            figsize = (ncols*axsize[0], nrows*axsize[1])

        if figure is None:
            self.figure = plt.figure(
                figsize=figsize,
                dpi=dpi,
                facecolor=facecolor,
                edgecolor=edgecolor,
                frameon=frameon,
                layout=layout,
            )
        else:
            self.figure = figure

        # create axes
        _axes = self.figure.subplots(nrows = nrows, ncols = ncols)
        if isinstance(_axes, np.ndarray): self.axes = _axes.flatten()
        else: self.axes = [_axes]

        # plot
        for ax, label, fns in zip_longest(self.axes, self.titles, self.plots):
            plot = _Plot(ax=ax, figure=self.figure)

            if fns is not None:
                for fn in fns:
                    fn(plot)
            else:
                ax.set_axis_off()

            if label is not None:
                plot.axtitle(label, **label_kwargs)

    def clear(self):
        self.plots = []

    def savefig(
        self,
        path,
        nrows: int | float | None = None,
        ncols: int | float | None = None,
        figure: "Figure | _Plot | Fig | None" = None,
        figsize: float | tuple[float, float] | None = None,
        axsize: float | tuple[float, float] | None = None,
        dpi: float | None = None,
        facecolor: T.Any | None = None,
        edgecolor: T.Any | None = None,
        frameon: bool = True,
        layout: T.Literal["constrained", "compressed", "tight", "none"] | None = 'compressed',
        **label_kwargs: T.Unpack[_K_Text],
        ):
        loc = locals().copy()
        del loc['path'], loc['self']
        label_kwargs = loc.pop('label_kwargs')
        loc.update(label_kwargs)

        self.show(**loc)
        self.figure.savefig(path, bbox_inches='tight', pad_inches=0)
        # self.close()
        return self

    def close(self):
        plt.close(self.figure)

@method2func(Fig.linechart)
def linechart(*args, **kwargs) -> Fig: return Fig().add().linechart(*args, **kwargs)

@method2func(Fig.scatter)
def scatter(*args, **kwargs) -> Fig: return Fig().add().scatter(*args, **kwargs)

@method2func(Fig.imshow)
def imshow(*args, **kwargs) -> Fig: return Fig().add().imshow(*args, **kwargs)

@method2func(Fig.funcplot1d)
def funcplot1d(*args, **kwargs) -> Fig: return Fig().add().funcplot1d(*args, **kwargs)

@method2func(Fig.funcplot2d)
def funcplot2d(*args, **kwargs) -> Fig: return Fig().add().funcplot2d(*args, **kwargs)

@method2func(Fig.pcolormesh)
def pcolormesh(*args, **kwargs) -> Fig: return Fig().add().pcolormesh(*args, **kwargs)

def imshow_grid(images, labels = None, norm = 'norm', cmap = 'gray', fig = None, **kwargs) -> Fig:
    if labels is None:
        labels = [None] * len(images)

    if fig is None: fig = Fig()
    for image, label in zip(images, labels):
        fig.add(label).imshow(image, norm = norm, cmap = cmap, **kwargs).set_axis_off()

    return fig
