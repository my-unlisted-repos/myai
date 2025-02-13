from contextlib import nullcontext
from typing import Any, Literal, TypeAlias, Unpack

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from torchvision.utils import make_grid

from ..python_tools import (
    func2func,
    method2func,
    method2method,
    method2method_return_override,
)
from ..torch_tools import (
    ensure_numpy_or_none_recursive,
    ensure_numpy_recursive,
    make_segmentation_overlay,
    maybe_detach_cpu,
    maybe_detach_cpu_recursive,
    maybe_ensure_pynumber,
)
from ..transforms import tonumpy, totensor
from ._norm import _normalize
from ._types import _FontSizes, _K_Collection, _K_Figure, _K_Line2D, _K_Text
from ._utils import _prepare_image_for_plotting

_PlotFwdRef: TypeAlias = "_Plot"
"""Forward reference to _Plot for the method2method_return_override decorator"""

def _get_grid(x, y, z, mode:Literal["linear", "nearest", "clough", "rbf"], xlim, ylim, zlim, step):
    from scipy.interpolate import (
        CloughTocher2DInterpolator,
        LinearNDInterpolator,
        NearestNDInterpolator,
        RBFInterpolator,
    )
    INTERPOLATORS = {
        'linear': LinearNDInterpolator,
        'nearest': NearestNDInterpolator,
        'clough': CloughTocher2DInterpolator,
        'rbf': RBFInterpolator,
    }
    if mode not in INTERPOLATORS: raise ValueError(f'Invalid mode {mode}')
    xmin, xmax = np.min(x), np.max(x)
    if xlim is not None:
        if xlim[0] is not None: xmin = xlim[0]
        if xlim[1] is not None: xmax = xlim[1]
    ymin, ymax = np.min(y), np.max(y)
    if ylim is not None:
        if ylim[0] is not None: ymin = ylim[0]
        if ylim[1] is not None: ymax = ylim[1]
    if zlim is not None:
        z = np.clip(z, *zlim)
    x_grid = np.linspace(xmin, xmax, step)
    y_grid = np.linspace(ymin, ymax, step)
    X, Y = np.meshgrid(x_grid,y_grid)
    interpolator = INTERPOLATORS[mode]((x, y), z)
    Z = interpolator(X, Y)
    return X, Y, Z

def _prepare_data_for_plotting(*data, x, y, ensure_x=False):
    if len(data) == 1:
        if (x is not None) and (y is not None): raise ValueError('both args and x or y are specified!')
        d = ensure_numpy_recursive(data[0])
        # data is 1d array
        if d.ndim == 1:
            if y is not None: x = data[0]
            else: y = data[0]
        # data is 2d array, by default assume it is sequence of (x, y) tuples
        else:
            if d.ndim > 2: d = d.squeeze()
            if d.ndim > 2: raise ValueError(d.shape)
            if d.ndim == 1: return _prepare_data_for_plotting(d, x=x, y=y)
            if d.shape[1] == 2:
                x = d[:,0]; y = d[:,1]
            elif d.shape[0] == 2:
                x,y = d
            else:
                raise ValueError(d.shape)

    elif len(data) == 2:
        if (x is not None) or (y is not None): raise ValueError('both args and x or y are specified!')
        x, y = data
    elif len(data) > 2: raise ValueError(f"got too many args {len(data) = }")

    if y is None: raise ValueError('No data to plot')
    x = ensure_numpy_or_none_recursive(x)
    y = ensure_numpy_recursive(y)

    if ensure_x and x is None: x = np.arange(0, len(x), 1) # type:ignore
    return x, y

class _Plot:
    def __init__(
        self,
        **kwargs: Unpack[_K_Figure],
    ):
        figure = kwargs.pop("figure", None)
        ax = kwargs.pop("ax", None)

        if "figsize" in kwargs and isinstance(kwargs["figsize"], (int, float)):
            kwargs["figsize"] = (kwargs["figsize"], kwargs["figsize"])

        kwargs.setdefault("layout", "constrained")

        if ax is None:
            if figure is None:
                fig_kwargs: dict = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in {"projection", "polar", "label"}
                }
                figure = plt.figure(**fig_kwargs)
            ax_kwargs: dict = {
                k: v for k, v in kwargs.items() if k in {"projection", "polar", "label"}
            }
            ax = figure.add_subplot(**ax_kwargs)

        if figure is None: figure = ax.get_figure()
        if figure is None: raise ValueError("figure is None")

        self.figure: Figure = figure # type:ignore
        self.ax: Axes = ax

    def grid(
        self,
        major_alpha: float | None = 0.3,
        minor_alpha: float | None = 0.1,
        axis: Literal['both', 'x', 'y']="both",
        **line_kwargs: Unpack[_K_Line2D]
    ):
        k: dict[str, Any] = dict(line_kwargs)
        k.pop('alpha', None)
        if major_alpha is not None and major_alpha > 0: self.ax.grid(which = 'major', axis=axis, alpha=major_alpha, **k)
        if minor_alpha is not None and minor_alpha > 0: self.ax.grid(which = 'minor', axis=axis, alpha=minor_alpha, **k)
        return self

    def linechart(
        self,
        *data,
        x=None,
        y=None,
        scalex=True,
        scaley=True,
        **line_kwargs: Unpack[_K_Line2D],
    ):
        x, y = _prepare_data_for_plotting(*data, x=x, y=y, ensure_x=False)
        if x is None: self.ax.plot(y, scalex=scalex, scaley=scaley, **line_kwargs)
        else: self.ax.plot(x, y, scalex=scalex, scaley=scaley, **line_kwargs)
        return self

    def scatter(
        self,
        *data,
        x = None,
        y = None,
        s=None,
        c=None,
        marker=None,
        vmin=None,
        vmax=None,
        alpha=None,
        plotnonfinite=False,
        **collection_kwargs: Unpack[_K_Collection],
    ):
        x, y = _prepare_data_for_plotting(*data, x=x, y=y, ensure_x=True)
        c = maybe_detach_cpu_recursive(c)
        s = maybe_detach_cpu_recursive(s)

        loc = locals().copy()
        del loc["self"], loc['data']
        collection_kwargs = loc.pop("collection_kwargs")
        loc.update(collection_kwargs)

        self.ax.scatter(**loc)
        return self

    @method2method_return_override(Axes.imshow, _PlotFwdRef)
    def imshow(self,*args,**kwargs,):
        if len(args) >= 3:
            args = list(args)
            norm = args.pop(3)
        else: norm = kwargs.pop('norm', None)

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'gray'


        if len(args) > 0:
            args = list(args)
            args[0] = _normalize(_prepare_image_for_plotting(ensure_numpy_recursive(args[0])), norm)

        elif 'X' in kwargs:
            kwargs['X'] = _normalize(_prepare_image_for_plotting(ensure_numpy_recursive(kwargs['X'])), norm)

        self.ax.imshow(*args, **kwargs)
        return self

    def segmentation(
        self,
        x: torch.Tensor | np.ndarray,
        alpha=0.3,
        bg_index = 0,
        colors=None,
        bg_alpha:float = 0.,
        **kwargs,
    ):
        x = totensor(maybe_detach_cpu_recursive(x), dtype=torch.float32).squeeze()
        # argmax if not argmaxed
        if x.ndim == 3:
            if x.shape[0] < x.shape[2]: x = x.argmax(0)
            else: x = x.argmax(-1)

        if x.ndim < 2: raise ValueError(f'Got x of shape {x.shape}')

        segm = make_segmentation_overlay(x, colors = colors, bg_index = bg_index) * 255
        segm = torch.cat([segm, torch.zeros_like(segm[0, None])], 0)
        segm[3] = torch.where(segm[:3].amax(0) > torch.tensor(0), torch.tensor(int(alpha*255)), torch.tensor(bg_alpha))

        self.ax.imshow(segm.permute(1,2,0).to(torch.uint8), **kwargs)
        return self

    def imshow_grid(
        self,
        x,
        nrows: int | None = None,
        ncols: int | None = None,
        padding: int = 2,
        value_range: tuple[int,int] | None = None,
        normalize: bool = True,
        scale_each: bool = False,
        pad_value: float = 0,
        **kwargs,
    ):
        x = torch.from_numpy(ensure_numpy_recursive(x))
        # add channel dim
        if x.ndim == 3: x = x.unsqueeze(1)
        # ensure channel first
        if x.shape[1] > x.shape[-1]: x = x.movedim(-1, 1)

        # distribute rows and cols
        if nrows is None:
            if ncols is None:
                ncols = len(x) ** 0.5
            nrows = len(x) / ncols # type:ignore

        # ensure rows are correct
        if nrows is None: raise ValueError('shut up pylance')
        nrows = round(nrows)
        nrows = max(nrows, 1)

        # make the grid
        grid = make_grid(x.float(), nrow=nrows, padding = padding, normalize=normalize, value_range=value_range, scale_each=scale_each, pad_value=pad_value)
        # this returns (C, H, W)

        return self.imshow(grid.moveaxis(0, -1), **kwargs,)

    @torch.no_grad
    def funcplot1d(
        self,
        func,
        start,
        stop,
        num = None,
        step = None,
        vectorize:Literal['list', 'numpy', 'torch', 'vmap'] | None = None,
        chunk_size = None,
        ylim=None,
        **line_kwargs: Unpack[_K_Line2D],
    ):
        if step is None and num is None: raise ValueError("funcplot1d needs either step or num")
        if step is None: step = (stop - start) / num
        if vectorize == 'list': values = func(np.arange(start, stop, step).tolist())
        elif vectorize == 'numpy': values = func(np.arange(start, stop, step))
        elif vectorize == 'torch': values = func(torch.arange(start, stop, step))
        elif vectorize == 'vmap': values = torch.vmap(func, chunk_size=chunk_size)(torch.arange(start, stop, step))
        else: values = [func(v) for v in np.arange(start, stop, step).tolist()]
        self.linechart(x = np.arange(start, stop, step), y = values, **line_kwargs)
        if ylim is not None: self.ylim(*ylim)
        return self

    def path(
        self,
        *data,
        x=None,
        y=None,
        s=None,
        c=None,
        linecolor: str | None = None,
        linewidth=None,
        line_alpha=None,
        marker=None,
        cmap=None,
        edgecolors=None,
        marker_linewidths=None,
        marker_alpha=None,
        label=None,
        front: Literal["line", "marker"] = "line",
        **kwargs,
    ):
        x, y = _prepare_data_for_plotting(*data, x=x, y=y, ensure_x=True)
        s = maybe_detach_cpu_recursive(s)
        c = maybe_detach_cpu_recursive(c)

        self.linechart(x,y, label=label, color=linecolor, alpha=line_alpha, linewidth=linewidth, zorder=1 if front == 'marker' else 2, **kwargs,)
        self.scatter(x, y, s=s, c=c, marker=marker, cmap=cmap, alpha=marker_alpha,linewidths=marker_linewidths,edgecolors=edgecolors,zorder=2 if front == 'marker' else 1, **kwargs)
        return self

    def contour(
        self,
        x,
        y,
        z,
        levels=10,
        step=300,
        cmap=None,
        mode: Literal["linear", "nearest", "clough", "rbf"] = "linear",
        xlim=None,
        ylim=None,
        zlim=None,
        alpha=None,
        norm = None,
        linewidths = None,
        linestyles = None,
    ):
        x = ensure_numpy_recursive(x)
        y = ensure_numpy_recursive(y)
        z = ensure_numpy_recursive(z)
        X, Y, Z = _get_grid(x=x, y=y, z=z, mode=mode, xlim=xlim, ylim=ylim, zlim=zlim, step=step)
        self.ax.contour(X, Y, Z, levels=levels, cmap=cmap, alpha = alpha, norm = norm, linewidths = linewidths, linestyles=linestyles)
        return self

    def contourf(self, x, y, z, levels=10, step = 300, cmap = None, mode:Literal["linear", "nearest", "clough", "rbf"] = 'linear', xlim = None, ylim = None, zlim = None, alpha = None):
        x = ensure_numpy_recursive(x)
        y = ensure_numpy_recursive(y)
        z = ensure_numpy_recursive(z)
        X, Y, Z = _get_grid(x=x, y=y, z=z, mode=mode, xlim=xlim, ylim=ylim, zlim=zlim, step=step)
        self.ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha = alpha)
        return self

    def pcolormesh(
        self,
        x,
        y,
        z,
        step=300,
        cmap=None,
        contour=False,
        contour_cmap="binary",
        contour_levels=10,
        contour_alpha=0.5,
        mode:Literal["linear", "nearest", "clough"]="linear",
        xlim=None,
        ylim=None,
        zlim=None,
        alpha=None,
        shading=None,
        norm = None,
        antialiased: bool = True,
    ):
        x = ensure_numpy_recursive(x)
        y = ensure_numpy_recursive(y)
        z = ensure_numpy_recursive(z)
        X, Y, Z = _get_grid(x=x, y=y, z=z, mode=mode, xlim=xlim, ylim=ylim, zlim=zlim, step=step)
        self.ax.pcolormesh(X, Y, Z, cmap=cmap, alpha = alpha, shading = shading, antialiased = antialiased, zorder=0, norm=norm)
        if contour: self.contour(x,y,z, levels=contour_levels, step=step, cmap=contour_cmap, mode=mode, alpha=contour_alpha, xlim=xlim, ylim=ylim, zlim=zlim, norm=norm)
        return self

    def quiver(
        self,
        x,
        y,
        u,
        v,
        c = None,
        angles: Literal['uv', 'xy'] = 'uv',
        pivot: Literal['tail', 'mid', 'middle', 'tip'] = 'tail',
        scale: float | None = None,
        scale_units: Literal['width', 'height', 'dots', 'inches', 'x', 'y', 'xy'] = 'width',
        units: Literal['width', 'height', 'dots', 'inches', 'x', 'y', 'xy'] = 'width',
        width: float | None = None,
        headwidth: float = 3,
        headlength: float = 5,
        headaxislength: float = 4.5,
        minshaft: float = 1,
        minlength: float = 1,
        color: Any = None,
    ):
        x = ensure_numpy_recursive(x)
        y = ensure_numpy_recursive(y)
        u = ensure_numpy_recursive(u)
        v = ensure_numpy_recursive(v)
        c = ensure_numpy_recursive(c)
        loc = locals().copy()
        del loc["self"], loc["x"], loc["y"], loc["u"], loc["v"], loc["c"]
        args = [i for i in (x,y,u,v,c) if i is not None]
        self.ax.quiver(*args,**loc)
        return self

    @torch.no_grad
    def _make_2d_function_mesh(
        self,
        f,
        xrange,
        yrange,
        num: int | None,
        step: int | None,
        lib: "Literal['np', 'numpy', 'torch', 'vmap', 'torch-elementwise'] | Any | None",
        chunk_size,
        enable_grad,
        dtype,
        device,
    ):
        xrange = tonumpy(xrange)
        yrange = tonumpy(yrange)

        if lib in ('np', 'numpy'): lib = np
        if lib == 'torch': lib = torch
        if lib is None:
            lib = np
            elementwise = True
        else:
            elementwise = False
        if lib == 'vmap':
            lib = torch
            vmap = True
        else:
            vmap = False
        if num is None: num = (xrange[1] - xrange[0]) / step
        x = lib.linspace(xrange[0], xrange[1], num) # type:ignore
        y = lib.linspace(yrange[0], yrange[1], num) # type:ignore
        X,Y = lib.meshgrid(x, y, indexing='ij') # grid of point # type:ignore
        try:
            X = X.astype(dtype)
            Y = Y.astype(dtype)
        except Exception:
            try:
                X = X.to(device = device, dtype=dtype)
                Y = Y.to(device = device, dtype=dtype)
            except Exception:
                print('cant move to dtype and device')

        with torch.enable_grad() if enable_grad else nullcontext():
            if vmap:
                Z = torch.vmap(f, in_dims = (0, 1), out_dims = (0, 1), chunk_size = chunk_size)(X, Y)
            else:
                if elementwise: Z = np.vectorize(f)(X, Y)
                else: Z = f(X, Y)
            return X, Y, Z

    @torch.no_grad
    def funcplot2d(
        self,
        f,
        xrange,
        yrange,
        num: int | None = 1000,
        step  = None,
        cmap = 'gray',
        norm = None,
        surface_alpha = 1.,
        levels = 12,
        contour_cmap = 'binary',
        contour_lw = 0.5,
        contour_alpha = 0.3,
        grid_alpha = 0.,
        grid_color = 'gray',
        grid_lw=0.5,
        lib: "Literal['np', 'numpy', 'torch', 'vmap', 'torch-elementwise'] | Any | None" = np,
        dtype=None,
        device=None,
        chunk_size = None,
    ):
        X, Y, Z = self._make_2d_function_mesh(f, xrange, yrange, num = num, step=step, lib=lib, chunk_size = chunk_size,dtype=dtype,device=device, enable_grad=False)
        X, Y, Z = maybe_detach_cpu_recursive(X),maybe_detach_cpu_recursive(Y),maybe_detach_cpu_recursive(Z)
        self.ax.pcolormesh(X, Y, Z, cmap=cmap, alpha = surface_alpha, norm = norm)
        if levels: self.ax.contour(X, Y, Z, levels=levels, cmap=contour_cmap, linewidths=contour_lw, alpha=contour_alpha, norm = norm)
        if grid_alpha > 0: self.grid(alpha=grid_alpha, lw=grid_lw, color = grid_color)

    # @torch.no_grad
    # def gradplot2d(
    #     self,
    #     f,
    #     xrange,
    #     yrange,
    #     num: int | None = 1000,
    #     step  = None,
    #     angles: Literal['uv', 'xy'] = 'uv',
    #     pivot: Literal['tail', 'mid', 'middle', 'tip'] = 'tail',
    #     scale: float | None = None,
    #     scale_units: Literal['width', 'height', 'dots', 'inches', 'x', 'y', 'xy'] = 'width',
    #     units: Literal['width', 'height', 'dots', 'inches', 'x', 'y', 'xy'] = 'width',
    #     width: float | None = None,
    #     headwidth: float = 3,
    #     headlength: float = 5,
    #     headaxislength: float = 4.5,
    #     minshaft: float = 1,
    #     minlength: float = 1,
    #     c: Any = None,
    #     color: Any = None,
    # ):
    #     X, Y, Z = self._make_2d_function_mesh(f, xrange, yrange, num = num, step=step, lib=torch, enable_grad=True)
    #     grad = torch.autograd.grad()
    #     self.ax.pcolormesh(X, Y, Z, cmap=cmap, alpha = surface_alpha,)
    #     if levels: self.ax.contour(X, Y, Z, levels=levels, cmap=contour_cmap, linewidths=contour_lw, alpha=contour_alpha)
    #     if grid_alpha > 0: self.grid(alpha=grid_alpha, lw=grid_lw, color = grid_color)


    def funcplot2d_quiver(
        self,
        f,
        xrange,
        yrange,
        num: int | None = 50,
        step  = None,
        lib: "Literal['np', 'numpy', 'torch'] | Any | None" = np,
        angles: Literal['uv', 'xy'] = 'uv',
        pivot: Literal['tail', 'mid', 'middle', 'tip'] = 'tail',
        scale: float | None = None,
        scale_units: Literal['width', 'height', 'dots', 'inches', 'x', 'y', 'xy'] = 'width',
        units: Literal['width', 'height', 'dots', 'inches', 'x', 'y', 'xy'] = 'width',
        width: float | None = None,
        headwidth: float = 3,
        headlength: float = 5,
        headaxislength: float = 4.5,
        minshaft: float = 1,
        minlength: float = 1,
        c: Any = None,
        color: Any = None,
    ):
        pass
        # X, Y, Z = self._make_2d_function_mesh(f, xrange, yrange, num = num, step=step, lib=lib,)
        # u, v = Z
        # return self.quiver(
        #     x=X,y=Y,u=u,v=v,angles=angles,pivot=pivot,scale=scale,scale_units=scale_units,units=units,width=width,headwidth=headwidth,headlength=headlength,headaxislength=headaxislength,minshaft=minshaft,minlength=minlength,c=c,color=color,
        # )

    def vline(self, x, ymin = None, ymax = None, **kwargs: Unpack[_K_Line2D]):
        if ymin is not None and ymax is not None:
            self.ax.axvline(x, ymin=ymin, ymax=ymax, **kwargs)
        else:
            self.ax.axline((x, 0), (x, 1), **kwargs)
        return self

    def hline(self, y, xmin = None, xmax = None, **kwargs: Unpack[_K_Line2D]):
        if xmin is not None and xmax is not None:
            self.ax.axhline(y, xmin=xmin, xmax=xmax, **kwargs)
        else:
            self.ax.axline((0, y), (1, y), **kwargs)
        return self

    def line(self, xy1: tuple[float, float] | Any,xy2: tuple[float, float] | None | Any = None, slope = None, **kwargs: Unpack[_K_Line2D]):
        xy1 = tuple(ensure_numpy_recursive(xy1).tolist())
        if xy2 is not None: xy2 = tuple(ensure_numpy_recursive(xy2).tolist())
        slope = maybe_ensure_pynumber(slope)
        self.ax.axline(xy1, xy2, slope=slope, **kwargs)
        return self

    def origin_lines(self, linewidth = 0.5, color='black', coords = (0,0), **kwargs: Unpack[_K_Line2D]):  # type:ignore
        k: dict[str,Any] = dict(kwargs)
        self.vline(coords[0], linewidth=linewidth, color=color, **k)
        return self.hline(coords[1], linewidth=linewidth, color=color, **k)

    def point(self, x, y, s=None, c=None, alpha=None, marker = None, **kwargs):
        self.ax.scatter(x, y, s=s, c=c, alpha=alpha, marker=marker, **kwargs)
        return self

    def axtitle(
        self,
        label: Any,
        loc: Literal["center", "left", "right"] | None = None,
        pad: float | None = None,
        **kwargs: Unpack[_K_Text],
    ):
        self.ax.set_title(label = str(label)[:10000], loc = loc, pad = pad, **kwargs)
        return self

    def figtitle(self, t: Any, **kwargs: Unpack[_K_Text]):
        self.figure.suptitle(str(t)[:10000], **kwargs)
        return self

    def figsize(self, w: float | tuple[float, float] = (6.4, 4.8), h: float | None = None, forward: bool = True):
        """Width, height"""
        self.figure.set_size_inches(w, h, forward)
        return self

    def xlabel(self, label: Any, **kwargs: Unpack[_K_Text]):
        self.ax.set_xlabel(str(label)[:10000], **kwargs)
        return self

    def ylabel(self, label: Any, **kwargs: Unpack[_K_Text]):
        self.ax.set_ylabel(str(label)[:10000], **kwargs)
        return self

    def axlabels(self, xlabel: Any, ylabel: Any, **kwargs: Unpack[_K_Text]):
        if xlabel is not None: self.xlabel(xlabel, **kwargs)
        if ylabel is not None: self.ylabel(ylabel, **kwargs)
        return self

    def legend(
        self,
        loc: Literal[
            "upper left",
            "upper right",
            "lower left",
            "lower right",
            "upper center",
            "lower center",
            "center left",
            "center right",
            "center",
            "best",
        ]
        | tuple[float, float] = "best",
        size: float | None = 6,
        edgecolor=None,
        linewidth: float | None = 3.0,
        frame_alpha=0.3,
        prop=None,
    ):
        if prop is None: prop = {}
        if size is not None and 'size' not in prop: prop['size'] = size

        leg = self.ax.legend(loc=loc, prop=prop, edgecolor=edgecolor,)
        leg.get_frame().set_alpha(frame_alpha)

        if linewidth is not None:
            for line in leg.get_lines():
                line.set_linewidth(linewidth)

        return self

    @method2method(Axes.set_xlim)
    def xlim(self, *args, **kwargs):
        self.ax.set_xlim(*args, **kwargs)
        return self

    @method2method(Axes.set_ylim)
    def ylim(self, *args, **kwargs):
        self.ax.set_ylim(*args, **kwargs)
        return self

    @method2method(Axes.axis)
    def axis(self, *args, **kwargs):
        self.ax.axis(*args, **kwargs)
        return self

    def ticks(
        self,
        xmajor=True,
        ymajor=True,
        xminor: int | None | Literal["auto"] = "auto",
        yminor: int | None | Literal["auto"] = "auto",
    ):
        if xmajor: self.ax.xaxis.set_major_locator(AutoLocator())
        if ymajor: self.ax.yaxis.set_major_locator(AutoLocator())
        if xminor is not None: self.ax.xaxis.set_minor_locator(AutoMinorLocator(xminor)) # type:ignore
        if yminor is not None: self.ax.yaxis.set_minor_locator(AutoMinorLocator(yminor)) # type:ignore
        return self

    def tick_params(
        self,
        axis: Literal["x", "y", "both"] = "both",
        which: Literal["major", "minor", "both"] = "major",
        reset: bool = False,
        direction: Literal['in', 'out', 'inout'] | None = None,
        length: float | None = None,
        width: float | None = None,
        color = None,
        pad: float | None = None,
        labelsize: float | _FontSizes | None = None,
        labelcolor: Any | None = None,
        labelfontfamily: str | None = None,
        colors: Any | None = None,
        zorder: float | None = None,
        bottom: bool | None = None, top: bool | None = None, left: bool | None = None, right: bool | None = None,
        labelbottom: bool | None = None, labeltop: bool | None = None, labelleft: bool | None = None, labelright: bool | None = None,
        labelrotation: float | None = None,
        grid_color: Any | None = None,
        grid_alpha: float | None = None,
        grid_linewidth: float | None = None,
        grid_linestyle: str | None = None,
    ):
        loc: dict[str, Any] = locals().copy()
        del loc["self"]
        loc = {k:v for k,v in loc.items() if v is not None}
        self.ax.tick_params(**loc)
        return self

    def set_axis_off(self):
        self.ax.set_axis_off()
        return self

    def axoff(self): return self.set_axis_off()

    def xscale(self, scale: str | Any, **kwargs):
        self.ax.set_xscale(scale, **kwargs)
        return self

    def yscale(self, scale: str | Any, **kwargs):
        self.ax.set_yscale(scale, **kwargs)
        return self

    def colorbar(self, location = None, orientation = None, fraction=0.15, shrink = 1., aspect = 20.):
        plt.colorbar(self.ax.collections[0], location=location,orientation=orientation, fraction=fraction, shrink=shrink, aspect=aspect)
        #ax.get_figure().colorbar(matplotlib.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax, location=location, orientation=orientation, fraction=fraction, shrink=shrink, aspect=aspect) # type:ignore
        return self

    def preset(
        self,
        preset: Literal['plot', 'image'] = 'plot',
        xlabel = None,
        ylabel = None,
        title = None,
        legend=False,
        xscale = None,
        yscale = None,
        xlim = None,
        ylim = None,
    ):
        if xlim is not None: self.xlim(xlim)
        if ylim is not None: self.ylim(ylim)
        if xscale is not None: self.xscale(xscale)
        if yscale is not None: self.yscale(yscale)

        if preset == 'plot':
            self.ticks().grid()

        elif preset == 'image':
            if xlabel is None and ylabel is None:
                self.axoff()

        self.axlabels(xlabel, ylabel)
        if title is not None: self.axtitle(title)
        if legend: self.legend()

        return self