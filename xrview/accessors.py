import pandas as pd
import xarray as xr
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import gridplot
from bokeh.plotting import ColumnDataSource, figure
from bokeh.transform import CategoricalColorMapper, factor_cmap


def _infer_data(data, **kwargs):
    """"""
    orthogonal = {}
    parallel = {}
    dims = []

    allowed_params = {"x", "y", "hue", "col", "row"}
    assert not any(set(kwargs.keys()) - allowed_params)

    for param in kwargs.copy():
        if kwargs[param] == "data_vars":
            if isinstance(data, xr.DataArray):
                raise ValueError(
                    f"Cannot set {param}='data_vars' for DataArray."
                )
            parallel[param] = kwargs.pop(param)
        elif kwargs[param] in data.dims:
            orthogonal[param] = kwargs.pop(param)
            dims.append(orthogonal[param])

    for param, val in kwargs.items():
        if val in data.coords:
            if data[val].ndim != 1:
                raise ValueError(
                    f"'{param}' must point to a one-dimensional coordinate, "
                    f"'{val}' is {data[val].ndim}-dimensional."
                )
            if data[val].dims[0] in dims:
                parallel[param] = val
            else:
                orthogonal[param] = val
                dims.append(val)

    return orthogonal, parallel


def _dict_from_da(da, unstack=None):
    """"""
    df = da.drop_vars(
        [c for c in da.coords if c not in da.dims]
    ).to_dataframe()

    if unstack is not None:
        df = df.unstack(unstack)

    # convert column multi-index levels to str
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.set_levels(
            [l.astype(str) for l in df.columns.levels]
        )

    return ColumnDataSource.from_df(df)


def _to_cds(data, unstack=None):
    """"""
    source_dict = {}

    if hasattr(data, "data_vars"):
        for v in data.data_vars:
            source_dict.update(_dict_from_da(data[v], unstack))
    else:
        source_dict.update(_dict_from_da(data, unstack))

    source = ColumnDataSource(source_dict)

    for c in data.coords:
        if c not in data.dims and data[c].ndim == 1:
            source.add(data[c].values, c)

    return source


def _get_pretty_text_map(data):
    """"""
    variables = list(data.coords)
    if hasattr(data, "data_vars"):
        variables += list(data.data_vars)

    text_map = {}
    for name in variables:
        if "long_name" in data[name].attrs:
            text_map[name] = f"{data[name].attrs['long_name']}"
            if "units" in data[name].attrs:
                text_map[name] += f" [{data[name].attrs['units']}]"
        else:
            text_map[name] = name

    return text_map


def _get_discrete_colormap(n_vals):
    """"""
    if n_vals <= 10:
        return f"Category10_{n_vals}"
    elif n_vals <= 20:
        return f"Category20_{n_vals}"
    else:
        raise ValueError(
            "Cannot plot more than 20 different discrete hue values."
        )


def _infer_1d_metadata(data, x, y, hue):
    """"""
    if x is None:
        x = data.dims[0]

    if y is None:
        if hasattr(data, "name") and data.name is not None:
            y = data.name

    if hue is None:
        if data.ndim > 1:
            hue = data.dims[1]

    if hue is not None and hue in data.dims:
        hue = [f"{y}_{h}" for h in data[hue].values.astype(str)]

    return x, y, hue


def _create_figure(
    method,
    source,
    x,
    y,
    text_map,
    hue=None,
    hue_style="discrete",
    add_guide=None,
):
    """"""
    plot_kwargs = {}

    if x is None:
        raise ValueError(f"x must be one of {list(source.data.keys())}")
    if y is None:
        raise ValueError(f"y must be one of {list(source.data.keys())}")

    if hue is not None:
        if hue_style == "discrete":
            if isinstance(hue, str):
                hue_vals = pd.unique(source.data[hue])
                plot_kwargs["color"] = factor_cmap(
                    hue, _get_discrete_colormap(len(hue_vals)), hue_vals
                )
            else:
                mapper = CategoricalColorMapper(
                    palette=_get_discrete_colormap(len(hue)), factors=hue
                )
                cmap = {
                    key: value
                    for key, value in zip(mapper.factors, mapper.palette)
                }
        elif hue_style == "continuous":
            raise NotImplementedError("hue_style='continuous' tbd")
        elif hue_style is None:
            raise NotImplementedError("hue_style=None tbd")
        else:
            raise ValueError(
                "hue_style must be either None, 'discrete' or 'continuous'."
            )

    if add_guide or add_guide is None and hue is not None:
        if hue is None:
            raise ValueError("Cannot set add_guide when hue is None.")
        elif hue_style == "discrete":
            if isinstance(hue, str):
                plot_kwargs["legend_field"] = hue
        elif hue_style == "continuous":
            raise NotImplementedError("hue_style='continuous' tbd")
        elif hue_style is None:
            raise NotImplementedError("hue_style=None tbd")

    fig = figure()

    if x in text_map:
        fig.xaxis.axis_label = text_map[x]
    if y in text_map:
        fig.yaxis.axis_label = text_map[y]

    if isinstance(hue, list):
        for h in hue:
            getattr(fig, method)(
                x,
                h,
                source=source,
                legend_label=h,
                color=cmap[h],
                **plot_kwargs,
            )
    else:
        getattr(fig, method)(x, y, source=source, **plot_kwargs)

    return fig


def plot(
    data,
    method="scatter",
    x=None,
    y=None,
    hue=None,
    col=None,
    row=None,
    col_wrap=None,
    hue_style="discrete",
    add_guide=None,
    output=None,
):
    """"""
    text_map = _get_pretty_text_map(data)

    if col is None:
        source = _to_cds(data, hue)
        x, y, hue = _infer_1d_metadata(data, x, y, hue)
        layout = _create_figure(
            method, source, x, y, text_map, hue, hue_style, add_guide
        )

    elif col == "data_vars":
        fig_list = []

        for v in data.data_vars:
            if x is None:
                x_fig = v
                y_fig = y
                sharex = False
                sharey = True
            elif y is None:
                x_fig = x
                y_fig = v
                sharex = True
                sharey = False
            else:
                raise ValueError(
                    "Either x or y must be None when col='data_vars'."
                )

            source = _to_cds(data[v])
            fig = _create_figure(
                method,
                source,
                x_fig,
                y_fig,
                text_map,
                hue,
                hue_style,
                add_guide,
            )

            if sharex and len(fig_list):
                fig.x_range = fig_list[-1].x_range
            if sharey and len(fig_list):
                fig.y_range = fig_list[-1].y_range

            fig_list.append(fig)

        layout = gridplot([fig_list], ncols=col_wrap)

    if output == "notebook":
        output_notebook(hide_banner=True)
    elif str(output).endswith(".html"):
        output_file(output)
    elif output is not None:
        raise ValueError(
            "output must be 'notebook', an html file name or None."
        )

    show(layout)

    return layout


class BaseViewAccessor:
    """"""

    def __init__(self, xarray_obj):
        """"""
        self._obj = xarray_obj

    def scatter(self, **kwargs):
        """"""
        return plot(self._obj, "scatter", **kwargs)

    def line(self, **kwargs):
        """"""
        return plot(self._obj, "line", **kwargs)


@xr.register_dataset_accessor("view")
class DatasetViewAccessor(BaseViewAccessor):
    """"""


@xr.register_dataarray_accessor("view")
class DataArrayViewAccessor(BaseViewAccessor):
    """"""
