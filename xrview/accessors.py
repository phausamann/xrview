import pandas as pd
import xarray as xr
from bokeh.io import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.plotting import ColumnDataSource, figure
from bokeh.transform import factor_cmap


def _dict_from_da(da, unstack=None):
    """"""
    df = da.drop_vars(
        [c for c in da.coords if c not in da.dims]
    ).to_dataframe()

    if unstack is not None:
        return ColumnDataSource.from_df(df.unstack(unstack))
    else:
        return ColumnDataSource.from_df(df)


def _to_cds(data, unstack=None):
    """"""
    source_dict = {}

    if hasattr(data, "data_vars"):
        for v in data.data_vars:
            source_dict.update(_dict_from_da(data[v], unstack))
    else:
        source_dict.update(_dict_from_da(data, unstack))

    for c in data.coords:
        if c not in data.dims:
            source_dict[c] = data[c].values

    return ColumnDataSource(source_dict)


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
            hue_vals = pd.unique(source.data[hue])
            plot_kwargs["color"] = factor_cmap(
                hue, _get_discrete_colormap(len(hue_vals)), hue_vals
            )
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

    getattr(fig, method)(x, y, source=source, **plot_kwargs)

    return fig


def plot(
    data,
    method="scatter",
    x=None,
    y=None,
    col=None,
    col_wrap=None,
    hue=None,
    hue_style="discrete",
    add_guide=None,
):
    """"""
    text_map = _get_pretty_text_map(data)

    if col is None:
        source = _to_cds(data)
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

    output_notebook(hide_banner=True)
    show(layout)

    return layout


class BaseViewAccessor:
    """"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def scatter(self, **kwargs):
        return plot(self._obj, "scatter", **kwargs)


@xr.register_dataset_accessor("view")
class DatasetViewAccessor(BaseViewAccessor):
    """"""


@xr.register_dataarray_accessor("view")
class DataArrayViewAccessor(BaseViewAccessor):
    """"""
