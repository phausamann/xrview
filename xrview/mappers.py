from copy import copy

import numpy as np
import pandas as pd


def _make_glyph_map(data, x, handler, method, x_arg, y_arg, glyph_kwargs):
    """ Make a glyph map. """

    data_list = []

    for v in data.data_vars:
        if x not in data[v].dims:
            raise ValueError(x + ' is not a dimension of ' + v)
        elif len(data[v].dims) == 1:
            data_list.append((v, None, None))
        elif len(data[v].dims) == 2:
            dim = [d for d in data[v].dims if d != x][0]
            for dval in data[dim].values:
                data_list.append((v, dim, dval))
        else:
            raise ValueError(v + ' has too many dimensions')

    glyph_map = pd.DataFrame(
        data_list, columns=['var', 'dim', 'dim_val'])

    glyph_map.loc[:, 'handler'] = handler
    glyph_map.loc[:, 'method'] = method
    glyph_map.loc[:, 'x_arg'] = x_arg
    glyph_map.loc[:, 'y_arg'] = y_arg
    glyph_map.loc[:, 'glyph_kwargs'] = \
        [copy(glyph_kwargs) for _ in range(glyph_map.shape[0])]

    return glyph_map


def map_figures_and_glyphs(
        data, x, handlers, glyphs, overlay, fig_kwargs, added_figures,
        added_overlays, added_overlay_figures, palette):
    """ Make the figure and glyphs map. """

    glyph_map = pd.concat([
        _make_glyph_map(data, x, handlers[0], g.method, g.x_arg, g.y_arg,
                        g.glyph_kwargs) for g in glyphs], ignore_index=True)

    if overlay == 'dims':
        figure_names = glyph_map['var']
    else:
        if len(np.unique(glyph_map['dim'])) > 1:
            raise ValueError(
                'Dimensions of all data variables must match')
        else:
            figure_names = glyph_map['dim_val']

    if overlay == 'data_vars':
        legend_col = 'var'
    else:
        legend_col = 'dim_val'

    # make figure map for base figures
    figure_map = pd.DataFrame(columns=['figure', 'fig_kwargs'])
    for f_idx, f_name in enumerate(np.unique(figure_names)):
        glyph_map.loc[figure_names == f_name, 'figure'] = f_idx
        figure_map = figure_map.append(
            {'figure': None, 'fig_kwargs': copy(fig_kwargs)},
            ignore_index=True)
        # TODO: control whether to put the title or not
        figure_map.iloc[-1]['fig_kwargs'].update({'title': str(f_name)})

    # add additional figures
    for added_idx, element in enumerate(added_figures):

        added_glyph_map = pd.concat([
            _make_glyph_map(element.data, x, element.handler, g.method,
                            g.x_arg, g.y_arg, g.glyph_kwargs)
            for g in element.glyphs], ignore_index=True)

        added_glyph_map.loc[:, 'figure'] = f_idx + added_idx + 1
        glyph_map = glyph_map.append(added_glyph_map, ignore_index=True)

        figure_map = figure_map.append(
            {'figure': None, 'fig_kwargs': copy(fig_kwargs)},
            ignore_index=True)
        figure_map.iloc[-1]['fig_kwargs'].update({'title': element.name})

    # add additional overlays
    for added_idx, element in enumerate(added_overlays):

        added_glyph_map = pd.concat([
            _make_glyph_map(element.data, x, element.handler, g.method,
                            g.x_arg, g.y_arg, g.glyph_kwargs)
            for g in element.glyphs], ignore_index=True)

        # find the indices of the figures to overlay
        if added_overlay_figures[added_idx] is None:
            figure_idx = figure_map.index.values
        elif isinstance(added_overlay_figures[added_idx], int):
            figure_idx =[added_overlay_figures[added_idx]]
        else:
            titles = np.array(
                [a['title'] for a in figure_map['fig_kwargs']])
            _, title_idx = np.unique(titles, return_index=True)
            titles = titles[np.sort(title_idx)]
            figure_idx = figure_map.index[
                titles == added_overlay_figures[added_idx]].values

        for f_idx in figure_idx:
            added_glyph_map.loc[:, 'figure'] = f_idx
            glyph_map = glyph_map.append(
                added_glyph_map, ignore_index=True)

    # update glyph_kwargs
    colormap = {v: palette[i]
                for i, v in enumerate(pd.unique(glyph_map[legend_col]))}

    for idx, g in glyph_map.iterrows():

        if g['dim_val'] is None:
            y_col = str(g['var'])
        else:
            y_col = '_'.join((str(g['var']), str(g['dim_val'])))

        glyph_kwargs = {g.x_arg: 'index', g.y_arg: y_col}
        if g[legend_col] is not None and isinstance(g.method, str):
            legend = str(g[legend_col])
            color = colormap[g[legend_col]]
            glyph_kwargs.update({'legend': legend, 'color': color})

        glyph_kwargs.update(glyph_map.loc[idx, 'glyph_kwargs'])
        glyph_map.loc[idx, 'glyph_kwargs'].update(glyph_kwargs)

    glyph_map.loc[:, 'figure'] = glyph_map.loc[:, 'figure'].astype(int)

    return figure_map, glyph_map
