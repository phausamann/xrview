""" ``xrview.palettes`` """

from bokeh.palettes import Paired12


RGB = Paired12[5::-2] + Paired12[-1:5:-2] + Paired12[4::-2] + Paired12[-2:4:-2]
