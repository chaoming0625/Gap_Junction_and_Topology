# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import npbrain as nn
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage import transform

from core import *


def HSF_triangle_closed():
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'HSF_triangle_closed'
    loc_Stim.stim_sizes = [12, 12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'a': 10, 'lw': 3}
    Stimulus.show_stimulus(loc_Stim)


def HSF_triangle_open(position='left'):
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'HSF_triangle_open'
    loc_Stim.stim_sizes = [12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'a': 10, 'lw': 3, 'position': position}
    Stimulus.show_stimulus(loc_Stim)


def HSF_terminator_closed():
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'HSF_terminator_closed'
    loc_Stim.stim_sizes = [12, 12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'length': 10, 'lw': 3}
    Stimulus.show_stimulus(loc_Stim)


def HSF_terminator_open(position='left'):
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'HSF_terminator_open'
    loc_Stim.stim_sizes = [12, 12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'length': 10, 'lw': 3, 'position': position}
    Stimulus.show_stimulus(loc_Stim)


def HSF_rec_dots_closed():
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'HSF_rec_dots_closed'
    loc_Stim.stim_sizes = [12, 12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'length': 20, 'lw': 3, 'stride': 5}
    Stimulus.show_stimulus(loc_Stim)


def HSF_rectangle_closed():
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'HSF_rectangle_closed'
    loc_Stim.stim_sizes = [12, 12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'length': 10, 'lw': 3}
    Stimulus.show_stimulus(loc_Stim)


def HSF_rectangle_open(position='left'):
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'HSF_rectangle_open'
    loc_Stim.stim_sizes = [12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'length': 10, 'lw': 3, 'position': position}
    Stimulus.show_stimulus(loc_Stim)


def circle_smooth():
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'circle_smooth'
    loc_Stim.stim_sizes = [12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'inner_radius': 15, 'outer_radius': 25}
    Stimulus.show_stimulus(loc_Stim)


def one_hole_smooth(position='corner'):
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'one_hole_smooth'
    loc_Stim.stim_sizes = [12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'ring_inner_radius_s': 9,
                       'ring_inner_radius_b': 14,
                       'ring_outer_radius_s': 20,
                       'ring_outer_radius_b': 25,
                       'position': position}
    Stimulus.show_stimulus(loc_Stim)


def square_hole():
    loc_Stim = default_Stim.deepcopy()
    loc_Stim.name = 'square_hole'
    loc_Stim.stim_sizes = [12, 20, 12]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others = {'outer_height': 34,
                       'outer_width': 34,
                       'inner_height': 16,
                       'inner_width': 16}
    Stimulus.show_stimulus(loc_Stim)


def hole_with_breach():
    figsize = (4, 4)
    edge = False
    rotate = 'vertical'

    ##  important settings ##
    ## ------------------- ##
    angle = 60
    left_point, right_point = (71, 150), (130, 150)
    left_p, right_p = (81, 124), (119, 124)

    angle = 50
    left_point, right_point = (74, 153), (126, 153)
    left_p, right_p = (86, 127), (114, 127)

    angle = 30
    left_point, right_point = (85, 156), (116, 156)
    left_p, right_p = (90, 128), (110, 128)
    text_p = (96, 150)

    # angle = 30
    # left_point, right_point = (34.2, 61), (46.1, 61)
    # left_p, right_p = (36, 51), (44, 51)
    # text_p = (38.5, 60)

    angle = 40
    angle = 54
    left_point, right_point = (32., 61), (48.3, 61)
    left_p, right_p = (35.2, 51), (45, 51)
    text_p = (38.5, 60)

    # ------------------------
    # stimulus
    # ------------------------

    # height = width = 200
    # inner_radius = 30
    # outer_radius = 60

    height = width = 80
    inner_radius = 12
    outer_radius = 24

    center = [height // 2, width // 2]
    arr = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            dist = Stimulus._dist(i, j, (height / 2, width / 2))
            if dist < inner_radius:
                arr[i, j] = 0.
            elif inner_radius <= dist < outer_radius:
                arr[i, j] = 1.
            else:
                arr[i, j] = 0.
    for i in range(height):
        for j in range(width):
            if i < center[0]:
                ratio = np.abs(center[1] - j) / (center[0] - i)
                ang = np.arctan(ratio) * 360 / np.pi / 2
                if ang <= angle / 2:
                    arr[i, j] = 0.

    stim = 1 - arr
    if isinstance(rotate, int) and 360 > rotate > 0:
        stim = transform.rotate(stim, rotate)
    elif isinstance(rotate, str):
        if rotate == 'vertical':
            stim = stim[::-1, ]
    indexes = np.where(stim == 1.0)
    stim[indexes] = 0.9

    # ------------------------
    # visualization
    # ------------------------

    nn.visualize.mpl_style1(axes_edgecolor='black' if edge else 'white')
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax = fig.add_subplot(gs[:, :])

    # ### annotation arrow ###
    # connectionstyle = "arc3,rad=-0.3"
    # ax.annotate("", xy=left_p, xycoords='data',
    #             xytext=right_p, textcoords='data',
    #             arrowprops=dict(arrowstyle="-", color="0.4", shrinkA=5, shrinkB=5, lw=3,
    #                             patchA=None, patchB=None, connectionstyle=connectionstyle, ),
    #             )
    # font = {'family': 'lmodern', 'color': 'darkred', 'weight': 'normal', 'size': 25, }
    # ax.text(text_p[0], text_p[1], r'$\theta$', fontdict=font)
    #
    # ### two orange line ###
    # ax.plot([left_point[0], center[0]], [left_point[1], center[1]], 'orange', lw=3)
    # ax.plot([right_point[0], center[0]], [right_point[1], center[1]], 'orange', lw=3)
    ax.imshow(stim, cmap='gray', vmin=0., vmax=1.)
    # plt.plot(center[0], center[1], 'ro', markersize=6)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    plt.close(fig)


# HSF_triangle_closed()
# HSF_triangle_open('left')
# HSF_triangle_open('center')
# HSF_terminator_closed()
# HSF_terminator_open('left')
# HSF_terminator_open('right')
HSF_rec_dots_closed()
# HSF_rectangle_closed()
# HSF_rectangle_open('left')
# HSF_rectangle_open('right')
# circle_smooth()
# one_hole_smooth('center')
# one_hole_smooth('line_middle')
# one_hole_smooth('corner')
# square_hole()
# hole_with_breach()
