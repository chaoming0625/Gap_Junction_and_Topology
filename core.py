# -*- coding: utf-8 -*-

import copy
import sys
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import npbrain as nn

nn.profile.set_backend('numba')

colors = ['b', 'r', 'g', 'k', 'c', 'm', 'y', ]

__all__ = ['Stimulus', 'default_RGC', 'default_SC', 'default_Stim',
           'default_Sync', 'default_Run', 'ParTools', 'RGC_SC_Net',
           'common_setting', 'template_run']


class Stimulus:
    @staticmethod
    def show_stimulus(stim_par, save_name=None, target='img'):
        img, _, stim = Stimulus.get(stim_par)
        if target == 'img':
            img = img.reshape((stim_par.height, stim_par.width))
        elif target == 'stim':
            img = stim.reshape((stim_par.height, stim_par.width))
            img_max, img_min = img.max(), img.min()
            img = 1 - (img - img_min) / img_max
        else:
            raise ValueError

        # stim = transform.rotate(stim, 180)
        plt.imshow(img[::-1, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=1.)

        # plt.pcolor(stim, cmap='gray')
        # for i in range(stim_par.height):
        #     plt.axhline(i, color='b')
        # for i in range(stim_par.width):
        #     plt.axvline(i, color='b')

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)

    @staticmethod
    def _transform_numpy_br2(arr, indexes, stim_sizes):
        assert np.ndim(arr) == 1

        return_arr = arr.copy()
        for i, index in enumerate(indexes):
            return_arr[index] = stim_sizes[i]
        return return_arr

    @staticmethod
    def _dist(i, j, center):
        return np.power((i - center[0]) ** 2 + (j - center[1]) ** 2, 0.5)

    @staticmethod
    def get(stim_par):
        try:
            func = getattr(Stimulus, '_{}'.format(stim_par.name))
        except Exception:
            raise ValueError('Unknown function.')
        arr, indexes, stim = func(stim_par.height, stim_par.width, stim_par.stim_sizes, stim_par.others)
        return arr, indexes, stim

    @staticmethod
    def _black(height, width, stim_sizes, **kwargs):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.stim_sizes = [15,]
        """
        arr = np.zeros((height, width), dtype=np.float32)
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == 0)[0], ]
        stim = arr * stim_sizes[0]
        return arr, indexes, stim

    @staticmethod
    def _figure_ground_for_net_display(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 60
            Par_Stim.width = 60
            Par_Stim.stim_sizes = [12, 16, 20]
            Par_stim.others = {'big_scale': 20,
                               'small_scale': 5,
                               'connected': False,
                               'connected_width': 2,
                               'small_pos': 'center'}
        """
        arr = np.zeros((height, width))
        offset = pars.get('offset', 5)

        # big heart
        center1 = (height // 2 + offset, width // 2)
        for row_i in range(height):
            for col_i in range(width):
                x = (col_i - center1[1]) / pars['big_scale']
                y = (row_i - center1[0]) / pars['big_scale']
                if (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 < 0:
                    arr[row_i, col_i] = 1.0

        # small heart
        if pars.get('small_pos', 'left') == 'left':
            center2 = (height // 4 - offset, width // 4)
        elif pars.get('small_pos', 'left') == 'center':
            center2 = (height // 4 - offset, width // 2)
        elif pars.get('small_pos', 'left') == 'right':
            center2 = (height // 4 - offset, width * 3 // 4)
        for row_i in range(height):
            for col_i in range(width):
                x = (col_i - center2[1]) / pars['small_scale']
                y = (row_i - center2[0]) / pars['small_scale']
                if (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 < 0:
                    # arr[row_i, col_i] = 0.5
                    arr[row_i, col_i] = 1.0

        arr = arr.reshape((height * width), )
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]
        stim = np.zeros_like(arr)
        stim[indexes[0]] = stim_sizes[0]
        stim[indexes[1]] = stim_sizes[1]
        stim[indexes[2]] = stim_sizes[2]
        return 1 - arr, indexes, stim

    @staticmethod
    def _figure_ground_connected_for_net_display(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 60
            Par_Stim.width = 60
            Par_Stim.stim_sizes = [15,]
            Par_stim.others = {'big_scale': 20,
                               'small_scale': 5,
                               'connected_width': 2}
        """
        arr = np.zeros((height, width))
        offset = pars.get('offset', 5)

        # big heart
        center1 = (height // 2 + offset, width // 2)
        for row_i in range(height):
            for col_i in range(width):
                x = (col_i - center1[1]) / pars['big_scale']
                y = (row_i - center1[0]) / pars['big_scale']
                if (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 < 0:
                    arr[row_i, col_i] = 1.0

        # small heart
        if pars.get('small_pos', 'left') == 'left':
            center2 = (height // 4 - offset, width // 4)
        elif pars.get('small_pos', 'left') == 'center':
            center2 = (height // 4 - offset, width // 2)
        elif pars.get('small_pos', 'left') == 'right':
            center2 = (height // 4 - offset, width * 3 // 4)
        for row_i in range(height):
            for col_i in range(width):
                x = (col_i - center2[1]) / pars['small_scale']
                y = (row_i - center2[0]) / pars['small_scale']
                if (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 < 0:
                    arr[row_i, col_i] = 1.0

        # slop connection
        cwidth = pars.get('connected_width', 2)
        if pars.get('small_pos', 'left') == 'left':
            for i in range(height // 4, height // 2):
                for j in range(cwidth):
                    arr[i, i + j] = 1.0
        elif pars.get('small_pos', 'left') == 'center':
            for i in range(height // 4, height // 2):
                for j in range(cwidth):
                    arr[i, height // 2 + j - cwidth // 2] = 1.0
        elif pars.get('small_pos', 'left') == 'right':
            for i in range(height // 4, height // 2):
                for j in range(pars.get('connected_width', 2)):
                    raise ValueError

        arr = arr.reshape((height * width), )
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        stim[indexes[0]] = stim_sizes[0]
        stim[indexes[1]] = stim_sizes[1]
        return 1 - arr, indexes, stim

    @staticmethod
    def _cd_object_background_connected(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.stim_sizes = [12, 20]
            Par_Stim.others['radius'] = 10
            Par_Stim.others['width'] = 10
        """
        arr = np.zeros((height, width), dtype=np.float32)

        # background
        arr[:height // 2, :] = 1.0

        # object
        radius = pars['radius']
        center = [height // 4 * 3, width // 2]
        for row_i in range(height // 2, height):
            for col_i in range(0, width):
                if Stimulus._dist(row_i, col_i, center) < radius:
                    arr[row_i, col_i] = 1.

        # line
        lwidth = pars['width']
        assert lwidth < 2 * radius
        for row_i in range(height // 2, height // 4 * 3):
            for col_i in range(width // 2 - lwidth // 2,
                               width // 2 + (lwidth - lwidth // 2)):
                arr[row_i, col_i] = 1.

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        stim[indexes[0]] = stim_sizes[0]
        stim[indexes[1]] = stim_sizes[1]
        return 1 - arr, indexes, stim

    @staticmethod
    def _cd_object_background_disconnected(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.stim_sizes = [12, 20]
            Par_Stim.others['radius'] = 10
            Par_Stim.others['num_split'] = 2
        """
        arr = np.zeros((height, width), dtype=np.float32)

        # background
        s = pars.get('num_split', 2)
        arr[:height // s, :] = 1.0

        # object
        radius = pars['radius']
        center = [height // 4 * 3, width // 2]
        for row_i in range(height // 2, height):
            for col_i in range(0, width):
                if Stimulus._dist(row_i, col_i, center) < radius:
                    arr[row_i, col_i] = 1.

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        stim[indexes[0]] = stim_sizes[0]
        stim[indexes[1]] = stim_sizes[1]
        return 1 - arr, indexes, stim

    @staticmethod
    def _cd_objects_connected(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.stim_sizes = [12, 20]
            Par_Stim.others['radius'] = 10
            Par_Stim.others['width'] = 10
        """
        arr = np.zeros((height, width), dtype=np.float32)

        # object 1
        radius = pars['radius']
        center = [height // 2, width // 4]
        for row_i in range(0, height):
            for col_i in range(0, width):
                if Stimulus._dist(row_i, col_i, center) < radius:
                    arr[row_i, col_i] = 1.0

        # object 2
        radius = pars['radius']
        center = [height // 2, width // 4 * 3]
        for row_i in range(0, height):
            for col_i in range(0, width):
                if Stimulus._dist(row_i, col_i, center) < radius:
                    arr[row_i, col_i] = 1.0

        # line
        lwidth = pars['width']
        for col_i in range(width // 4, width // 4 * 3):
            for row_i in range(height // 2 - lwidth // 2,
                               height // 2 + lwidth - lwidth // 2):
                arr[row_i, col_i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        stim[indexes[0]] = stim_sizes[0]
        stim[indexes[1]] = stim_sizes[1]
        return 1 - arr, indexes, stim

    @staticmethod
    def _cd_objects_disconnected(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.stim_sizes = [12, 20, 20]
            Par_Stim.others['radius'] = 10
        """
        arr = np.zeros((height, width), dtype=np.float32)

        # object 1
        radius = pars['radius']
        center = [height // 2, width // 4]
        for row_i in range(0, height):
            for col_i in range(0, width):
                if Stimulus._dist(row_i, col_i, center) < radius:
                    arr[row_i, col_i] = 0.5

        # object 2
        radius = pars['radius']
        center = [height // 2, width // 4 * 3]
        for row_i in range(0, height):
            for col_i in range(0, width):
                if Stimulus._dist(row_i, col_i, center) < radius:
                    arr[row_i, col_i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]
        arr[indexes[1]] = 1.0
        stim = np.zeros_like(arr)
        stim[indexes[0]] = stim_sizes[0]
        stim[indexes[1]] = stim_sizes[1]
        stim[indexes[2]] = stim_sizes[2]
        return 1 - arr, indexes, stim

    @staticmethod
    def _hd_hole_figure_ground_reverse(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'circle'
            Par_Stim.stim_sizes = [15, 20, 15]
            Par_Stim.others = {'inner_radius': 22, 'outer_radius': 25}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        indexes1 = []  # inside of the ring
        indexes2 = []  # on the ring
        indexes3 = []  # corner 1
        indexes4 = []  # corner 2
        indexes5 = []  # corner 3
        indexes6 = []  # corner 4
        for i in range(height):
            for j in range(width):
                dist = Stimulus._dist(i, j, (height / 2, width / 2))
                if dist < pars['inner_radius']:
                    indexes1.append(i * width + j)
                    arr[i, j] = 0.
                elif pars['inner_radius'] <= dist < pars['outer_radius']:
                    indexes2.append(i * width + j)
                    arr[i, j] = 1.
                else:
                    arr[i, j] = 0.

                    if i <= height // 2:
                        if j <= width // 2:
                            indexes3.append(i * width + j)
                        else:
                            indexes4.append(i * width + j)
                    else:
                        if j <= width // 2:
                            indexes5.append(i * width + j)
                        else:
                            indexes6.append(i * width + j)
        arr = arr.reshape((height * width,))
        indexes = [indexes1, indexes2, indexes3,
                   indexes4, indexes5, indexes6]
        stim = np.zeros_like(arr)
        for i in range(6):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _triangle(height, width, stim_sizes, pars):
        """Examples:
        Par_Stim.height = 100
        Par_Stim.width = 100
        Par_Stim.others = {'height' = 45, 'width' = 30}
        """
        arr = np.ones((height, width), dtype=np.float32)
        center = [height // 2, width // 2]

        tri_height = pars['height']
        tri_width = pars['width']

        point_1 = [center[0], center[1] - 2 * tri_height // 3]
        point_2 = [center[0] - tri_width // 2, center[1] + tri_height // 3]
        point_3 = [center[0] + tri_width // 2, center[1] + tri_height // 3]

        k1 = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
        k2 = (point_3[1] - point_1[1]) / (point_3[0] - point_1[0])

        for i in range(width):
            for j in range(point_1[1], point_2[1]):
                if i == point_1[0]:
                    arr[i, j] = 0.
                elif ((j - point_1[1]) / (i - point_1[0]) < k1) & ((j - point_1[1]) / (i - point_1[0]) < 0):
                    arr[i, j] = 0.
                elif ((j - point_1[1]) / (i - point_1[0]) > k2) & ((j - point_1[1]) / (i - point_1[0]) > 0):
                    arr[i, j] = 0.

        arr = arr.T
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [1., 0.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _triangle_hole(height, width, stim_sizes, pars):
        """Examples:
            Par_Stim.height = 60
            Par_Stim.width = 60
            Par_Stim.others = {'outer_height' = 45, 'outer_width' = 30,
                                    'inner_height' = 15, 'inner_width' = 10,}
        """
        arr = np.ones((height, width), dtype=np.float32)
        center = [height // 2, width // 2]

        outer_height = pars['outer_height']
        outer_width = pars['outer_width']

        inner_height = pars['inner_height']
        inner_width = pars['inner_width']

        outer_point1 = [center[0], center[1] - 2 * outer_height // 3]
        outer_point2 = [center[0] - outer_width // 2, center[1] + outer_height // 3]
        outer_point3 = [center[0] + outer_width // 2, center[1] + outer_height // 3]

        out_k1 = (outer_point2[1] - outer_point1[1]) / (outer_point2[0] - outer_point1[0])
        out_k2 = (outer_point3[1] - outer_point1[1]) / (outer_point3[0] - outer_point1[0])

        for i in range(width):
            for j in range(outer_point1[1], outer_point2[1]):
                if i == outer_point1[0]:
                    arr[i, j] = 0.
                elif ((j - outer_point1[1]) / (i - outer_point1[0]) < out_k1) & (
                        (j - outer_point1[1]) / (i - outer_point1[0]) < 0):
                    arr[i, j] = 0.
                elif ((j - outer_point1[1]) / (i - outer_point1[0]) > out_k2) & (
                        (j - outer_point1[1]) / (i - outer_point1[0]) > 0):
                    arr[i, j] = 0.

        inner_point1 = [center[0], center[1] - 2 * inner_height // 3]
        inner_point2 = [center[0] - inner_width // 2, center[1] + inner_height // 3]
        inner_point3 = [center[0] + inner_width // 2, center[1] + inner_height // 3]

        in_k1 = (inner_point2[1] - inner_point1[1]) / (inner_point2[0] - inner_point1[0])
        in_k2 = (inner_point3[1] - inner_point1[1]) / (inner_point3[0] - inner_point1[0])

        for i in range(width):
            for j in range(inner_point1[1], inner_point2[1]):
                if i == inner_point1[0]:
                    arr[i, j] = 0.5
                elif ((j - inner_point1[1]) / (i - inner_point1[0]) < in_k1) & (
                        (j - inner_point1[1]) / (i - inner_point1[0]) < 0):
                    arr[i, j] = 0.5
                elif ((j - inner_point1[1]) / (i - inner_point1[0]) > in_k2) & (
                        (j - inner_point1[1]) / (i - inner_point1[0]) > 0):
                    arr[i, j] = 0.5

        arr = arr.T
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [1., 0.5, 0.]]
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _sigmoid(height, width, stim_sizes, pars):
        """Example:
        """
        arr = np.ones((height, width), dtype=np.float32)
        outer_radius = pars['outer_radius']
        inner_radius = pars['inner_radius']

        center = [width // 2, height // 2]

        subcenter_1 = [width // 2, center[1] + outer_radius -
                       (outer_radius - inner_radius) // 2]

        subcenter_2 = [width // 2, center[1] - outer_radius +
                       (outer_radius - inner_radius) // 2]

        for i in range(center[0] - outer_radius, center[0] + 1):
            for j in range(subcenter_2[1], subcenter_1[1] + outer_radius + 1):
                if Stimulus._dist(i, j, (subcenter_1[0], subcenter_1[1])) < outer_radius:
                    arr[i, j] = 0.
                if Stimulus._dist(i, j, (subcenter_1[0], subcenter_1[1])) < inner_radius:
                    arr[i, j] = 1.

            for j in range(subcenter_2[1] - outer_radius, subcenter_2[1] - outer_radius // 2 + 1):
                if Stimulus._dist(i, j, (subcenter_2[0], subcenter_2[1])) < outer_radius:
                    arr[i, j] = 0.
                if Stimulus._dist(i, j, (subcenter_2[0], subcenter_2[1])) < inner_radius:
                    arr[i, j] = 1.

        for i in range(center[0], center[0] + outer_radius + 1):
            for j in range(subcenter_2[1] - outer_radius, subcenter_1[1]):
                if Stimulus._dist(i, j, (subcenter_2[0], subcenter_2[1])) < outer_radius:
                    arr[i, j] = 0.
                if Stimulus._dist(i, j, (subcenter_2[0], subcenter_2[1])) < inner_radius:
                    arr[i, j] = 1.

            for j in range(subcenter_1[1] + outer_radius // 2, subcenter_1[1] + outer_radius + 1):
                if Stimulus._dist(i, j, (subcenter_1[0], subcenter_1[1])) < outer_radius:
                    arr[i, j] = 0.
                if Stimulus._dist(i, j, (subcenter_1[0], subcenter_1[1])) < inner_radius:
                    arr[i, j] = 1.

        arr = arr.T
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _HSF_triangle_closed(height, width, stim_sizes, pars):
        """Example:
              Par_Stim.height = 50
              Par_Stim.width = 50
              Par_Stim.stim_sizes = [12, 12, 20]
              Par_Stim.others = {'a': 10, 'lw': 3}
        """
        assert pars['a'] > 1
        arr = np.zeros((height, width), dtype=np.float32)
        center = (width // 2, height // 2 - int(0.5 * pars['a']))  # (col_index, row_index)
        left = (center[0] - int(1.5 * pars['a']), center[1] - pars['a'])
        right = (center[0] + int(1.5 * pars['a']), center[1] - pars['a'])
        up = (center[0], center[1] + 2 * pars['a'])

        # center
        leqs = lambda row, col: (col - up[0]) / (left[0] - up[0]) - \
                                (row - up[1]) / (left[1] - up[1])
        reqs = lambda row, col: (col - up[0]) / (right[0] - up[0]) - \
                                (row - up[1]) / (right[1] - up[1])
        for col_i in range(left[0] + 2, right[0] + 1):
            for row_i in range(right[1] + 1, up[1]):
                if leqs(row_i, col_i) < 0 and reqs(row_i, col_i) < 0:
                    arr[row_i, col_i] = 0.5

        # bottom line
        for col_i in range(left[0], right[0] + 1):
            for row_pad in range(pars['lw'] - 1):
                arr[left[1] - row_pad, col_i] = 1.0

        # left line
        for i in range(pars['lw'] + 2):
            eqs = lambda row, col: (col - up[0]) / (left[0] - up[0]) - \
                                   (row - up[1] + i) / (left[1] - up[1])
            for col_i in range(left[0], up[0]):
                for row_i in range(left[1], up[1] + 1):
                    if eqs(row_i, col_i) >= 0:
                        arr[row_i, col_i,] = 1.0
                        break

        # right line
        for i in range(pars['lw'] + 2):
            eqs = lambda row, col: (col - up[0]) / (right[0] - up[0]) - \
                                   (row - up[1] + i) / (right[1] - up[1])
            for col_i in range(right[0], up[0] - 1, -1):
                for row_i in range(right[1], up[1] + 1):
                    if eqs(row_i, col_i) >= 0:
                        arr[row_i, col_i,] = 1.0
                        break

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]
        arr[indexes[1]] = 0.
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _HSF_triangle_open(height, width, stim_sizes, pars):
        """Example:
              Par_Stim.height = 50
              Par_Stim.width = 50
              Par_Stim.stim_sizes = [12, 20]
              Par_Stim.others = {'a': 10, 'lw': 3, 'position': 'left' or 'center'}
        """
        assert pars['a'] > 1
        arr = np.zeros((height, width), dtype=np.float32)
        center = (width // 2, height // 2)  # (col_index, row_index)
        left = (center[0] - pars['a'], center[1])
        right = (center[0] + pars['a'], center[1])

        # bottom line
        for col_i in range(left[0], right[0] + 1):
            for row_pad in range(pars['lw'] - 1):
                arr[left[1] - row_pad, col_i] = 1.0

        # left line
        for i in range(pars['a']):
            for j in range(pars['lw']):
                arr[left[1] + i + j, left[0] + i] = 1.0

        # right line
        if pars['position'] == 'left':
            l_s = left[0]
        elif pars['position'] == 'center':
            l_s = center[0]
        else:
            raise ValueError
        for i in range(pars['a']):
            for j in range(pars['lw']):
                arr[left[1] - pars['lw'] + 2 - i - j, l_s + i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _HSF_terminator_closed(height, width, stim_sizes, pars):
        """Example:
              Par_Stim.height = 50
              Par_Stim.width = 50
              Par_Stim.stim_sizes = [12, 12, 20]
              Par_Stim.others = {'length': 10, 'lw': 3}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        w_center = width // 2
        h_center = height // 2

        # center
        row_i1 = h_center - pars['length'] + pars['lw']
        row_i2 = h_center
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for row_i in range(row_i1, row_i2):
                arr[row_i, col_i] = 0.5

        # row 1
        row_i = h_center - pars['length']
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # row 2
        row_i = h_center
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # row 3
        row_i = h_center + pars['length']
        for col_i in range(w_center - pars['length'], w_center + pars['length'] + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # col 1
        col_i = w_center - pars['length']
        for row_i in range(h_center - pars['length'], h_center + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # col 2
        col_i = w_center + pars['length']
        for row_i in range(h_center - pars['length'], h_center + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]
        arr[indexes[1]] = 0
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _HSF_terminator_open(height, width, stim_sizes, pars):
        """Example:
              Par_Stim.height = 50
              Par_Stim.width = 50
              Par_Stim.stim_sizes = [12, 20]
              Par_Stim.others = {'length': 10, 'lw': 3, 'position': 'left' or 'right'}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        w_center = width // 2
        h_center = height // 2

        # row 1
        row_i = height // 2 - pars['length']
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # row 2
        row_i = height // 2
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

                # row 3
        row_i = height // 2 + pars['length']
        for col_i in range(w_center - pars['length'], w_center + pars['length'] + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # col 1
        if pars['position'] == 'left':
            col_i = w_center - pars['length']
        elif pars['position'] == 'right':
            col_i = w_center + pars['length']
        else:
            raise ValueError
        for row_i in range(h_center, h_center + pars['length'] + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # col 2
        col_i = w_center + pars['length']
        for row_i in range(h_center - pars['length'], h_center + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _HSF_rec_dots_closed(height, width, stim_sizes, pars):
        """Example:
              Par_Stim.height = 50
              Par_Stim.width = 50
              Par_Stim.stim_sizes = [12, 12, 20]
              Par_Stim.others = {'length': 20, 'lw': 2, 'stride': 4}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        w_center = width // 2
        h_center = height // 2 + pars['length'] // 2

        # center
        row_i1 = h_center - pars['length'] + pars['lw']
        row_i2 = h_center
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for row_i in range(row_i1, row_i2):
                arr[row_i, col_i] = 0.5

        # row 1
        row_i = h_center - pars['length']
        for col_i in range(w_center - pars['length'], w_center + pars['length'], pars['stride']):
            for i in range(pars['lw']):
                for j in range(pars['lw']):
                    arr[row_i + i, col_i + j] = 1.0

        # row 2
        row_i = h_center
        for col_i in range(w_center - pars['length'], w_center + pars['length'], pars['stride']):
            for i in range(pars['lw']):
                for j in range(pars['lw']):
                    arr[row_i + i, col_i + j] = 1.0

        # col 1
        col_i = w_center - pars['length']
        for row_i in range(h_center - pars['length'], h_center + pars['lw'], pars['stride']):
            for i in range(pars['lw']):
                for j in range(pars['lw']):
                    arr[row_i + j, col_i + i] = 1.0

        # col 2
        col_i = w_center + pars['length']
        for row_i in range(h_center - pars['length'], h_center + pars['lw'], pars['stride']):
            for i in range(pars['lw']):
                for j in range(pars['lw']):
                    arr[row_i + j, col_i + i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]
        arr[indexes[1]] = 0
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _HSF_rectangle_closed(height, width, stim_sizes, pars):
        """Example:
              Par_Stim.height = 50
              Par_Stim.width = 50
              Par_Stim.stim_sizes = [12, 12, 20]
              Par_Stim.others = {'length': 10, 'lw': 3}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        w_center = width // 2
        h_center = height // 2 + pars['length'] // 2

        # center
        row_i1 = h_center - pars['length'] + pars['lw']
        row_i2 = h_center
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for row_i in range(row_i1, row_i2):
                arr[row_i, col_i] = 0.5

        # row 1
        row_i = h_center - pars['length']
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # row 2
        row_i = h_center
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # col 1
        col_i = w_center - pars['length']
        for row_i in range(h_center - pars['length'], h_center + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # col 2
        col_i = w_center + pars['length']
        for row_i in range(h_center - pars['length'], h_center + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]
        arr[indexes[1]] = 0
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _HSF_rectangle_open(height, width, stim_sizes, pars):
        """Example:
              Par_Stim.height = 50
              Par_Stim.width = 50
              Par_Stim.stim_sizes = [12, 20]
              Par_Stim.others = {'length': 10, 'lw': 3, 'position': 'left' or 'right'}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        w_center = width // 2
        h_center = height // 2

        # row 1
        row_i = height // 2 - pars['length']
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # row 2
        row_i = height // 2
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # col 1
        if pars['position'] == 'left':
            col_i = w_center - pars['length']
        elif pars['position'] == 'right':
            col_i = w_center + pars['length']
        else:
            raise ValueError
        for row_i in range(h_center, h_center + pars['length'] + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # col 2
        col_i = w_center + pars['length']
        for row_i in range(h_center - pars['length'], h_center + pars['lw']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _cross(height, width, stim_sizes, pars):
        """Example:
              Par_Stim.height = 50
              Par_Stim.width = 50
              Par_Stim.stim_sizes = [12, 20]
              Par_Stim.others = {'length': 10, 'lw': 5}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        w_center = width // 2
        h_center = height // 2

        # row
        row_i = height // 2 - pars['lw'] // 2
        for col_i in range(w_center - pars['length'], w_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i + i, col_i] = 1.0

        # col
        col_i = w_center - pars['lw'] // 2
        for row_i in range(h_center - pars['length'], h_center + pars['length']):
            for i in range(pars['lw']):
                arr[row_i, col_i + i] = 1.0

        # final
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _square(height, width, stim_sizes, pars):
        """Examples:
            Par_Stim.height = 60
            Par_Stim.width = 60
            Par_Stim.others = {'height': 30, 'width': 30}
        """
        arr = np.ones((height, width), dtype=np.float32)
        center = [height // 2, width // 2]

        sq_height = pars['height']
        half_height = sq_height // 2

        sq_width = pars['width']
        half_width = sq_width // 2

        for row in range(center[0] - half_height, height - (center[0] - half_height)):
            for col in range(center[1] - half_width, width - (center[1] - half_width)):
                arr[row, col] = 0.

        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [1., 0.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return arr, indexes, stim

    @staticmethod
    def _square_hole(height, width, stim_sizes, pars):
        """Examples:
            Par_Stim.height = 60
            Par_Stim.width = 60
            Par_Stim.stim_sizes = [12, 12, 20]
            Par_Stim.others = {'outer_height': 30, 'outer_width': 30,
                                  'inner_height': 16, 'inner_width': 16, }
        """
        arr = np.ones((height, width), dtype=np.float32)
        center = [height // 2, width // 2]

        for row in range(center[0] - int(pars['outer_height'] / 2),
                         height - (center[0] - int(pars['outer_height'] / 2))):
            for col in range(center[1] - int(pars['outer_width'] / 2),
                             width - (center[1] - int(pars['outer_width'] / 2))):
                arr[row, col] = .5
        for row in range(center[0] - int(pars['inner_height'] / 2),
                         height - (center[0] - int(pars['inner_height'] / 2))):
            for col in range(center[1] - int(pars['inner_width'] / 2),
                             width - (center[1] - int(pars['inner_width'] / 2))):
                arr[row, col] = 0.

        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]

        arr[indexes[0]] = 1.
        arr[indexes[1]] = 0.
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return arr, indexes, stim

    @staticmethod
    def _circle(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'circle'
            Par_Stim.stim_sizes = [10, 15]
            Par_Stim.others = {'radius': 15, }
        """
        arr = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                if Stimulus._dist(i, j, (height / 2, width / 2)) < pars['radius']:
                    arr[i, j] = 1.
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _circle_smooth(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'circle'
            Par_Stim.stim_sizes = [10, 15]
            Par_Stim.others = {'inner_radius': 15, 'outer_radius': 25}
        """
        arr = np.ones((height, width), dtype=np.float32)
        indices1 = []
        indices2 = []
        indices3 = []
        for i in range(height):
            for j in range(width):
                dist = Stimulus._dist(i, j, (height / 2, width / 2))
                idx = i * width + j
                if dist < pars['inner_radius']:
                    arr[i, j] = 0.0
                    indices1.append(idx)
                elif dist < pars['outer_radius']:
                    arr[i, j] = (dist - pars['inner_radius']) / \
                                (pars['outer_radius'] - pars['inner_radius'])
                    indices2.append(idx)
                else:
                    arr[i, j] = 1.0
                    indices3.append(idx)
        # image
        arr = arr.reshape((height * width,))
        # index
        indexes = [indices1, indices2, indices3]
        # stimulus
        stim_sizes = stim_sizes
        stim = (1 - arr) * (stim_sizes[1] - stim_sizes[0]) + stim_sizes[0]
        return arr, indexes, stim

    @staticmethod
    def _one_hole(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'circle'
            Par_Stim.stim_sizes = [15, 20, 15]
            Par_Stim.others = {'inner_radius': 15, 'outer_radius': 20, 'position': 'corner' or 'line_middle'}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        indexes1 = []
        indexes2 = []
        indexes3 = []
        position = pars.get('position', 'center')
        if position == 'center':
            center = (height / 2, width / 2)
        elif position == 'corner':
            center = (pars['outer_radius'] - 1, pars['outer_radius'] - 1)
        elif position == 'line_middle':
            center = (pars['outer_radius'] - 1, width / 2)
        else:
            raise ValueError

        for i in range(height):
            for j in range(width):
                dist = Stimulus._dist(i, j, center)
                if dist < pars['inner_radius']:
                    indexes1.append(i * width + j)
                    arr[i, j] = 0.
                elif pars['inner_radius'] <= dist < pars['outer_radius']:
                    indexes2.append(i * width + j)
                    arr[i, j] = 1.
                else:
                    indexes3.append(i * width + j)
                    arr[i, j] = 0.
        arr = arr.reshape((height * width,))
        indexes = [indexes1, indexes2, indexes3]
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _one_hole_smooth(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 80
            Par_Stim.width = 80
            Par_Stim.name = 'one_hole_smooth'
            Par_Stim.stim_sizes = [15, 20]
            Par_Stim.others = {'ring_inner_radius_s': 9,
                                'ring_inner_radius_b': 14,
                                'ring_outer_radius_s': 20,
                                'ring_outer_radius_b': 25,
                                'position': 'corner' or 'line_middle'}
        """
        arr = np.zeros((height, width), dtype=np.float32)
        position = pars.get('position', 'center')
        if position == 'center':
            center = (height / 2, width / 2)
        elif position == 'corner':
            center = (pars['ring_outer_radius_b'] - 1,
                      pars['ring_outer_radius_b'] - 1)
        elif position == 'line_middle':
            center = (pars['ring_outer_radius_b'] - 1,
                      width / 2)
        else:
            raise ValueError

        indexes1 = []
        indexes2 = []
        indexes3 = []
        for i in range(height):
            for j in range(width):
                dist = Stimulus._dist(i, j, center)
                if dist < pars['ring_inner_radius_s']:
                    arr[i, j] = 0.
                    indexes1.append(i * width + j)
                elif dist < pars['ring_inner_radius_b']:
                    arr[i, j] = (dist - pars['ring_inner_radius_s']) / \
                                (pars['ring_inner_radius_b'] -
                                 pars['ring_inner_radius_s'])
                    indexes2.append(i * width + j)
                elif dist < pars['ring_outer_radius_s']:
                    arr[i, j] = 1.
                    indexes2.append(i * width + j)
                elif dist < pars['ring_outer_radius_b']:
                    arr[i, j] = (pars['ring_outer_radius_b'] - dist) / \
                                (pars['ring_outer_radius_b'] -
                                 pars['ring_outer_radius_s'])
                    indexes2.append(i * width + j)
                else:
                    arr[i, j] = 0.
                    indexes3.append(i * width + j)
        arr = arr.reshape((height * width,))
        indexes = [indexes1, indexes2, indexes3]
        stim = arr * (stim_sizes[1] - stim_sizes[0]) + stim_sizes[0]
        return 1 - arr, indexes, stim

    @staticmethod
    def _two_holes(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'circle'
            Par_Stim.stim_sizes = [15, 15, 15, 20]
            Par_Stim.others = {'inner_radius': 8, 'outer_radius': 24}
        """
        inner_radius = pars['inner_radius']
        outer_radius = pars['outer_radius']
        assert inner_radius * 2 <= outer_radius
        center0 = (height // 2, width // 2)
        center1 = (height // 2 - outer_radius // 2, width // 2)
        center2 = (height // 2 + outer_radius // 2, width // 2)

        arr = np.zeros((height, width), dtype=np.float32)
        indexes1 = []  # background
        indexes2 = []  # hole 1
        indexes3 = []  # hole 2
        indexes4 = []  # black object
        for i in range(height):
            for j in range(width):
                dist0 = Stimulus._dist(i, j, center0)
                dist1 = Stimulus._dist(i, j, center1)
                dist2 = Stimulus._dist(i, j, center2)
                if dist1 < inner_radius:
                    indexes2.append(i * width + j)
                    arr[i, j] = 0.
                elif dist2 < inner_radius:
                    indexes3.append(i * width + j)
                    arr[i, j] = 0.
                elif dist0 < outer_radius:
                    indexes4.append(i * width + j)
                    arr[i, j] = 1.
                else:
                    indexes1.append(i * width + j)
                    arr[i, j] = 0.
        arr = arr.reshape((height * width,))
        indexes = [indexes1, indexes2, indexes3, indexes4]
        stim = np.zeros_like(arr)
        for i in range(4):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _four_holes(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'circle'
            Par_Stim.stim_sizes = [15, 15, 15, 15, 15, 20]
            Par_Stim.others = {'inner_radius': 8, 'outer_radius': 24}
        """
        inner_radius = pars['inner_radius']
        outer_radius = pars['outer_radius']
        assert inner_radius * 2 <= outer_radius
        center0 = (height // 2, width // 2)
        center1 = (height // 2 - outer_radius // 2, width // 2)
        center2 = (height // 2 + outer_radius // 2, width // 2)
        center3 = (height // 2, width // 2 + outer_radius // 2)
        center4 = (height // 2, width // 2 - outer_radius // 2)

        arr = np.zeros((height, width), dtype=np.float32)
        indexes1 = []  # background
        indexes2 = []  # hole 1
        indexes3 = []  # hole 2
        indexes4 = []  # hole 3
        indexes5 = []  # hole 4
        indexes6 = []  # black object
        for i in range(height):
            for j in range(width):
                dist0 = Stimulus._dist(i, j, center0)
                dist1 = Stimulus._dist(i, j, center1)
                dist2 = Stimulus._dist(i, j, center2)
                dist3 = Stimulus._dist(i, j, center3)
                dist4 = Stimulus._dist(i, j, center4)
                if dist1 < inner_radius:
                    indexes2.append(i * width + j)
                    arr[i, j] = 0.
                elif dist2 < inner_radius:
                    indexes3.append(i * width + j)
                    arr[i, j] = 0.
                elif dist3 < inner_radius:
                    indexes4.append(i * width + j)
                    arr[i, j] = 0.
                elif dist4 < inner_radius:
                    indexes5.append(i * width + j)
                    arr[i, j] = 0.
                elif dist0 < outer_radius:
                    indexes6.append(i * width + j)
                    arr[i, j] = 1.
                else:
                    indexes1.append(i * width + j)
                    arr[i, j] = 0.
        arr = arr.reshape((height * width,))
        indexes = [indexes1, indexes2, indexes3, indexes4, indexes5, indexes6]
        stim = np.zeros_like(arr)
        for i in range(6):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _hole_loophole(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'circle'
            Par_Stim.stim_sizes = [15, 20, 15]
            Par_Stim.others = {'inner_radius': 15,
                                  'outer_radius': 20,
                                  'angle': 60}
        """
        center = [height // 2, width // 2]
        arr = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                dist = Stimulus._dist(i, j, (height / 2, width / 2))
                if dist < pars['inner_radius']:
                    arr[i, j] = 0.
                elif pars['inner_radius'] <= dist < pars['outer_radius']:
                    arr[i, j] = 1.
                else:
                    arr[i, j] = 0.
        for i in range(height):
            for j in range(width):
                if i < center[0]:
                    ratio = np.abs(center[1] - j) / (center[0] - i)
                    angle = np.arctan(ratio) * 360 / np.pi / 2
                    if angle <= pars['angle'] / 2:
                        arr[i, j] = 0.

        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return 1 - arr, indexes, stim

    @staticmethod
    def _two_splits(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'two_splits'
            Par_Stim.stim_sizes = [15, 20]
            Par_Stim.others = {'split_by': 'row',}
        """
        split_by = pars.get('split_by', 'row')
        assert split_by in ['row', 'column']

        arr = np.zeros((height, width), dtype=np.float32)

        if split_by == 'row':
            index = int(height // 2)
            arr[:index, :] = 1.
        if split_by == 'column':
            index = int(width // 2)
            arr[:, :index] = 1.
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 1.]]
        stim = np.zeros_like(arr)
        for i in range(2):
            stim[indexes[i]] = stim_sizes[i]
        return arr, indexes, stim

    @staticmethod
    def _three_splits(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'three_splits'
            Par_Stim.stim_sizes = [15, 20, 15]
            # Par_Stim.stim_sizes = [15, 20, 25]
            Par_Stim.others = {'split_by': 'row',}
        """
        split_by = pars.get('split_by', 'row')
        assert split_by in ['row', 'column']

        arr = np.zeros((height, width), dtype=np.float32)

        if split_by == 'row':
            index = int(height // 3)
            arr[:index, :] = 0.
            arr[index:index * 2, :] = 0.5
            arr[index * 2:, :] = 1.
        if split_by == 'column':
            index = int(width // 3)
            arr[:, :index] = 0.
            arr[:, index:index * 2] = 0.5
            arr[:, index * 2:] = 1.
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return arr, indexes, stim

    @staticmethod
    def _three_splits_two_values(height, width, stim_sizes, pars):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.name = 'three_splits_two_values'
            Par_Stim.stim_sizes = [15, 20, 15]
            # Par_Stim.stim_sizes = [15, 20, 25]
            Par_Stim.others = {'split_by': 'row',}
        """
        split_by = pars.get('split_by', 'row')
        assert split_by in ['row', 'column']

        arr = np.zeros((height, width), dtype=np.float32)

        if split_by == 'row':
            index = int(height // 3)
            arr[:index, :] = 0.
            arr[index:index * 2, :] = 0.5
            arr[index * 2:, :] = 1.
        if split_by == 'column':
            index = int(width // 3)
            arr[:, :index] = 0.
            arr[:, index:index * 2] = 0.5
            arr[:, index * 2:] = 1.
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == id_)[0] for id_ in [0., 0.5, 1.]]
        stim = np.zeros_like(arr)
        for i in range(3):
            stim[indexes[i]] = stim_sizes[i]
        return arr, indexes, stim


default_RGC = nn.Dict(
    seed=-1,
    # 1. dynamic parameters
    V_reset=0,
    V_th=10,
    V_initial='reset',
    tau=5,
    tau_refractory=3.5,
    noise_sigma=0.5,
    noise_correlated_with_input=False,

    # 2. local gap junction
    gj_w=0.05,
    gj_spikelet=0.15,
    gj_conn='grid_eight'
)

default_SC = nn.Dict(
    # 1. dynamic parameters
    V_reset=0,
    V_th=10,
    V_initial='random',
    tau=5,
    tau_refractory=0.5,
    noise_mean=1.0,
    noise_sigma=.1,

    # 2. parameters of connections between RGCs and RONs
    R2N_delay=0.1,
    R2N_current=1.0,
)

default_Stim = nn.Dict(
    height=60,
    width=60,
    name='two_splits',
    stim_sizes=(10, 15),
    others={}
)

default_Sync = nn.Dict(
    value=None,
    t_start=0,
    t_end=20,
    bin=0.5
)

default_Run = nn.Dict(
    repeat=1,  # the repeat number of a single model
    dt=0.01,
    duration=20,
    print_params=True,
    save_path=None,
    title_format='ns={:.2f},stim={},gjw={:.2f}',
    title_vars=['IF-noise_sigma', 'Stim-stim_sizes', 'IF-gj_w'],
    others={}
)


class ParTools:
    @staticmethod
    def show_vars(obj, name=None):
        if name is not None:
            print('-' * 30)
            print(name)
        print('-' * 30)
        for k in obj.__dir__():
            if not k.startswith('__'):
                print('{}={}'.format(k, getattr(obj, k)))
        print()

    @staticmethod
    def search_var(all_pars, name):
        var_splits = name.split('-')
        par = all_pars[var_splits[0]]
        for val in var_splits[1:]:
            if isinstance(par, dict):
                par = par[val]
            else:
                par = getattr(par, val)

        if isinstance(par, dict):
            raise ValueError("par={}, name={}".format(par, name))
        elif isinstance(par, (list, tuple)):
            return '-'.join([str(a) for a in par])
        else:
            return par

    @staticmethod
    def cls2dict(cls):
        return {k: copy.deepcopy(getattr(cls, k)) for k in cls.__dir__() if not k.startswith('__')}


class RGC_SC_Net(nn.Network):
    def __init__(self, par_RGC, par_RUN, par_STIM, par_SC, par_SYNC=None):
        self.par_RUN = par_RUN
        if par_RGC.seed <= 0:
            par_RGC.seed = np.random.randint(10, 100000)
        self.par_RGC = par_RGC
        self.par_SC = par_SC
        self.par_STIM = par_STIM
        self.par_SYNC = par_SYNC

        # show parameters
        if par_RUN.print_params:
            ParTools.show_vars(par_RUN, 'Run Params:')
            ParTools.show_vars(par_RGC, 'RGC Params:')
            ParTools.show_vars(par_SC, 'RON Params:')
            ParTools.show_vars(par_STIM, 'Stim Params:')
            if par_SYNC is not None:
                ParTools.show_vars(par_SYNC, 'Sync Params:')

        # stimulus definition
        self.stim_img, self.stim_indexes, self.stim_values = Stimulus.get(par_STIM)

        # model definition
        nn.profile.set_dt(par_RUN.dt)

        noise = par_RGC.noise_sigma * self.stim_values \
            if par_RGC.noise_correlated_with_input else par_RGC.noise_sigma
        rgc = nn.LIF((par_STIM.height, par_STIM.width),
                     ref=par_RGC.tau_refractory,
                     Vr=par_RGC.V_reset,
                     Vth=par_RGC.V_th,
                     tau=par_RGC.tau,
                     noise=noise)
        self._initialize_potential(par_RGC, rgc)
        if par_RGC.gj_conn in ['grid_eight', 'grid8']:
            conn = nn.connect.grid_eight(par_STIM.height, par_STIM.width, include_self=False)
        elif par_RGC.gj_conn in ['grid_four', 'grid4']:
            conn = nn.connect.grid_four(par_STIM.height, par_STIM.width, include_self=False)
        elif par_RGC.gj_conn in ['grid24']:
            conn = nn.connect.grid_N(par_STIM.height, par_STIM.width, N=2, include_self=False)
        else:
            raise ValueError()
        gj = nn.GapJunction_LIF(rgc, rgc, weights=par_RGC.gj_w, connection=conn, k_spikelet=par_RGC.gj_spikelet)
        mon_rgc = nn.StateMonitor(rgc, ['V', 'spike'])
        self.rgc = rgc
        self.gj = gj
        self.mon_rgc = mon_rgc

        sc = nn.LIF(1,
                    ref=par_SC.tau_refractory,
                    Vr=par_SC.V_reset,
                    Vth=par_SC.V_th,
                    tau=par_SC.tau,
                    noise=par_SC.noise_sigma)
        self._initialize_potential(par_SC, sc)
        conn = nn.connect.all2all(rgc.num, sc.num, include_self=True)
        chem = nn.VoltageJumpSynapse(rgc, sc, weights=par_SC.R2N_current, connection=conn,
                                     delay=par_SC.R2N_delay)
        mon_sc = nn.StateMonitor(sc, ['V', 'spike'])
        self.sc = sc
        self.chem = chem
        self.mon_sc = mon_sc

        self.net = super(RGC_SC_Net, self).__init__(gj_syn=gj, chem_syn=chem,
                                                    rgc=rgc, mon_rgc=mon_rgc,
                                                    sc=sc, mon_sc=mon_sc)

    def _initialize_potential(self, neu_par, neu):
        if neu_par.V_initial == 'uniform':
            neu.state[0] = np.random.rand(neu.num) * (neu_par.V_th - neu_par.V_reset) + neu_par.V_reset
        if neu_par.V_initial == 'gaussian':
            a = (neu_par.V_th - neu_par.V_reset) / 6
            b = (neu_par.V_th + neu_par.V_reset) / 2
            neu.state[0] = np.random.randn(neu.num) * a + b
            neu.state[0] = np.clip(neu.state[0], neu_par.V_reset, neu_par.V_th - 0.1)
        if neu_par.V_initial == 'reset':
            neu.state[0] = neu_par.V_reset

    def run(self, **kwargs):
        super(RGC_SC_Net, self).run(duration=self.par_RUN.duration, report=True,
                                    inputs=[(self.rgc, self.stim_values), (self.sc, self.par_SC.noise_mean)],
                                    **kwargs)

        if self.par_SYNC is not None:
            start = int(self.par_SYNC.t_start / nn.profile.get_dt())
            end = int(self.par_SYNC.t_end / nn.profile.get_dt())
            bin_size = int(self.par_SYNC.bin / nn.profile.get_dt())
            CC = nn.measure.cross_correlation(self.mon_rgc.spike[start: end], bin_size)
            self.par_SYNC.value = CC

    def show(self, figsize=None, save_name=None, title_=None, show=True):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(3, 1, figure=fig)

        # axes 1 : show RGC spikes
        # ------------------------
        ax11 = fig.add_subplot(gs[0:2, :])

        iis, tts = nn.raster_plot(self.mon_rgc, self.run_time())
        ax11.plot(tts, iis, '.', markersize=1)
        ax11.set_xlim(-0.1, self.par_RUN.duration + 0.1)
        ax11.set_ylim(-0.1, self.par_STIM.height * self.par_STIM.width + 0.1)
        ax11.set_ylabel('RGCs')
        if title_ is not None:
            ax11.set_title(title_)

        # axes 2 : show RON membrane potentials
        # --------------------------------------
        ax12 = fig.add_subplot(gs[2:, :])
        ax12.plot(self.run_time(), self.mon_sc.V[:, 0])

        ax12.set_xlim(-0.1, self.par_RUN.duration + 0.1)
        ax12.set_ylim(-0.1, self.par_SC.V_th + 0.1)
        ax12.set_ylabel('SC')
        ax12.set_xlabel('Time [{} ms]'.format(self.par_RUN.duration))

        # Final
        fig.align_ylabels([ax11, ax12])

        # Save or show
        # --------------------
        if save_name:
            plt.savefig(save_name)
            plt.close(fig)
        else:
            if show:
                plt.show()

    def show_RGC_spikes_in_image(self, time_phases, figsize=None,
                                 save_filename=None, show_fig=True):
        assert isinstance(time_phases, (tuple, list))
        assert isinstance(time_phases[0], (tuple, list))
        assert len(time_phases[0]) == 2

        iis, tts = nn.raster_plot(self.mon_rgc, self.run_time())
        if len(time_phases) == 2:
            gray_values = [0.4, 0.7]
        elif len(time_phases) == 3:
            gray_values = [0.3, 0.5, 0.7]
        else:
            gray_values = np.linspace(0.05, 0.95, len(time_phases))

        # get stimulus
        arr = np.zeros((self.par_STIM.height, self.par_STIM.width), dtype=np.float32)

        # spike bunch 1
        for i, phase in enumerate(time_phases):
            indexes = np.where(np.logical_and(tts > phase[0], tts < phase[1]))[0]
            neuron_indexes = iis[indexes]
            for n_index in neuron_indexes:
                row = n_index // self.par_STIM.width
                col = n_index % self.par_STIM.width
                arr[row, col] = gray_values[i]

        # show stimulus
        plt.figure(figsize=figsize or (8, 8))
        # plt.pcolor(arr, cmap='gray')
        plt.pcolor(arr, cmap=plt.cm.summer, vmin=0., vmax=1.)
        for i in range(self.par_STIM.height):
            plt.axhline(i, color='b')
        for i in range(self.par_STIM.width):
            plt.axvline(i, color='b')

        plt.colorbar()
        plt.tight_layout()

        if save_filename is None:
            if show_fig:
                plt.show()
        else:
            plt.savefig(save_filename)


def common_setting():
    # param of Run
    # ---------------
    loc_Run = default_Run.deepcopy()
    loc_Run.show_net_profile = False
    loc_Run.print_params = False
    loc_Run.duration = 8
    loc_Run.save_path = None
    loc_Run.title_format = ''
    loc_Run.title_vars = []
    loc_Run.dt = 0.01

    # param of RON
    # ---------------
    loc_SC = default_SC.deepcopy()
    loc_SC.tau = 1
    loc_SC.V_reset = 0
    loc_SC.V_th = 10
    loc_SC.noise_mean = 2
    loc_SC.tau_refractory = 0.35
    loc_SC.R2N_delay = 0.1
    loc_SC.R2N_current = 0.15

    # param of Stim
    # ---------------
    loc_Stim = default_Stim.deepcopy()

    # param of RGC
    # ----------------
    loc_RGC = default_RGC.deepcopy()
    loc_RGC.V_reset = 0
    loc_RGC.V_th = 10
    loc_RGC.V_initial = 'gaussian'
    loc_RGC.tau = 5
    loc_RGC.tau_refractory = 3.5

    return loc_RGC, loc_SC, loc_Run, loc_Stim


def template_run(loc_RGC, loc_SC, loc_Run, loc_Stim):
    print('stim={},ns={:.1f},gj={}-{:.2f},{}'.format(
        '-'.join([str(a) for a in sorted(set(loc_Stim.stim_sizes))]),
        loc_RGC.noise_sigma, loc_RGC.gj_w,
        loc_RGC.gj_spikelet, loc_Stim.name
    ))

    net = RGC_SC_Net(par_RGC=loc_RGC, par_RUN=loc_Run, par_STIM=loc_Stim, par_SC=loc_SC)
    if loc_Run.repeat > 1:
        for _ in range(loc_Run.repeat):
            net.run(repeat=True)
            net.show()
    else:
        net.run()
        net.show()
