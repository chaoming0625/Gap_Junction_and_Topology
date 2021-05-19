# -*- coding: utf-8 -*-
# %% [markdown]
# # *(Wang, et, al., 2020)* Gap Junction Network for Topological Detection

# %% [markdown]
# Implementation of the paper: *Wang, Chaoming, Risheng Lian, Xingsi Dong, Yuanyuan Mi, and Si Wu. "A neural network model with gap junction for topological detection." Frontiers in computational neuroscience 14 (2020). https://doi.org/10.3389/fncom.2020.571982*
#
# - Author : Chaoming Wang
# - Contact: adaduo@outlook.com

# %%
import numpy as np
import numba as nb
import brainpy as bp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# %%
bp.backend.set('numba', dt=0.01)

# %%
colors = ['mediumseagreen', 'cornflowerblue', 'darkkhaki', 'k', 'c', 'm', 'y', ]


# %%
@nb.njit
def numba_seed(_seed):
    np.random.seed(_seed)


# %% [markdown]
# ## Stimulus Generation

# %%
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

        plt.imshow(img[::-1, :], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=1.)

        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)

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
    def _black(height, width, stim_sizes, *args, **kwargs):
        """Example:
            Par_Stim.height = 50
            Par_Stim.width = 50
            Par_Stim.stim_sizes = [15,]
        """
        arr = np.zeros((height, width), dtype=np.float32)
        arr = arr.reshape((height * width,))
        indexes = [np.where(arr == 0)[0], ]
        stim = np.ones_like(arr) * stim_sizes[0]
        return arr, indexes, stim

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


# %% [markdown]
# ## Network Models

# %%
class LIF(bp.NeuGroup):
    target_backend = ['numpy', 'numba']

    def __init__(self, size, t_refractory=1., V_rest=0., V_reset=-5.,
                 V_th=20., tau=10., noise=0., **kwargs):
        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.tau = tau
        self.t_refractory = t_refractory
        self.noise = noise

        # variables
        num = bp.size2len(size)
        self.t_last_spike = bp.ops.ones(num) * -1e7
        self.input = bp.ops.zeros(num)
        self.refractory = bp.ops.zeros(num, dtype=bool)
        self.spike = bp.ops.zeros(num, dtype=bool)
        self.V = bp.ops.ones(num) * V_rest

        def f_part(V, t, Iext, V_rest, tau, noise):
            dvdt = (-V + V_rest + Iext) / tau
            return dvdt

        def g_part(V, t, Iext, V_rest, tau, noise):
            return noise / tau

        self.integral = bp.sdeint(f=f_part, g=g_part)

        super(LIF, self).__init__(size=size, **kwargs)

    def update(self, _t):
        for i in range(self.num):
            if _t - self.t_last_spike[i] <= self.t_refractory:
                self.refractory[i] = True
                self.spike[i] = False
            else:
                self.refractory[0] = False
                V = self.integral(self.V[i],
                                  _t,
                                  self.input[i],
                                  self.V_rest,
                                  self.tau,
                                  self.noise)
                if V >= self.V_th:
                    self.V[i] = self.V_reset
                    self.spike[i] = True
                    self.t_last_spike[i] = _t
                    self.refractory[i] = True
                else:
                    self.spike[i] = False
                    self.V[i] = V
            self.input[i] = 0.


# %%
class GapJunctionForLif(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, gjw=1., k_spikelet=0.1, **kwargs):
        self.gjw = gjw
        self.k_spikelet = k_spikelet

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        super(GapJunctionForLif, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for i in range(self.size):
            pre_id, post_id = self.pre_ids[i], self.post_ids[i]
            self.post.input[post_id] += self.gjw * (self.pre.V[pre_id] - self.post.V[post_id])
            if self.pre.spike[pre_id] and not self.post.refractory[post_id]:
                self.post.V[post_id] += self.gjw * self.k_spikelet


# %%
class VoltageJump(bp.TwoEndConn):
    target_backend = ['numpy', 'numba']

    def __init__(self, pre, post, conn, delay=0., weight=1., **kwargs):
        # parameters
        self.delay = delay
        self.weight = weight

        # connections
        self.conn = conn(pre.size, post.size)
        self.pre_ids, self.post_ids = conn.requires('pre_ids', 'post_ids')
        self.size = len(self.pre_ids)

        # variables
        self.Isyn = self.register_constant_delay('Isyn', size=self.size, delay_time=delay)

        super(VoltageJump, self).__init__(pre=pre, post=post, **kwargs)

    def update(self, _t):
        for i in range(self.size):
            pre_id, post_id = self.pre_ids[i], self.post_ids[i]
            self.Isyn.push(i, self.pre.spike[pre_id] * self.weight)
            if not self.post.refractory[post_id]:
                self.post.V += self.Isyn.pull(i)


# %%
def build(par_RGC, par_SC, par_STIM, duration=20., visualize=True, time_phases=None, rand_seed=12345):
    numba_seed(rand_seed)
    np.random.seed(rand_seed)

    # stimulus definition
    stim_img, stim_indexes, stim_values = Stimulus.get(par_STIM)

    # RGC group
    rgc = LIF(size=(par_STIM.height, par_STIM.width),
              t_refractory=par_RGC.tau_refractory,
              V_rest=par_RGC.V_reset,
              V_reset=par_RGC.V_reset,
              V_th=par_RGC.V_th,
              tau=par_RGC.tau,
              noise=par_RGC.noise_sigma,
              monitors=['V', 'spike'])
    a = (par_RGC.V_th - par_RGC.V_reset) / 6
    b = (par_RGC.V_th + par_RGC.V_reset) / 2
    rgc.V[:] = np.random.randn(rgc.num) * a + b
    rgc.V[:] = np.clip(rgc.V, par_RGC.V_reset, par_RGC.V_th - 0.1)

    # Gap junctions
    gj = GapJunctionForLif(rgc, rgc,
                           gjw=par_RGC.gj_w,
                           conn=bp.connect.GridEight(include_self=False),
                           k_spikelet=par_RGC.gj_spikelet)

    # SC read-out neuron
    sc = LIF(size=1,
             t_refractory=par_SC.tau_refractory,
             V_rest=par_SC.V_reset,
             V_reset=par_SC.V_reset,
             V_th=par_SC.V_th,
             tau=par_SC.tau,
             noise=par_SC.noise_sigma,
             monitors=['V', 'spike'])
    sc.V[:] = par_SC.noise_mean

    # RGC -> SC synapse
    chem = VoltageJump(rgc, sc,
                       conn=bp.connect.All2All(include_self=True),
                       weight=par_SC.R2N_current,
                       delay=par_SC.R2N_delay)

    # network
    net = bp.Network(rgc=rgc, gj=gj, sc=sc, chem=chem)
    net.run(duration=duration,
            inputs=[(rgc, 'input', stim_values), (sc, 'input', par_SC.noise_mean)],
            report=False)
    
    if visualize:
        # visualization
        if time_phases is not None:
            iis, tts = bp.measure.raster_plot(rgc.mon.spike, rgc.mon.ts)
            # SPIKE
            fig, gs = bp.visualize.get_figure(3, 2, row_len=2, col_len=6)
            ax11 = fig.add_subplot(gs[0: 2, 0])
            for i, phase in enumerate(time_phases):
                indexes = np.where(np.logical_and(tts > phase[0], tts < phase[1]))[0]
                neuron_indexes = iis[indexes]
                time_indexes = tts[indexes]
                # print("{}-st cluster length: {}".format(i, len(indexes)))
                ax11.scatter(time_indexes, neuron_indexes, c=colors[i], s=1)
            plt.xlim(-0.1, duration + 0.1)
            plt.ylim(-1, par_STIM.height * par_STIM.width)

            ax12 = fig.add_subplot(gs[2:, 0])
            bp.visualize.line_plot(sc.mon.ts, sc.mon.V,
                                   xlim=(-0.1, duration + 0.1),
                                   ylim=(par_SC.V_reset - 0.1, par_SC.V_th + 0.1),
                                   ylabel='SC Membrane Potential',
                                   xlabel=f'Time [{duration} ms]')
            fig.align_ylabels([ax11, ax12])
            plt.tight_layout()

            # IMAGE
            assert isinstance(time_phases, (tuple, list))
            assert isinstance(time_phases[0], (tuple, list))
            assert len(time_phases[0]) == 2

            ax = fig.add_subplot(gs[:, 1])
            for i, phase in enumerate(time_phases):
                indexes = np.where(np.logical_and(tts > phase[0], tts < phase[1]))[0]
                neuron_indexes = iis[indexes]
                for n_index in neuron_indexes:
                    row = n_index // par_STIM.width
                    col = n_index % par_STIM.width
                    aa = np.linspace(col, col + 1, 10).astype(np.float32)
                    bb = np.ones_like(aa, dtype=np.float32) * row
                    cc = bb + 1
                    plt.fill_between(aa, bb, cc, color=colors[i], alpha=1.)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, par_STIM.width])
            ax.set_ylim([0, par_STIM.height])
        else:
            fig, gs = bp.visualize.get_figure(3, 1, row_len=3, col_len=6)
            ax11 = fig.add_subplot(gs[0: 2, :])
            bp.visualize.raster_plot(rgc.mon.ts, rgc.mon.spike,
                                     xlim=(-0.1, duration + 0.1),
                                     ylim=(-1, par_STIM.height * par_STIM.width),
                                     ylabel='Spike Index of RGCs')
            ax12 = fig.add_subplot(gs[2:, :])
            bp.visualize.line_plot(sc.mon.ts, sc.mon.V,
                                   xlim=(-0.1, duration + 0.1),
                                   ylim=(par_SC.V_reset - 0.1, par_SC.V_th + 0.1),
                                   ylabel='SC Membrane Potential',
                                   xlabel=f'Time [{duration} ms]')
            fig.align_ylabels([ax11, ax12])
            plt.tight_layout()

        plt.show()
    return net


# %% [markdown]
# ## Parameters

# %%
loc_RGC = bp.tools.DictPlus(
    seed=12345,
    # 1. dynamic parameters
    V_reset=0, V_th=10, tau=5, tau_refractory=3.5, noise_sigma=1.,
    # 2. local gap junction
    gj_w=2.8, gj_spikelet=0.15, gj_conn='grid_eight'
)

# %%
loc_SC = bp.tools.DictPlus(
    # 1. dynamic parameters
    V_reset=0, V_th=10, tau=1, tau_refractory=0.4, noise_mean=1.0, noise_sigma=.1,
    # 2. parameters of connections between RGCs and RONs
    R2N_delay=0.1, R2N_current=0.24,
)

# %% [markdown]
# ## Running Results

# %% [markdown]
# ### Connectivity detection

# %%
# Object connected with background

loc_Stim = bp.tools.DictPlus(
    height=70, width=70, name='cd_object_background_connected', stim_sizes=[12, 20],
    others={'radius': 6, 'width': 4,  }
)

Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 8.)], 
      rand_seed=12345, )

# %%
# Object disconnected with background

loc_Stim = bp.tools.DictPlus(
    height=70, width=70, name='cd_object_background_disconnected', stim_sizes=[12, 20],
    others={'radius': 6, 'num_split': 2}
)

Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 2.1), (2.1, 4), (4, 8.)], 
      rand_seed=12345, )

# %% [markdown]
# ### Hole detection

# %%
# Circle

loc_Stim = bp.tools.DictPlus(height=95, width=95, name='circle',
                             stim_sizes=[12, 20], others={'radius': 20})
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim,
      duration=8., time_phases=[(0, 4), (4, 8)], rand_seed=12345, )

# %%
# One hole

loc_Stim = bp.tools.DictPlus(
    height=95, width=95, name='one_hole', stim_sizes=[12, 20, 12],
    others={'inner_radius': 10, 'outer_radius': 28}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 6.05), (6.05, 8.)], rand_seed=12345, )

# %%
# two holes

loc_Stim = bp.tools.DictPlus(
    height=95, width=95, name='two_holes', stim_sizes=[12, 12, 12, 20],
    others={'inner_radius': 9, 'outer_radius': 28}
)
Stimulus.show_stimulus(loc_Stim)

RGC = loc_RGC.deepcopy()
RGC['noise_sigma'] = 1.7
build(par_RGC=RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(1, 3), (3, 5.65), (5.65, 7), (7, 8)],
      rand_seed=3456789)

# %% [markdown]
# ### Insensitive to spatial frequency

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='HSF_rec_dots_closed', stim_sizes=[12, 12, 20],
    others={'length': 20, 'lw': 4, 'stride': 5}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 6.2), (6.2, 8)], 
      rand_seed=12345, )


# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='HSF_rectangle_closed', stim_sizes=[12, 12, 20],
    others={'length': 18, 'lw': 4}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 6), (6, 8)], 
      rand_seed=12345, )


# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='HSF_rectangle_open', stim_sizes=[12, 20],
    others={'length': 18, 'lw': 4, 'position': 'left'}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 8)], 
      rand_seed=12345, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='HSF_terminator_closed', stim_sizes=[12, 12, 20],
    others={'length': 18, 'lw': 3}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 6.5), (6.5, 8)], 
      rand_seed=2345, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='HSF_terminator_open', stim_sizes=[12, 20],
    others={'length': 18, 'lw': 3, 'position': 'left'}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 8)], 
      rand_seed=2345, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='HSF_triangle_closed', stim_sizes=[12, 12, 20],
    others={'a': 12, 'lw': 4}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 6.1), (6.1, 8)], 
      rand_seed=2345, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='HSF_triangle_open', stim_sizes=[12, 20],
    others={'a': 12, 'lw': 4, 'position': 'left'}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 8)], 
      rand_seed=2345, )

# %% [markdown]
# ### Sensitivity of Topological Detection

# %%
SC = loc_SC.deepcopy()
SC['tau_refractory'] = 0.6

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='hole_loophole', stim_sizes=[12, 20, 12],
    others={'inner_radius': 10, 'outer_radius': 20, 'angle': 40}
)
Stimulus.show_stimulus(loc_Stim)
build(par_RGC=loc_RGC, par_SC=SC, par_STIM=loc_Stim, duration=8.,
          time_phases=[(0, 4), (4, 6.5), (6.5, 8)], 
          rand_seed=23456, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='hole_loophole', stim_sizes=[12, 20, 12],
    others={'inner_radius': 10, 'outer_radius': 20, 'angle': 60}
)
Stimulus.show_stimulus(loc_Stim)
build(par_RGC=loc_RGC, par_SC=SC, par_STIM=loc_Stim, duration=8.,
          time_phases=[(0, 4), (4, 8)], 
          rand_seed=23456, )

# %%
angles = np.arange(40, 61, 2)
sc_spikes = []

for angle in angles:
    loc_Stim = bp.tools.DictPlus(height=80, width=80, name='hole_loophole', stim_sizes=[12, 20, 12],
                                 others={'inner_radius': 10, 'outer_radius': 20, 'angle': angle})
    net = build(par_RGC=loc_RGC, par_SC=SC, par_STIM=loc_Stim, duration=8., visualize=False, rand_seed=23456, )
    sc_spikes.append(net.sc.mon.spike.sum())
    
    
plt.plot(angles, sc_spikes)
plt.xlabel('Angle of the Loophole')
plt.ylabel('SC spikes')

# %% [markdown]
# ### Other Cases

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='cross', stim_sizes=[12, 20],
    others={'length': 18, 'lw': 5}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 8)], 
      rand_seed=2345, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='square', stim_sizes=[12, 20],
    others={'height': 30, 'width': 30}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 8)], 
      rand_seed=2345, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='square_hole', stim_sizes=[12, 20, 12, ],
    others={'outer_height': 30, 'outer_width': 30, 'inner_height': 16, 'inner_width': 16, }
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 6), (6, 8)], 
      rand_seed=2345, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='circle_smooth', stim_sizes=[12, 20],
    others={'inner_radius': 10, 'outer_radius': 20}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 8)], 
      rand_seed=23456, )

# %%
loc_Stim = bp.tools.DictPlus(
    height=80, width=80, name='one_hole_smooth', stim_sizes=[12, 20],
    others={'ring_inner_radius_s': 9,
            'ring_inner_radius_b': 14,
            'ring_outer_radius_s': 25,
            'ring_outer_radius_b': 30,
            'position': 'center'}
)
Stimulus.show_stimulus(loc_Stim)

build(par_RGC=loc_RGC, par_SC=loc_SC, par_STIM=loc_Stim, duration=8.,
      time_phases=[(0, 4), (4, 6), (6, 8)], 
      rand_seed=23456, )
