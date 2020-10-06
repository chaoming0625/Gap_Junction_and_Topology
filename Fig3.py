# -*- coding: utf-8 -*-

from core import template_run
from core import common_setting


def circle():
    loc_RGC, loc_SC, loc_Run, loc_Stim = common_setting()

    # parameters of stimulus
    loc_Stim.name = 'circle'
    loc_Stim.stim_sizes = [12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others['radius'] = 20

    # parameters of RGC
    loc_RGC.noise_correlated_with_input = False
    loc_RGC.noise_sigma = 1.7
    loc_RGC.gj_w = 2.8
    loc_RGC.gj_spikelet = 0.15
    loc_RGC.height = loc_Stim.height
    loc_RGC.width = loc_Stim.width

    loc_SC.tau_refractory = 0.35
    loc_SC.R2N_current = 0.24

    # running
    # Stimulus.show_stimulus(loc_Stim)
    template_run(loc_RGC, loc_SC, loc_Run, loc_Stim)


def one_hole():
    loc_RGC, loc_SC, loc_Run, loc_Stim = common_setting()

    # parameters of stimulus
    loc_Stim.name = 'one_hole'
    loc_Stim.stim_sizes = [12, 20, 12]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others['inner_radius'] = 9
    loc_Stim.others['outer_radius'] = 28

    # parameters of RGC
    loc_RGC.noise_correlated_with_input = False
    loc_RGC.noise_sigma = 1.7
    loc_RGC.gj_w = 2.8
    loc_RGC.gj_spikelet = 0.15
    loc_RGC.height = loc_Stim.height
    loc_RGC.width = loc_Stim.width

    loc_SC.tau_refractory = 0.35
    loc_SC.R2N_current = 0.24

    # running
    # Stimulus.show_stimulus(loc_Stim)
    template_run(loc_RGC, loc_SC, loc_Run, loc_Stim)


def two_holes():
    loc_RGC, loc_SC, loc_Run, loc_Stim = common_setting()

    # parameters of stimulus
    loc_Stim.name = 'two_holes'
    loc_Stim.stim_sizes = [12, 12, 12, 20]
    loc_Stim.height = 95
    loc_Stim.width = 95
    loc_Stim.others['inner_radius'] = 9
    loc_Stim.others['outer_radius'] = 28

    # parameters of RGC
    loc_RGC.noise_correlated_with_input = False
    loc_RGC.noise_sigma = 1.7
    loc_RGC.gj_w = 2.8
    loc_RGC.gj_spikelet = 0.15
    loc_RGC.height = loc_Stim.height
    loc_RGC.width = loc_Stim.width

    loc_Run.repeat = 10
    loc_SC.tau_refractory = 0.35
    loc_SC.R2N_current = 0.24

    # running
    # Stimulus.show_stimulus(loc_Stim)
    template_run(loc_RGC, loc_SC, loc_Run, loc_Stim)


if __name__ == '__main__':
    # circle()
    one_hole()
    # two_holes()
