"""
comp_sweeps.py

Used for a statistical comparison between Wolff and Swendsen-Wang algorithms.
Could be implemented more elegantly. Right now a lot of code is copy-pasted from param_sweep.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import param_sweep as ps
import argparse


def comp_sweeps(wolff, sw, config_idx, free_idx, dir):
    assert not wolff.sw
    assert sw.sw

    avg_wolff, var_wolff = wolff.read_avg_var()

    avg_sw, var_sw = sw.read_avg_var()

    plot_idx = list(config_idx) + [slice(None)]
    plot_idx[free_idx] = slice(None)
    plot_idx = tuple(plot_idx)

    avg = avg_wolff[plot_idx] - avg_sw[plot_idx]
    var = var_wolff[plot_idx]/wolff.ntraj + var_sw[plot_idx]/sw.ntraj
    sig = avg / np.sqrt(var)

    beta = wolff.beta[plot_idx]
    beta_union = np.array([])
    for beta_space in beta:
        beta_union = np.union1d(beta_union, beta_space)        
    k_space = wolff.k[free_idx]

    plot_sig = np.full((sig.shape[0], len(beta_union), sig.shape[-1]), np.nan)
    for k_idx, k_row in enumerate(plot_sig):
        k_row[np.isin(beta_union, beta[k_idx])] = sig[k_idx, :]

    ylabel = rf'$k_{free_idx}$'
    for stat_idx, stat in enumerate(ps.Sweep.headers):
        if not stat.plot:
            continue

        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(beta_union, k_space, plot_sig[..., stat_idx], shading='nearest')
        fig.colorbar(pcm)
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{stat.axis}')
        fig.savefig(f'{dir}/{stat.label}.svg', **ps.FIG_SAVE_OPTIONS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # First dir: wolff sweep
    # Second dir: sw sweep
    # Third dir: where to save figures
    parser.add_argument('--compare', nargs=3, default=None, required=True)
    args = parser.parse_args()

    for idx in range(len(args.compare)):
        if args.compare[idx].endswith('/'):
            args.compare[idx] = args.compare[idx][-1]
    wolff = ps.Sweep.load(args.compare[0])
    sw = ps.Sweep.load(args.compare[1])
    comp_sweeps(wolff, sw, (0,)*13, ps.FCC_IDX[-1], args.compare[2])