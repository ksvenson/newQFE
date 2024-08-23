"""
param_sweep.py

Author: Kai Svenson
Contact: kai.svenson628@gmail.com
Github: https://github.com/ksvenson/newQFE

Performs a parameter sweep and analysis of the 3d affine-transformed Ising model using `PROGRAM`.

TODO: Do not loop over configruations of k that are permutations of each other.
TODO: Investigate better estimator for the variance in `Sweep.multi_hist_step`.
      Existing one is straight-forward, but the author is investigating other (less-trivial) estimators that could be better.
"""

import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle as pkl
import multiprocessing as mp

KFLAGS = 'CDEFGHIJKLMNO'  # arguments for `PROGRAM`
CORES_PER_NODE = 40  # on the lq1 cluster the the Fermilab Lattice QCD Facility 
BETA_DECIMALS = 6  # beta is written in the data filenames with 6 decimal points

PROGRAM = 'ising_cubic'
CONDA_ENV = 'default'  # conda environment on the lq1 cluster

# Indices corresponding to the edges in the simple cubic, FCC, and BCC lattices.
SC_IDX = (0, 1, 2)
FCC_IDX = (3, 4, 5, 6, 7, 8)
BCC_IDX = (9, 10, 11, 12)

FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}


class Stat():
    """
    A class that helps ing plotting observables.
    """
    def __init__(self, label, axis=None, plot=False):
        """
        `label`: A reference to this statistic used in code. Ex: 'eng_var'.
        `axis`: How this statistic should be named in a plot. Ex: 'Energy Variance'.
        `plot`: Set to `True` if you want this statistic to be plotted.
        """
        self.label=label
        self.axis=axis
        self.plot=plot

class Sweep():
    """
    Keeps track of and organizes all parameters/files needed for a parameter sweep.
    """
    # `headers` are the columns of the .obs files created by `PROGRAM`.
    headers = [Stat('generation'), Stat('flip_metric', axis='Flip Metric', plot=True)]
    for i in range(13):
        headers.append(Stat(f'k{i}_energy', axis=f'Direction {i} Energy', plot=True))
    headers.append(Stat('magnetization', axis='Magnetization', plot=True))
    plot_mask = np.array([stat.plot for stat in headers])
    
    def __init__(self, nx, ny, nz, seeds, beta, ntherm, ntraj, base_dir, k, nwolff):
        """
        Initalize a parameter sweep.
        `nx`, `ny`, `nz`: Number of lattice sites along x, y, and z directions.
        `seeds`: A list of (ideally prime) numbers to be used as rng seeds.
        `beta`: All temperatures to simulate. Unique for each configuration of `k`.
        `ntherm`: Number of independent lattices to generate before measuring observables (for lattice to "thermalize").
        `ntraj`: Number of samples to take for a single rng seed. Total number of samples will be `len(seeds) * ntraj`.
        `base_dir`: Directory to store all necessary files.
        `k`: All coupling parameters to simulate.
        `nwolff`: Number of wolff updates to perform between each independent lattice. Set to `None` to use the Swendsen-Wang algorithm.
        """
        
        # Checks to make sure all arrays have the right shape
        assert len(k) == len(SC_IDX + FCC_IDX + BCC_IDX)
        assert len(beta.shape) - 1 == len(k)
        for i, ki in enumerate(k):
            assert beta.shape[i] == len(ki)
        if nwolff is not None:
            assert beta.shape == nwolff.shape
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.seeds = seeds
        self.beta = beta
        self.ntherm = ntherm
        self.ntraj = ntraj
        self.n_samples = len(self.seeds) * self.ntraj
        self.k = k
        self.nwolff = nwolff
        
        self.create(base_dir)
            
    @classmethod
    def load(cls, base_dir):
        """
        Loads a `Sweep` object into Python. Renames files in case `sweep.base_dir` was renamed.
        """
        with open(base_dir + '/params.pkl', 'rb') as f:
            sweep = pkl.load(f)
            sweep.create(base_dir)
            sweep.save()
            return sweep

    @classmethod
    def get_idxes(cls, keyword):
        """
        Gets indices of statistics in `cls.headers` that include `keyword`.
        """
        return [idx for idx, stat in enumerate(cls.headers) if keyword in stat.label]

    def save(self):
        """
        Save this `Sweep` object.
        """
        with open(self.params, 'wb') as f:
            pkl.dump(self, f)

    def create(self, base_dir):
        """
        Names files, creates directories, writes batch script, saves.
        """
        self.base_dir = base_dir
        self.data_dir = self.base_dir + '/data'
        self.figs_dir = self.base_dir + '/figs'
        self.stdout_dir = self.base_dir + '/stdout'
        self.stdout_fname = self.stdout_dir + '/slurm_%A.out'
        self.err_fname = self.stdout_dir + '/slurm_%A.err'
        self.mh_stdout_fname = self.stdout_dir + '/slurm_%A_mh.out'
        self.mh_err_fname = self.stdout_dir + '/slurm_%A_mh.err'
        self.commands = self.base_dir + '/commands.txt'
        self.batch = self.base_dir + '/batch.sh'
        self.params = self.base_dir + '/params.pkl'
        self.multi_hist_batch = self.base_dir + '/multi_hist_batch.sh'
        self.multi_hist_results = self.base_dir + '/multi_hist_results.npz'

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)
        os.makedirs(self.stdout_dir, exist_ok=True)
        self.write_script()
        self.save()

    def write_script(self):
        """
        Writes main batch script.
        Creates a list of commands in `self.commands`.
        Run these commands on the lq1 cluster by running:
        $ sbatch `self.batch`
        """
        with open(self.batch, 'w', newline='\n') as f:
            f.write('#!/bin/sh\n')
            f.write('#SBATCH --job-name=affine_parameter_sweep\n')
            f.write('#SBATCH --partition=lq1_cpu\n')
            f.write('#SBATCH --qos=normal\n')
            f.write(f'#SBATCH --output={self.stdout_fname}\n')
            f.write(f'#SBATCH --error={self.err_fname}\n')
            f.write('\n')
            f.write('#SBATCH --nodes=1\n')
            f.write('#SBATCH --tasks-per-node=1\n')
            f.write(f'#SBATCH --cpus-per-task={CORES_PER_NODE}\n')
            f.write('\n')
            f.write('module load parallel\n')
            f.write('\n')
            f.write(f'srun --ntasks 1 --cpus-per-task {CORES_PER_NODE} parallel -j {CORES_PER_NODE} -a {self.commands}\n')
        with open (self.commands, 'w', newline='\n') as f:
            count = 1
            for idx in np.ndindex(self.beta.shape):
                for seed in self.seeds:
                    f.write(f'{PROGRAM} -S {seed} -d {self.data_dir}')
                    f.write(f' -X {self.nx} -Y {self.ny} -Z {self.nz}')
                    f.write(f' -h {self.ntherm} -t {self.ntraj}')
                    for flag_idx, flag in enumerate(KFLAGS):
                        f.write(f' -{flag} {self.k[flag_idx][idx[flag_idx]]}')
                    f.write(f' -B {self.beta[idx]}')
                    if self.nwolff is None:
                        f.write(' -w -1')  # Use the Swendsen-Wang algorithm
                    else:
                        f.write(f' -w {self.nwolff[idx]}')
                    f.write(' ; ')
                    f.write(f'echo "Completed {count} of {self.beta.size * len(self.seeds)} on $(date) using node $(hostname)"')
                    f.write(f'\n')
                    count += 1

    def write_multi_hist_script(self):
        """
        Writes a batch script to perform a multiple histogram analysis on the lq1 cluster.
        """
        with open(self.multi_hist_batch, 'w', newline='\n') as f:
            f.write('#!/bin/sh\n')
            f.write('#SBATCH --job-name=affine_multi_hist\n')
            f.write('#SBATCH --partition=lq1_cpu\n')
            f.write('#SBATCH --qos=normal\n')
            f.write(f'#SBATCH --output={self.mh_stdout_fname}\n')
            f.write(f'#SBATCH --error={self.mh_err_fname}\n')
            f.write('\n')
            f.write('#SBATCH --nodes=1\n')
            f.write('#SBATCH --tasks-per-node=1\n')
            f.write(f'#SBATCH --cpus-per-task={CORES_PER_NODE}\n')
            f.write('\n')
            f.write('module load mambaforge\n')
            f.write(f'conda activate {CONDA_ENV}\n')
            f.write('\n')
            f.write(f'srun --ntasks 1 --cpus-per-task {CORES_PER_NODE} python -u param_sweep.py --base {self.base_dir} --multi-hist-cluster\n')

    def get_data_fnames(self, idx):
        """
        Get the filenames corresponding to the configuration `self.beta[idx]`.
        `idx` points to the temperature and all coupling parameters.
        There will be `len(self.seeds)` such filenames.
        """
        dir = f'{self.data_dir}/'
        for i, ki in enumerate(self.k):
            dir += f'{ki[idx[i]]:.2f}_'
        dir = dir[:-1]

        fnames = []
        for seed in self.seeds:
            fname = f'{self.nx}_{self.ny}_{self.nz}_{self.beta[idx]:.{BETA_DECIMALS}f}_{seed}.obs'
            path = f'{dir}/{fname}'
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Missing data file: {path}')
            fnames.append(path)
        return fnames

    def get_raw(self, config_idx):
        """
        Returns a numpy array with all data corresponding to the configuration `config_idx`.
        `config_idx` points to all coupling parameters, and a range of beta.

        This is the only method that reads from the data files for the sake of consistency.
        """
        raw = np.full(self.beta.shape[-1] + (self.n_samples, len(Sweep.headers)), np.nan)
        for beta_idx in range(raw.shape[0]):
            fnames = self.get_data_fnames(config_idx + (beta_idx,))
            for seed_idx, fname in enumerate(fnames):
                raw[beta_idx, seed_idx * self.ntraj : (seed_idx + 1) * self.ntraj] = np.genfromtxt(fname, delimiter=' ')
        return raw

    def read_avg_var(self):
        """
        Computes the averages and variances of all observables.
        """
        avg = np.full(self.beta.shape + (len(Sweep.headers),), np.nan)
        var = np.full(self.beta.shape + (len(Sweep.headers),), np.nan)
        for config_idx in np.ndindex(self.beta.shape[:-1]):
            raw = self.get_raw(config_idx)
            avg[config_idx] = raw.mean(axis=-2)
            var[config_idx] = raw.var(axis=-2)
        return avg, var
    
    @staticmethod
    def stagger_data(data, beta, beta_union):
        """
        Creates a new array `stagger` identical to `data`, but with the beta dimension expanded to `len(beta_union)`.
        The extra spaces are filled with `np.nan`.

        This method is necessary because each configuration has a unique range of beta that we sweep over.
        In order to plot `data`,  we need to arange our data so that it all lines up with a single beta space: `beta_union`.
        
        Pictorally, if data looks like:
        [[ arr 1 ],
         [ arr 2 ],
         [ arr 3 ]]

        Then `stagger` may look like:
        [[ arr 1 ]       ,
              [ arr 2]   ,
           [ arr 3]      ]
        
        wher the blank spaces are filled with `np.nan`.
        

        `data` should have >=3 dimensions. The first dimenions are for the configruations of k.
        The second to last dimension is for beta.
        The last dimension is an array of observables.
        """
        stagger = np.full(data.shape[:-2] + (len(beta_union), data.shape[-1]), np.nan)
        for idx in np.ndindex(stagger.shape[:-2]):
            stagger[idx, np.isin(beta_union, beta[idx])] = data[idx]
        return stagger

    def obs_plot(self, obs, stats, config_idx, free_idx, k_space, beta_space, surface=False, pcolormesh_kwargs={}):
        """
        Makes a heat map/surface plot of all the observables in `obs`.
        On the vertical axis is the coupling parameter correponding to `free_idx`. All other coupling parameters are fixed by `config_idx`.
        On the horizontal axis is temperature.

        If `surface` is False, makes a heat map. Else, makes a surface plot.
        """
        plot_idx = list(config_idx) + [slice(None)]  # To include beta dimension
        plot_idx[free_idx] = slice(None)
        plot_idx = tuple(plot_idx)

        beta_union = np.unique(beta_space[plot_idx])
        plot_obs = Sweep.stagger_data(obs[plot_idx], beta_space[plot_idx], beta_union)

        for stat_idx, stat in enumerate(stats):
            if stat.plot:
                if surface:
                    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                    ax.plot_surface(*np.meshgrid(beta_union, k_space), plot_obs[..., stat_idx])
                else:
                    fig, ax = plt.subplots()
                    pcm = ax.pcolormesh(beta_union, k_space, plot_obs[..., stat_idx], shading='nearest', **pcolormesh_kwargs)
                    fig.colorbar(pcm)
                ax.set(xlabel=r'$\beta$', ylabel=rf'$k_{free_idx}$', title=f'{self.base_dir}\n{stat.axis}')
                fig.savefig(f'{self.figs_dir}/{stat.label}.svg', **FIG_SAVE_OPTIONS)
                plt.close()

    def raw_obs_plot(self, config_idx, free_idx):
        """
        Plots averages and variances of all observables. See `obs_plot` for details.
        """
        avg, var = self.read_avg_var()
        self.obs_plot(avg, Sweep.headers, config_idx, free_idx, self.k[free_idx], self.beta)

        var_stats = []
        for avg_stat in Sweep.headers:
            var_label = avg_stat.label + '_var'
            var_axis = None
            if avg_stat.axis is not None:
                var_axis = avg_stat.axis + ' Variance'
            var_stats.append(Stat(var_label, var_axis, plot=avg_stat.plot))
        self.obs_plot(var, var_stats, config_idx, free_idx, self.k[free_idx], self.beta)

    def multi_hist_obs_plot(self, config_idx, free_idx):
        """
        Plots averages and variances of all observables saved in `self.multi_hist_results`. See `obs_plot` for details.
        """
        res = np.load(self.multi_hist_results)
        interp_beta = res['interp_beta']
        avg = res['avg']
        var = res['var']
        avg_stats = []
        var_stats = []
        for raw_stat in Sweep.headers:
            if raw_stat.plot:
                multi_hist_label = 'multi_hist_' + raw_stat.label
                multi_hist_axis = 'Multi-Histogram ' + raw_stat.axis
                avg_stats.append(Stat(multi_hist_label, multi_hist_axis, plot=raw_stat.plot))
                var_stats.append(Stat(multi_hist_label + '_var', multi_hist_axis + ' Variance', plot=raw_stat.plot))
        self.obs_plot(avg, avg_stats, config_idx, free_idx, self.k[free_idx], interp_beta)
        self.obs_plot(var, var_stats, config_idx, free_idx, self.k[free_idx], interp_beta)

    def refine_nwolff(self):
        """
        Creates a new `Sweep` object and directory with an improved set of `nwolff` parameters.

        First, we read the "flip metric" column of the data files.
        The flip metric is the average number of times a spin is flipped between independent lattice generations.
        To optimize execution time and statistics, the flip metric should be close to 1.
        To obtain the improved values, we scale the old values of `nwolff` by `1/flip_metric`, and round up.
        We also impose `5 <= nwolff <= self.nx * self.ny * self.nz`.
        """
        if self.nwolff is None:
            print('Sweep uses the Swendsen-Wang algorithm. Can not refine nwolff.')
            return
        avg, var = self.read_avg_var()
        flip_metric = avg[..., Sweep.get_idxes('flip_metric')[0]]
        self.nwolff = np.clip(((self.nwolff // flip_metric) + 1).astype(int), 5, self.nx * self.ny * self.nz)
        
        self.create(self.base_dir + '_rw')

    def refine_beta(self, step_size=0.0001, num_steps=20):
        """
        Creates a new `Sweep` object and directory with a beta range that only samples around criticality.

        The critical temperature `beta_crit` is estimated by finding the maximum energy variance in each direction, and then averaging over these maxima.
        Then from `beta_crit`, we go `num_steps` in each direction with step size `step_size`.

        WARNING: This method will needed to be edited if a lattice other than FCC is used.
        """
        avg, var = self.read_avg_var()
        eng_var = var[..., Sweep.get_idxes('energy')]
        eng_var = eng_var[..., np.array(FCC_IDX)]  # Edit this line if using a different type of lattice.

        max_eng_var = np.argmax(eng_var, axis=-2)  # find maximum along temperature axis
        max_eng_var = np.mean(max_eng_var, axis=-1).astype(int)  # average over all directions
        beta_crit = np.take_along_axis(self.beta, max_eng_var[..., np.newaxis], axis=-1).reshape(self.beta.shape[:-1])
        beta_increments = step_size * (np.arange(2*num_steps + 1) - num_steps)
        self.beta = np.add.outer(beta_crit, beta_increments).round(BETA_DECIMALS)

        self.create(self.base_dir + '_rb')

    def multi_hist(self, interp_beta):
        """
        Interpolate/extrapolate observables from `self.beta` to `interp_beta` using the multiple histogram method.
        See Newman and Barkema, Section 8.2.
        
        Saves the results in `self.multi_hist_results`.
        
        WARNING: This method will use all available cores on a machine. It is intended to be executed on the lq1 cluster.
        """
        pool = mp.Pool()
        args = [(idx, interp_beta) for idx in np.ndindex(self.beta.shape[:-1])]
        res = np.array(pool.starmap(self.multi_hist_step, args))
        avg = res[:, 0].reshape(interp_beta.shape + (np.count_nonzero(Sweep.plot_mask),))
        var = res[:, 1].reshape(interp_beta.shape + (np.count_nonzero(Sweep.plot_mask),))
        np.savez(self.multi_hist_results, interp_beta=interp_beta, avg=avg, var=var)

    def multi_hist_step(self, config_idx, interp_beta, tol=1e-2):
        """
        Performs a multiple histogram analysis with all observables corresponding to `config_idx`.
        See Newman and Barkema, Section 8.2.

        This method is resource intensive. It is intended to be called by `multi_hist`, and executed in parallel on the lq1 cluster.
        """
        print(f'Config Idx: {config_idx}')
        beta_space = self.beta[config_idx]
        k_vals = np.array([self.k[dir][idx] for dir, idx in enumerate(config_idx)])
        log_Z = np.zeros(beta_space.shape)  # intialize Z
        raw = self.get_raw(config_idx)
        energy = -1 * np.sum(k_vals * raw[..., Sweep.get_idxes('energy')], axis=-1)  # number in file is sum(s_i * s_{i+1})
        
        # Implementation follows Newman and Barkema. Section 8.2.1, Equation 8.36.
        beta_diff = np.add.outer(beta_space, -1 * beta_space)  # \beta_k - \beta_j
        exponent = np.multiply.outer(energy, beta_diff)        # E_{is} * (\beta_k - \beta_j)
        
        # Iteration do-while loop.
        print(f'{config_idx} Entering iteration loop')
        while True:
            new_log_Z = -1 * sp.special.logsumexp(exponent - log_Z, axis=-1)  # sum over j
            new_log_Z = sp.special.logsumexp(new_log_Z, axis=(0, 1))          # sum over i and s
            new_log_Z -= np.log(self.n_samples)                               # divide by n_j (which in constant in our case)

            convergence_metric = np.linalg.norm((new_log_Z - log_Z)/new_log_Z)
            if convergence_metric < tol:
                break
            log_Z = new_log_Z
            
            print(f'{config_idx} Completed iteration with convergence metric {convergence_metric}')
        print(f'{config_idx} Exited iteration loop')

        # Now we interpolate using Equation 8.39.
        beta_diff = np.add.outer(interp_beta[config_idx], -1 * beta_space)  # \beta - \beta_j
        exponent = np.multiply.outer(energy, beta_diff)                     # E_{is} * (\beta - \beta_j)
        denominator = -1 * sp.special.logsumexp(exponent - log_Z, axis=-1)  # sum over j
        interp_log_Z = sp.special.logsumexp(denominator, axis=(0, 1))       # sum over i and s
        interp_log_Z -= np.log(self.n_samples)                              # divide by n_j (which in constant in our case)

        # Preparing observables
        obs = raw[..., Sweep.plot_mask]
        offset = obs.min(axis=(0, 1)) - 1  # Find minimum across temperature and samples
        obs -= offset                      # Ensure we only work with non-negative numbers
        obs = np.log(obs)                  # We calculate the log of the expectation value
        
        # Computing averages
        avg = obs[..., np.newaxis, :] + denominator[..., np.newaxis]  # Q_{is} / denominator
        avg = sp.special.logsumexp(avg, axis=(0, 1))                  # sum over i and s
        avg -= interp_log_Z[:, np.newaxis] + np.log(self.n_samples)   # divide by Z(\beta) and n_j
        avg = np.exp(avg) + offset                                    # undo log and offset

        # Computing obs**2 so we can compute the variance
        var = 2 * obs[..., np.newaxis, :] + denominator[..., np.newaxis]  # Q_{is}^2 / denominator
        var = sp.special.logsumexp(var, axis=(0, 1))                      # sum over i and s
        var -= interp_log_Z[:, np.newaxis] + np.log(self.n_samples)       # divide by Z(\beta) and n_j
        # FIXME: There could be a better estimator for the variance, see TODO at top of file.
        var = np.exp(var) + 2 * offset * avg - offset**2 - avg**2         # This correction is needed since we computed \expval{(obs - offset)^2}
        
        print(f'{config_idx} completed')
        return np.stack((avg, var))

    def eng_hist(self, config_idx):
        """
        Creates a histogram of all energies measured corresponding to `config_idx`.
        Entries are color-coded according to what temperature they came from.
        """
        raw = self.get_raw(config_idx)
        k_vals = np.array([self.k[dir][idx] for dir, idx in enumerate(config_idx)])
        energy = -1 * np.sum(k_vals * raw[..., Sweep.get_idxes('energy')], axis=-1)

        fig, ax = plt.subplots()
        num_bins = get_num_bins(energy)

        cmap = plt.get_cmap('viridis_r')
        colors = cmap(np.linspace(0, 1, energy.shape[0]))

        ax.hist(energy.T, histtype='barstacked', bins=num_bins, color=colors)
        norm = mpl.colors.Normalize(vmin=self.beta[config_idx][0], vmax=self.beta[config_idx][-1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal')
        cbar.set_label(r'$\beta$')

        ax.set(xlabel='Energy', ylabel='Counts', title=f'Energy Histogram for Configuration {config_idx}')
        fig.savefig(f'{self.figs_dir}/eng_hist.svg', **FIG_SAVE_OPTIONS)
        plt.close()

    def comp_sweeps(self, other):
        """
        Perform a 2-sample Kolmogorov–Smirnov test between all observables of `self`, and all observables of `other`.
        Save results in `f'{self.base_dir}/comp_{other.base_dir}.npy'`.

        Mainly used to check that the Wolff algorithm and Swendsen-Wang algorithm are drawing observables from the same distribution.
        """
        assert self.beta.shape == other.beta.shape
        p_vals = np.full(self.beta.shape + (np.count_nonzero(Sweep.plot_mask),), np.nan)
        for config_idx in np.ndindex(self.beta.shape[:-1]):        
            self_data = self.get_raw(config_idx)[..., Sweep.plot_mask]
            otehr_data = other.get_raw(config_idx)[..., Sweep.plot_mask]

            p_vals[config_idx] = sp.stats.ks_2samp(self_data, otehr_data, axis=-2).pvalue
        np.save(f'{self.base_dir}/comp_{other.base_dir}.npy', p_vals)

    def plot_comp_sweeps(self, other, config_idx, free_idx):
        """
        Plot results from `comp_sweeps`.
        """
        p_vals = np.clip(np.load(f'{self.base_dir}/comp_{other.base_dir}.npy'), a_min=1e-20, a_max=1)  # In order to plot with a log-scale, need to clip zeros.
        stats = []
        for stat in Sweep.headers:
            if stat.plot:
                label = f'{stat.label}_comp_{other.base_dir}'
                axis = f'{stat.axis} KS Test p Value'
                stats.append(Stat(label, axis=axis, plot=stat.plot))

        kwargs = {'cmap': 'viridis',   
                  'norm': 'log'}
        self.obs_plot(p_vals, stats, config_idx, free_idx, self.k[free_idx], self.beta, pcolormesh_kwargs=kwargs)

def get_seeds(n):
    """
    Get `n` prime numbers starting with 1009. Used for rng seeds.
    """
    primes = []
    i = 1009
    while (len(primes) < n):
        primes.append(i)
        for divisor in range(3, i // 3 + 1):
            if i % divisor == 0:
                primes = primes[:-1]
                break
        i += 2
    return np.array(primes)


def get_num_bins(x):
    """
    Calculate ideal number of bins for a histogram using Freedman–Diaconis rule.
    """
    r = np.max(x) - np.min(x)
    p75, p25 = np.percentile(x, [75, 25])
    iqr = p75 - p25
    width = 2 * iqr / (x.size**(1/3))
    return round(np.mean(r / width))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='Name of base directory')
    
    parser.add_argument('--sw', action='store_true', help='Initialize sweep using Swendsen-Wang algorithm. Otherwise, use Wolff algorithm.')
    parser.add_argument('--ntraj', default=int(1e3), type=int, help='Number of samples to take for each rng seed.')
    parser.add_argument('--seeds', default=10, type=int, help='Number of rng seeds to use.')
    parser.add_argument('--beta-step', default=0.001, type=float, help='Used with --init and --refine-beta')
    parser.add_argument('--num-beta-steps', default=20, type=float, help='Only used with --refine-beta')

    parser.add_argument('--init', action='store_true', help='Create new sweep.')
    parser.add_argument('--analysis', action='store_true', help='Make plots of all measured observables.')
    parser.add_argument('--refine-nwolff', action='store_true', help='Create new sweep with improved nwolff parameter.')
    parser.add_argument('--refine-beta', action='store_true', help='Create new sweep which only samples around criticality.')
    parser.add_argument('--edit', action='store_true', help='No specific purpose. Edit Python script to perform any edits you need.')
    parser.add_argument('--multi-hist-local', action='store_true', help='Writes a batch script to be sent to the lq1 cluster.')
    parser.add_argument('--multi-hist-cluster', action='store_true', help='Performs a multiple histogram analysis on existing data. Resource intensive. Intended to only be used on the lq1 cluster.')
    parser.add_argument('--multi-hist-plot', action='store_true', help='Plot the results of a multiple histogram analysis.')
    parser.add_argument('--eng-hist', action='store_true', help='Create an histogram of energies for a specific configuration. Edit Python script directly to pick which configuration.')
    parser.add_argument('--calc-comp', help='Perform and save a Kolmogorov–Smirnov test between the sweeps located at the directories specified by --base and --calc_comp.')
    parser.add_argument('--plot-comp', help='Plot the results of the saved Kolmogorov–Smirnov test')
    args = parser.parse_args()

    requires_load_list = [args.analysis,
                          args.refine_nwolff,
                          args.refine_beta,
                          args.edit,
                          args.multi_hist_local,
                          args.multi_hist_cluster,
                          args.multi_hist_plot,
                          args.eng_hist,
                          args.calc_comp,
                          args.plot_comp]
    requires_load = False
    for flag in requires_load_list:
        if flag:
            requires_load = True
            break

    if args.base.endswith('/'):
        args.base = args.base[:-1]
    
    if args.init:
        nx = 32
        ny = 32
        nz = 32

        seeds = get_seeds(args.seeds)

        ntherm = int(2*1e3)

        k_step = 0.05
        beta_step = args.beta_step

        sc_k = [[0]] * len(SC_IDX)
        fcc_k = [[1]] * (len(FCC_IDX) - 1)
        fcc_k.append(np.arange(0.5, 1.5 + k_step, k_step).round(2))
        bcc_k = [[0]] * len(BCC_IDX)

        k = sc_k + fcc_k + bcc_k
        k = [np.array(arr) for arr in k]

        start = 0.092
        end = 0.115
        steps = round((end-start)/args.beta_step) + 1
        beta_space = np.linspace(start, end, steps, endpoint=True).round(BETA_DECIMALS)
        beta_shape = tuple(len(ki) for ki in k) + beta_space.shape
        beta = np.full(beta_shape, np.nan)
        # We don't know where the critical point is a priori. Take some data first, then use `refine_beta`
        # sample only around criticality.
        beta[...] = beta_space

        nwolff = None
        if not args.sw:
            # We don't know what nwolff should be a priori. Take some data first, then use `refine_nwolff`
            # to adjust `nwolff` appropriately.
            nwolff = np.full(beta_shape, 100)
        
        sweep = Sweep(nx, ny, nz, seeds, beta, ntherm, args.ntraj, args.base, k, nwolff)

    if requires_load:
        sweep = Sweep.load(args.base)

        if args.analysis:
            sweep.raw_obs_plot((0,)*13, FCC_IDX[-1])

        if args.refine_nwolff:
            sweep.refine_nwolff()

        if args.refine_beta:
            sweep.refine_beta(step_size=args.beta_step, num_steps=args.num_beta_steps)

        if args.edit:
            print(sweep.beta.shape)
            # Perform any edits you want here
            sweep.create(sweep.base_dir)

        if args.multi_hist_local:
            sweep.write_multi_hist_script()

        if args.multi_hist_cluster:
            # Perform a multiple histogram analysis that samples `res` times as many points in beta.
            res = 5
            interp_beta = np.full(sweep.beta.shape[:-1] + (res * sweep.beta.shape[-1],), np.nan)
            for config_idx in np.ndindex(sweep.beta.shape[:-1]):
                interp_beta[config_idx] = np.linspace(sweep.beta[config_idx][0], sweep.beta[config_idx][-1], num=interp_beta.shape[-1])
            sweep.multi_hist(interp_beta)
        
        if args.multi_hist_plot:
            sweep.multi_hist_obs_plot((0,)*13, FCC_IDX[-1])
        
        if args.eng_hist:
            # Can pick any config_idx you want.
            config_idx = [0]*13
            config_idx[FCC_IDX[-1]] = len(sweep.k[FCC_IDX[-1]]) // 2
            config_idx = tuple(config_idx)

            sweep.eng_hist(config_idx)
        
        if args.calc_comp:
            if args.calc_comp.endswith('/'):
                args.calc_comp = args.calc_comp[:-1]
            other = Sweep.load(args.calc_comp)
            sweep.comp_sweeps(other)
        
        if args.plot_comp:
            if args.plot_comp.endswith('/'):
                args.plot_comp = args.plot_comp[:-1]
            other = Sweep.load(args.plot_comp)
            sweep.plot_comp_sweeps(other, (0,)*13, FCC_IDX[-1])
