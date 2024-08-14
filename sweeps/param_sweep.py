import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import pickle as pkl
import multiprocessing as mp

KFLAGS = 'CDEFGHIJKLMNO'
CORES_PER_NODE = 40
BETA_DECIMALS = 6

PROGRAM = 'ising_cubic'
CONDA_ENV = 'default'

SC_IDX = (0, 1, 2)
FCC_IDX = (3, 4, 5, 6, 7, 8)
BCC_IDX = (9, 10, 11, 12)

FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}

# TODO Do not loop over configruations of k that are permutations of each other.

class Stat():
    def __init__(self, label, axis=None, plot=False):
        self.label=label
        self.axis=axis
        self.plot=plot

class Sweep():
    headers = [Stat('generation'), Stat('flip_metric', axis='Flip Metric', plot=True)]
    for i in range(13):
        headers.append(Stat(f'k{i}_energy', axis=f'Direction {i} Energy', plot=True))
    headers.append(Stat('magnetization', axis='Magnetization', plot=True))
    plot_mask = np.array([stat.plot for stat in headers])
    
    def __init__(self, nx, ny, nz, seed, beta, ntherm, ntraj, base_dir, k, sw=False):
        assert len(k) == 13
        for i, ki in enumerate(k):
            assert beta.shape[i] == len(ki)
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.seed = seed
        self.beta = beta
        self.ntherm = ntherm
        self.ntraj = ntraj
        
        self.base_dir = base_dir
        self.name_files()

        self.k = k
        self.sw = sw

        if not self.sw:
            self.nwolff = np.full(beta_shape, 100)
        
        self.create()
            
    @classmethod
    def load(cls, base):
        with open(base + '/params.pkl', 'rb') as f:
            sweep = pkl.load(f)
            sweep.base_dir = base
            sweep.name_files()
            sweep.save()
            return sweep

    @classmethod
    def get_idxes(cls, keyword):
        return [idx for idx, stat in enumerate(cls.headers) if keyword in stat.label]

    def save(self):
        with open(self.params, 'wb') as f:
            pkl.dump(self, f)

    def create(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)
        os.makedirs(self.stdout_dir, exist_ok=True)
        self.write_script()
        self.save()

    def name_files(self):
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

    def write_script(self):
        with open(self.batch, 'w', newline='\n') as f:
            f.write('#!/bin/sh\n')
            f.write('#SBATCH --job-name=affine_parameter_sweep\n')
            f.write('#SBATCH --partition=lq1_cpu\n')
            f.write('#SBATCH --qos=normal\n')
            f.write(f'#SBATCH --output={self.stdout_fname}\n')
            f.write(f'#SBATCH --error={self.err_fname}\n')
            f.write('\n')
            f.write(f'#SBATCH --nodes=1\n')
            f.write(f'#SBATCH --tasks-per-node=1\n')
            f.write(f'#SBATCH --cpus-per-task={CORES_PER_NODE}\n')
            f.write('\n')
            f.write('module load parallel\n')
            f.write('\n')
            f.write(f'srun --ntasks 1 --cpus-per-task {CORES_PER_NODE} parallel -j {CORES_PER_NODE} -a {self.commands}\n')
        with open (self.commands, 'w', newline='\n') as f:
            for count, idx in enumerate(np.ndindex(self.beta.shape)):
                f.write(f'{PROGRAM} -S {self.seed} -d {self.data_dir}')
                f.write(f' -X {self.nx} -Y {self.ny} -Z {self.nz}')
                f.write(f' -h {self.ntherm} -t {self.ntraj}')
                for flag_idx, flag in enumerate(KFLAGS):
                    f.write(f' -{flag} {self.k[flag_idx][idx[flag_idx]]}')
                f.write(f' -B {self.beta[idx]}')
                if self.sw:
                    f.write(' -w -1')  # Use the Swendsen-Wang algorithm
                else:
                    f.write(f' -w {self.nwolff[idx]}')
                f.write(' ; ')
                f.write(f'echo "Completed {count+1} of {self.beta.size} on $(date) using node $(hostname)"')
                f.write(f'\n')

    def write_multi_hist_script(self):
        with open(self.multi_hist_batch, 'w', newline='\n') as f:
            f.write('#!/bin/sh\n')
            f.write('#SBATCH --job-name=affine_multi_hist\n')
            f.write('#SBATCH --partition=lq1_cpu\n')
            f.write('#SBATCH --qos=normal\n')
            f.write(f'#SBATCH --output={self.mh_stdout_fname}\n')
            f.write(f'#SBATCH --error={self.mh_err_fname}\n')
            f.write('\n')
            f.write(f'#SBATCH --nodes=1\n')
            f.write(f'#SBATCH --tasks-per-node=1\n')
            f.write(f'#SBATCH --cpus-per-task={CORES_PER_NODE}\n')
            f.write('\n')
            f.write('module load mambaforge\n')
            f.write(f'conda activate {CONDA_ENV}\n')
            f.write('\n')
            f.write(f'srun --ntasks 1 --cpus-per-task {CORES_PER_NODE} python -u param_sweep.py --base {self.base_dir} --multi-hist-cluster\n')

    def get_data_dir(self, idx):
        dir = f'{self.data_dir}/'
        for i, ki in enumerate(self.k):
            dir += f'{ki[idx[i]]:.2f}_'
        dir = dir[:-1]

        fname = f'{self.nx}_{self.ny}_{self.nz}_{self.beta[idx]:.{BETA_DECIMALS}f}_{self.seed}.obs'
        path = f'{dir}/{fname}'
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing data file: {path}')
        return f'{dir}/{fname}'

    def read_avg_var(self):
        avg = np.empty(self.beta.shape + (len(Sweep.headers),))
        var = np.empty(self.beta.shape + (len(Sweep.headers),))
        for idx in np.ndindex(self.beta.shape):          
            path = self.get_data_dir(idx)
            data = np.genfromtxt(path, delimiter=' ')
            avg[idx] = data.mean(axis=0)
            var[idx] = data.var(axis=0)
        return avg, var
    
    @staticmethod
    def stagger_data(data, beta, beta_union):
        """
        Each configuration has a unique range over beta.
        This method expands the beta dimension of the `data` array to capute the union of all the unique beta spaces.
        The new entries are filled with `np.nan`.
        `data` should have >=3 dimensions. The first dimenions are for the configruations of k.
        The second to last dimension is for beta.
        The last dimension is an array of observables.
        """
        stagger = np.full(data.shape[:-2] + (len(beta_union), data.shape[-1]), np.nan)
        for idx in np.ndindex(stagger.shape[:-2]):
            stagger[idx, np.isin(beta_union, beta[idx])] = data[idx]
        return stagger

    def obs_plot(self, obs, stats, config_idx, free_idx, k_space, beta_space):
        plot_idx = list(config_idx) + [slice(None)]
        plot_idx[free_idx] = slice(None)
        plot_idx = tuple(plot_idx)

        beta_union = np.unique(beta_space[plot_idx])
        plot_obs = Sweep.stagger_data(obs[plot_idx], beta_space[plot_idx], beta_union)

        for stat_idx, stat in enumerate(stats):
            if not stat.plot:
                continue
            fig, ax = plt.subplots()
            pcm = ax.pcolormesh(beta_union, k_space, plot_obs[..., stat_idx], shading='nearest')
            fig.colorbar(pcm)
            ax.set_xlabel(r'$\beta$')
            ax.set_ylabel(rf'$k_{free_idx}$')
            ax.set_title(f'{self.base_dir}\n{stat.axis}')
            fig.savefig(f'{self.figs_dir}/{stat.label}.svg', **FIG_SAVE_OPTIONS)

    def raw_obs_plot(self, config_idx, free_idx):
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

    def multi_hist_obs_plot(self, config_idx, free_idx, interp_beta):
        self.multi_hist(interp_beta)
        res = np.load(self.multi_hist_results)
        expvals = res['expvals']
        stats = []
        for raw_stat in Sweep.headers:
            if raw_stat.plot:
                multi_hist_label = 'multi_hist_' + raw_stat.label
                multi_hist_axis = 'Multi-Histogram ' + raw_stat.axis
                stats.append(Stat(multi_hist_label, multi_hist_axis, plot=raw_stat.plot))
        self.obs_plot(expvals, stats, config_idx, free_idx, self.k[free_idx], interp_beta)

    def refine_nwolff(self):
        if self.sw:
            print('Sweep uses the Swedsen-Wang algorithm. Can not refine nwolff.')
            return
        avg, var = self.read_avg_var()
        flip_metric = avg[..., Sweep.get_idxes('flip_metric')[0]]
        self.nwolff = np.clip(((self.nwolff // flip_metric) + 1).astype(int), 5, self.nx * self.ny * self.nz)
        
        self.base_dir = self.base_dir + '_rw'
        self.name_files()
        self.create()

    def refine_beta(self, step_size=0.0001, num_steps=20):
        avg, var = self.read_avg_var()
        eng_var = var[..., Sweep.get_idxes('energy')]
        eng_var = eng_var[..., np.array(FCC_IDX)]

        eng_max_var = np.argmax(eng_var, axis=-2, keepdims=True)  # find maximum along temperature axis
        eng_max_var = np.mean(eng_max_var, axis=-1).astype(int)  # average over all directions
        new_beta = np.empty(self.beta.shape[:-1] + (2*num_steps + 1,))
        new_beta[...] = step_size * (np.arange(new_beta.shape[-1]) - num_steps)
        self.beta = (new_beta + np.take_along_axis(self.beta, eng_max_var, axis=-1)).round(BETA_DECIMALS)

        self.base_dir = self.base_dir + '_rb'
        self.name_files()
        self.create()

    def multi_hist_step(self, config_idx, interp_beta, obs_mask, tol=1e-2):
        print(f'Config Idx: {config_idx}')
        beta_space = self.beta[config_idx]
        k_vals = np.array([self.k[dir][idx] for dir, idx in enumerate(config_idx)])
        log_Z = np.zeros(beta_space.shape)  # intialize Z
        raw = np.empty(beta_space.shape + (self.ntraj, len(Sweep.headers)))
        for beta_idx in range(beta_space.shape[0]):
            path = self.get_data_dir(config_idx + (beta_idx,))
            raw[beta_idx] = np.genfromtxt(path, delimiter=' ')
        energy = -1 * np.sum(k_vals * raw[..., Sweep.get_idxes('energy')], axis=-1)  # number in file is sum(s_i * s_{i+1})
        
        # Implementation follows Newman and Barkema. Section 8.2.1, Equation 8.36.
        beta_diff = np.add.outer(beta_space, -1 * beta_space)
        exponent = np.multiply.outer(energy, beta_diff)
        # Iteration do-while loop.
        
        print(f'{config_idx} Entering iteration loop')

        while False: # True:
            new_log_Z = -1 * sp.special.logsumexp(exponent - log_Z, axis=-1)
            new_log_Z = sp.special.logsumexp(new_log_Z, axis=(0, 1))
            new_log_Z -= np.log(self.ntraj)

            convergence_metric = np.linalg.norm((new_log_Z - log_Z)/new_log_Z)
            if convergence_metric < tol:
                break
            log_Z = new_log_Z
            
            print(f'{config_idx} Completed iteration with convergence metric {convergence_metric}')
        print('Exited iteration loop')

        # Preparing observables
        obs = raw[..., obs_mask]
        offset = obs.min(axis=1) - 1  # Find minimum along axis with length `self.traj`
        obs -= offset[:, np.newaxis, :]  # Ensure we only work with non-negative numbers
        obs = np.log(obs)  # We calculate the log of the expectation value

        # Now we interpolate using Equation 8.39.
        beta_diff = np.add.outer(interp_beta[config_idx], -1 * beta_space)
        exponent = np.multiply.outer(energy, beta_diff)
        denominator = -1 * sp.special.logsumexp(exponent - log_Z, axis=-1)
        interp_log_Z = sp.special.logsumexp(denominator, axis=(0, 1))
        interp_log_Z -= np.log(self.ntraj)
        
        expvals = obs[..., np.newaxis, :] + denominator[..., np.newaxis]
        expvals = sp.special.logsumexp(expvals, axis=(0, 1))
        expvals -= interp_log_Z[:, np.newaxis] + np.log(self.ntraj)
        expvals = np.exp(expvals) + offset
        
        print(f'{config_idx} completed')
        return expvals

    def multi_hist(self, interp_beta, obs_mask=None):
        if obs_mask is None:
            obs_mask = Sweep.plot_mask
        pool = mp.Pool()
        args = [(idx, interp_beta, obs_mask) for idx in np.ndindex(self.beta.shape[:-1])]
        expvals = np.array(pool.starmap(self.multi_hist_step, args))
        expvals = expvals.reshape(interp_beta.shape + (np.count_nonzero(obs_mask),))
        np.savez(self.multi_hist_results, interp_beta=interp_beta, expvals=expvals)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    
    parser.add_argument('--sw', action='store_true')
    parser.add_argument('--ntraj', default=int(1e3), type=int)
    parser.add_argument('--beta-step', default=0.0001, type=float)
    parser.add_argument('--num-beta-steps', default=20, type=float)

    parser.add_argument('--init', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--refine-nwolff', action='store_true')
    parser.add_argument('--refine-beta', action='store_true')
    parser.add_argument('--edit', action='store_true')
    parser.add_argument('--multi-hist-local', action='store_true')
    parser.add_argument('--multi-hist-cluster', action='store_true')
    args = parser.parse_args()

    if args.base.endswith('/'):
        args.base = args.base[:-1]
    
    if args.init:
        nx = 32
        ny = 32
        nz = 32

        seed = 1009

        ntherm = int(2*1e3)
        ntraj = args.ntraj

        k_step = 0.05
        beta_step = args.beta_step

        sc_k = [[0]] * len(SC_IDX)
        fcc_k = [[1]] * (len(FCC_IDX) - 1)
        fcc_k.append(np.arange(0.5, 1.5 + k_step, k_step).round(2))
        bcc_k = [[0]] * len(BCC_IDX)

        k = sc_k + fcc_k + bcc_k
        k = [np.array(arr) for arr in k]

        beta_space = np.arange(0.092, 0.115 + beta_step, beta_step).round(BETA_DECIMALS)
        beta_shape = tuple(len(ki) for ki in k) + beta_space.shape
        beta = np.empty(beta_shape)
        # Ideally, beta should straddle the critical point, and thus be unique for each configuration.
        # However, when we initialize the sweep, we don't exactly know where the critical point is.
        # Using `refine_beta` to create a new sweep that only samples around criticality.
        beta[...] = beta_space
        
        sweep = Sweep(nx, ny, nz, seed, beta, ntherm, ntraj, args.base, k, sw=args.sw)

    if args.analysis or args.refine_nwolff or args.refine_beta or args.edit or args.multi_hist_local or args.multi_hist_cluster:
        sweep = Sweep.load(args.base)
        if args.analysis:
            sweep.raw_obs_plot((0,)*13, FCC_IDX[-1])
        if args.refine_nwolff:
            sweep.refine_nwolff()
        if args.refine_beta:
            sweep.refine_beta(step_size=args.beta_step, num_steps=args.num_beta_steps)
        if args.edit:
            sweep.ntraj = args.ntraj
            sweep.create()
        if args.multi_hist_local:
            sweep.write_multi_hist_script()
        if args.multi_hist_cluster:
            interp_beta = np.empty(sweep.beta.shape[:-1] + (10 * sweep.beta.shape[-1],))
            for config_idx in np.ndindex(sweep.beta.shape[:-1]):
                interp_beta[config_idx] = np.linspace(sweep.beta[config_idx][0], sweep.beta[config_idx][-1], num=interp_beta.shape[-1])
            sweep.multi_hist(interp_beta)
