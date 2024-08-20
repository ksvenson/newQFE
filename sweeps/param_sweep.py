import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
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
# TODO self.ntraj is used in equations for the number of samples. Replace with self.ntraj * len(self.seeds)
# TODO replace get_data_dir with get_data_fnames

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
    
    def __init__(self, nx, ny, nz, seeds, beta, ntherm, ntraj, base_dir, k, sw=False):
        assert len(k) == 13
        for i, ki in enumerate(k):
            assert beta.shape[i] == len(ki)
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.seeds = seeds
        self.beta = beta
        self.ntherm = ntherm
        self.ntraj = ntraj
        self.n_samples = len(self.seeds) * self.ntraj
        
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
            count = 1
            for idx in np.ndindex(self.beta.shape):
                for seed in self.seeds:
                    f.write(f'{PROGRAM} -S {seed} -d {self.data_dir}')
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
                    f.write(f'echo "Completed {count} of {self.beta.size * len(self.seeds)} on $(date) using node $(hostname)"')
                    f.write(f'\n')
                    count += 1

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

    def get_data_fnames(self, idx):
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
        raw = np.empty(self.beta[config_idx].shape + (self.n_samples, len(Sweep.headers)))
        for beta_idx in range(raw.shape[0]):
            fnames = self.get_data_fnames(config_idx + (beta_idx,))
            for seed_idx, fname in enumerate(fnames):
                raw[beta_idx, seed_idx * self.ntraj : (seed_idx + 1) * self.ntraj] = np.genfromtxt(fname, delimiter=' ')
        return raw

    def read_avg_var(self):
        avg = np.empty(self.beta.shape + (len(Sweep.headers),))
        var = np.empty(self.beta.shape + (len(Sweep.headers),))
        for config_idx in np.ndindex(self.beta.shape[:-1]):
            raw = self.get_raw(config_idx)
            avg[config_idx] = raw.mean(axis=-2)
            var[config_idx] = raw.var(axis=-2)
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

    def obs_plot(self, obs, stats, config_idx, free_idx, k_space, beta_space, surface=False, pcolormesh_kwargs={}):
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

        max_eng_var = np.argmax(eng_var, axis=-2)  # find maximum along temperature axis
        max_eng_var = np.mean(max_eng_var, axis=-1).astype(int)  # average over all directions
        beta_crit = np.take_along_axis(self.beta, max_eng_var[..., np.newaxis], axis=-1).reshape(self.beta.shape[:-1])
        beta_increments = step_size * (np.arange(2*num_steps + 1) - num_steps)
        self.beta = np.add.outer(beta_crit, beta_increments).round(BETA_DECIMALS)

        self.base_dir = self.base_dir + '_rb'
        self.name_files()
        self.create()

    def multi_hist(self, interp_beta):
        pool = mp.Pool()
        args = [(idx, interp_beta) for idx in np.ndindex(self.beta.shape[:-1])]
        res = np.array(pool.starmap(self.multi_hist_step, args))
        avg = res[:, 0].reshape(interp_beta.shape + (np.count_nonzero(Sweep.plot_mask),))
        var = res[:, 1].reshape(interp_beta.shape + (np.count_nonzero(Sweep.plot_mask),))
        np.savez(self.multi_hist_results, interp_beta=interp_beta, avg=avg, var=var)

    def multi_hist_step(self, config_idx, interp_beta, tol=1e-2):
        print(f'Config Idx: {config_idx}')
        beta_space = self.beta[config_idx]
        k_vals = np.array([self.k[dir][idx] for dir, idx in enumerate(config_idx)])
        log_Z = np.zeros(beta_space.shape)  # intialize Z
        raw = self.get_raw(config_idx)
        energy = -1 * np.sum(k_vals * raw[..., Sweep.get_idxes('energy')], axis=-1)  # number in file is sum(s_i * s_{i+1})
        
        # Implementation follows Newman and Barkema. Section 8.2.1, Equation 8.36.
        beta_diff = np.add.outer(beta_space, -1 * beta_space)
        exponent = np.multiply.outer(energy, beta_diff)
        # Iteration do-while loop.
        
        print(f'{config_idx} Entering iteration loop')
        while True:
            new_log_Z = -1 * sp.special.logsumexp(exponent - log_Z, axis=-1)
            new_log_Z = sp.special.logsumexp(new_log_Z, axis=(0, 1))
            new_log_Z -= np.log(self.n_samples)

            convergence_metric = np.linalg.norm((new_log_Z - log_Z)/new_log_Z)
            if convergence_metric < tol:
                break
            log_Z = new_log_Z
            
            print(f'{config_idx} Completed iteration with convergence metric {convergence_metric}')
        print(f'{config_idx} Exited iteration loop')

        # Now we interpolate using Equation 8.39.
        beta_diff = np.add.outer(interp_beta[config_idx], -1 * beta_space)
        exponent = np.multiply.outer(energy, beta_diff)
        denominator = -1 * sp.special.logsumexp(exponent - log_Z, axis=-1)
        interp_log_Z = sp.special.logsumexp(denominator, axis=(0, 1))
        interp_log_Z -= np.log(self.n_samples)

        # Preparing observables
        obs = raw[..., Sweep.plot_mask]
        offset = obs.min(axis=(0, 1)) - 1  # Find minimum across temperature and samples`self.traj`
        obs -= offset  # Ensure we only work with non-negative numbers
        obs = np.log(obs)  # We calculate the log of the expectation value
        
        # Computing averages
        avg = obs[..., np.newaxis, :] + denominator[..., np.newaxis]
        avg = sp.special.logsumexp(avg, axis=(0, 1))
        avg -= interp_log_Z[:, np.newaxis] + np.log(self.n_samples)
        avg = np.exp(avg) + offset

        # Computing obs**2 so we can compute the variance
        var = 2 * obs[..., np.newaxis, :] + denominator[..., np.newaxis]
        var = sp.special.logsumexp(var, axis=(0, 1))
        var -= interp_log_Z[:, np.newaxis] + np.log(self.n_samples)
        var = np.exp(var) + 2 * offset * avg - offset**2 - avg**2  # This correction is needed since we computed \expval{(obs - offset)^2}
        
        print(f'{config_idx} completed')
        return np.stack((avg, var))

    def eng_hist(self, config_idx):
        raw = self.get_raw(config_idx)
        k_vals = np.array([self.k[dir][idx] for dir, idx in enumerate(config_idx)])
        energy = -1 * np.sum(k_vals * raw[..., Sweep.get_idxes('energy')], axis=-1)
        energy = energy[::5]

        fig, ax = plt.subplots()
        num_bins = 100

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
        assert self.beta.shape == other.beta.shape
        p_vals = np.empty(self.beta.shape + (np.count_nonzero(Sweep.plot_mask),))
        for config_idx in np.ndindex(self.beta.shape[:-1]):        
            self_data = self.get_raw(config_idx)[..., Sweep.plot_mask]
            otehr_data = other.get_raw(config_idx)[..., Sweep.plot_mask]

            p_vals[config_idx] = sp.stats.ks_2samp(self_data, otehr_data, axis=-2).pvalue
        np.save(f'{self.base_dir}/comp_{other.base_dir}.npy', p_vals)

    def plot_comp_sweeps(self, other, config_idx, free_idx):
        p_vals = np.clip(np.load(f'{self.base_dir}/comp_{other.base_dir}.npy'), a_min=1e-20, a_max=1)
        stats = []
        for stat in Sweep.headers:
            if stat.plot:
                label = f'{stat.label}_comp_{other.base_dir}'
                axis = f'{stat.axis} KS Test p Value'
                stats.append(Stat(label, axis=axis, plot=stat.plot))

        kwargs = {'cmap': 'viridis',   
                  'norm': 'log'}
        self.obs_plot(p_vals, stats, config_idx, free_idx, self.k[free_idx], self.beta, pcolormes_kwargs=kwargs)

def get_seeds(n):
    primes = [2]
    i = 3
    while (len(primes) < n):
        primes.append(i)
        for divisor in range(3, i // 3 + 1):
            if i % divisor == 0:
                primes = primes[:-1]
                break
        i += 2
    return np.array(primes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    
    parser.add_argument('--sw', action='store_true')
    parser.add_argument('--ntraj', default=int(1e3), type=int)
    parser.add_argument('--seeds', default=10, type=int)
    parser.add_argument('--beta-step', default=0.0001, type=float)
    parser.add_argument('--num-beta-steps', default=20, type=float)

    parser.add_argument('--init', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--refine-nwolff', action='store_true')
    parser.add_argument('--refine-beta', action='store_true')
    parser.add_argument('--edit', action='store_true')
    parser.add_argument('--multi-hist-local', action='store_true')
    parser.add_argument('--multi-hist-cluster', action='store_true')
    parser.add_argument('--multi-hist-plot', action='store_true')
    parser.add_argument('--eng-hist', action='store_true')
    parser.add_argument('--calc-comp')
    parser.add_argument('--plot-comp')
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
        ntraj = args.ntraj

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
        beta = np.empty(beta_shape)
        # Ideally, beta should straddle the critical point, and thus be unique for each configuration.
        # However, when we initialize the sweep, we don't exactly know where the critical point is.
        # Using `refine_beta` to create a new sweep that only samples around criticality.
        beta[...] = beta_space
        
        sweep = Sweep(nx, ny, nz, seeds, beta, ntherm, ntraj, args.base, k, sw=args.sw)

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
            sweep.seeds = np.array([sweep.seed])
            sweep.n_samples = sweep.ntraj * len(sweep.seeds)
            sweep.create()
        if args.multi_hist_local:
            sweep.write_multi_hist_script()
        if args.multi_hist_cluster:
            interp_beta = np.empty(sweep.beta.shape[:-1] + (5 * sweep.beta.shape[-1],))
            for config_idx in np.ndindex(sweep.beta.shape[:-1]):
                interp_beta[config_idx] = np.linspace(sweep.beta[config_idx][0], sweep.beta[config_idx][-1], num=interp_beta.shape[-1])
            sweep.multi_hist(interp_beta)
        if args.multi_hist_plot:
            sweep.multi_hist_obs_plot((0,)*13, FCC_IDX[-1])
        if args.eng_hist:
            config_idx = [0]*13
            # config_idx[FCC_IDX[-1]] = len(sweep.k[FCC_IDX[-1]]) // 2
            config_idx[FCC_IDX[-1]] = -1
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
