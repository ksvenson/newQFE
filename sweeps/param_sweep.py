import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

KFLAGS = 'CDEFGHIJKLMNO'
CORES_PER_NODE = 40
BETA_DECIMALS = 6

PROGRAM = 'ising_cubic'
DEFAULT_BASE_DIR = './sweep'

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
    
    def __init__(self, nx, ny, nz, seed, beta, ntherm, ntraj, base_dir, k, sw=False, nodes=1):
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
        # TODO: Haven't written the batch file to use multiple nodes yet.
        self.nodes = nodes

        if not self.sw:
            self.nwolff = np.full(beta_shape, 100)
        
        self.create()
            
    @classmethod
    def load(cls, base):
        with open(base + '/params.pkl', 'rb') as f:
            sweep = pkl.load(f)
            sweep.base_dir = base
            sweep.name_files()
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
        self.commands = self.base_dir + '/commands.txt'
        self.batch = self.base_dir + '/batch.sh'
        self.params = self.base_dir + '/params.pkl'

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
            f.write(f'#SBATCH --cpus-per-task={CORES_PER_NODE}')
            f.write('\n\n')
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

    def read_results(self):
        avg = np.empty(self.beta.shape + (len(Sweep.headers),))
        var = np.empty(self.beta.shape + (len(Sweep.headers),))
        for idx in np.ndindex(self.beta.shape):          
            dir = f'{self.data_dir}/'
            for i, ki in enumerate(self.k):
                dir += f'{ki[idx[i]]:.2f}_'
            dir = dir[:-1]

            fname = f'{self.nx}_{self.ny}_{self.nz}_{self.beta[idx]:.{BETA_DECIMALS}f}_{self.seed}.obs'
            fpath = f'{dir}/{fname}'
            if not os.path.isfile(fpath):
                print(f'Missing file: {fpath}')
                return None
            else:
                data = np.genfromtxt(fpath, delimiter=' ')
                avg[idx] = data.mean(axis=0)
                var[idx] = data.var(axis=0)
        return avg, var
    
    def energy_plot(self, config_idx, free_idx):
        res = self.read_results()
        if res is None:
            print('Can not make plots. Missing data.')
            return
        avg, var = res

        plot_idx = list(config_idx) + [slice(None)]
        plot_idx[free_idx] = slice(None)
        plot_idx = tuple(plot_idx)

        avg = avg[plot_idx]
        var = var[plot_idx]
        beta = self.beta[plot_idx]
        beta_union = np.array([])
        for beta_space in beta:
            beta_union = np.union1d(beta_union, beta_space)        
        k_space = self.k[free_idx]

        plot_avg = np.full((avg.shape[0], len(beta_union), avg.shape[-1]), np.nan)
        plot_var = np.full((var.shape[0], len(beta_union), var.shape[-1]), np.nan)
        for k_idx, k_row in enumerate(plot_avg):
            k_row[np.isin(beta_union, beta[k_idx])] = avg[k_idx, :]
        for k_idx, k_row in enumerate(plot_var):
            k_row[np.isin(beta_union, beta[k_idx])] = var[k_idx, :]

        ylabel = rf'$k_{free_idx}$'
        for stat_idx, stat in enumerate(Sweep.headers):
            if not stat.plot:
                continue

            fig, ax = plt.subplots()
            pcm = ax.pcolormesh(beta_union, k_space, plot_avg[..., stat_idx], shading='nearest')
            fig.colorbar(pcm)
            ax.set_xlabel(r'$\beta$')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{self.base_dir}\n{stat.axis}')
            fig.savefig(f'{self.figs_dir}/{stat.label}.svg', **FIG_SAVE_OPTIONS)

            fig, ax = plt.subplots()
            pcm = ax.pcolormesh(beta_union, k_space, plot_var[..., stat_idx], shading='nearest')
            fig.colorbar(pcm)
            ax.set_xlabel(r'$\beta$')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{self.base_dir}\n{stat.axis} Variance')
            fig.savefig(f'{self.figs_dir}/{stat.label}_var.svg', **FIG_SAVE_OPTIONS)

    def refine_nwolff(self):
        if self.sw:
            print('Sweep uses the Swedsen-Wang algorithm. Can not refine nwolff.')
            return
        res = self.read_results()
        if res is None:
            print('Missing data to refine nwolff.')
            return
        avg = res[0]
        flip_metric = avg[..., Sweep.get_idxes('flip_metric')[0]]
        self.nwolff = np.clip(((self.nwolff // flip_metric) + 1).astype(int), 5, self.nx * self.ny * self.nz)
        
        self.base_dir = self.base_dir + '_rw'
        self.name_files()
        self.create()

    def refine_beta(self, step_size=0.0001, num_steps=20):
        res = self.read_results()
        if res is None:
            print('Missing data to refine beta')
            return
        var = res[1]
        eng_var = var[..., Sweep.get_idxes('energy')]
        eng_var = var[..., np.array(FCC_IDX)]

        eng_max_var = np.argmax(eng_var, axis=-2, keepdims=True)  # find maximum along temperature axis
        eng_max_var = np.mean(eng_max_var, axis=-1).astype(int)  # average over all directions
        new_beta = np.empty(self.beta.shape[:-1] + (2*num_steps + 1,))
        new_beta[...] = step_size * (np.arange(new_beta.shape[-1]) - num_steps)
        self.beta = (new_beta + np.take_along_axis(self.beta, eng_max_var, axis=-1)).round(BETA_DECIMALS)

        self.base_dir = self.base_dir + '_rb'
        self.name_files()
        self.create()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    
    parser.add_argument('--sw', action='store_true')
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--ntraj', default=int(1e3), type=int)
    parser.add_argument('--beta-step', default=0.0001, type=float)
    parser.add_argument('--num-beta-steps', default=20, type=float)

    parser.add_argument('--init', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--refine-nwolff', action='store_true')
    parser.add_argument('--refine-beta', action='store_true')
    parser.add_argument('--edit', action='store_true')
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

        beta_space = np.arange(0.098, 0.106 + beta_step, beta_step).round(BETA_DECIMALS)
        beta_shape = tuple(len(ki) for ki in k) + beta_space.shape
        beta = np.empty(beta_shape)
        # Ideally, beta should straddle the critical point, and thus be unique for each configuration.
        # However, when we initialize the sweep, we don't exactly know where the critical point is.
        # Using `refine_beta` to create a new sweep that only samples around criticality.
        beta[...] = beta_space
        
        sweep = Sweep(nx, ny, nz, seed, beta, ntherm, ntraj, args.base, k, sw=args.sw, nodes=args.nodes)

    if args.analysis or args.refine_nwolff or args.refine_beta or args.edit:
        sweep = Sweep.load(args.base)
        if args.analysis:
            sweep.energy_plot((0,)*13, FCC_IDX[-1])
        if args.refine_nwolff:
            sweep.refine_nwolff()
        if args.refine_beta:
            sweep.refine_beta(step_size=args.beta_step, num_steps=args.num_beta_steps)
        if args.edit:
            sweep.nodes = args.nodes
            sweep.ntraj = args.ntraj
            sweep.create()
