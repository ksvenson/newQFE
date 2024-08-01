import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle as pkl

KFLAGS = 'CDEFGHIJKLMNO'
CORES = 40

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
            return sweep
    
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
            f.write('#SBATCH --nodes=1\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write(f'#SBATCH --cpus-per-task={CORES}\n')
            f.write('\n\n')
            f.write('module load parallel\n')
            f.write('\n')
            f.write(f'srun parallel -j {CORES} -a {self.commands}\n')
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
                f.write(f'echo "Completed {count+1} of {self.beta.size} on $(date)"')
                f.write(f'\n')

    def read_results(self):
        avg = np.empty(self.beta.shape + (len(Sweep.headers),))
        var = np.empty(self.beta.shape + (len(Sweep.headers),))
        for idx in np.ndindex(self.beta.shape):          
            dir = f'{self.data_dir}/'
            for i, ki in enumerate(self.k):
                dir += f'{ki[idx[i]]:.2f}_'
            dir = dir[:-1]

            fname = f'{self.nx}_{self.ny}_{self.nz}_{self.beta[idx]:.6f}_{self.seed}.obs'
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
        
        k_space = self.k[free_idx]
        beta_space = self.beta[config_idx]
        
        for i, stat in enumerate(Sweep.headers):
            if not stat.plot:
                continue
            plt.figure()
            sns.heatmap(avg[..., i], xticklabels=beta_space, yticklabels=k_space, cbar=True)
            plt.xlabel(r'$\beta$')
            plt.ylabel(stat.axis)
            plt.savefig(f'{self.figs_dir}/{stat.label}.png', **FIG_SAVE_OPTIONS)

            plt.figure()
            sns.heatmap(var[..., i], xticklabels=beta_space, yticklabels=k_space, cbar=True)
            plt.xlabel(r'$\beta$')
            plt.ylabel(stat.axis + ' Variance')
            plt.savefig(f'{self.figs_dir}/{stat.label}_var.png', **FIG_SAVE_OPTIONS)

    def refine_nwolff(self):
        if self.sw:
            print('Sweep uses the Swedsen-Wang algorithm. Can not refine nwolff.')
            return
        res = self.read_results()
        if res is None:
            print('Missing data to refine nwolff.')
            return
        avg = res[0]
        flip_metric = avg[..., 1]
        self.nwolff = np.clip((self.nwolff // flip_metric).astype(int), 1, 500)
        self.base_dir = self.base_dir + '_refined'
        self.name_files()
        self.create()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--sw', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--refine-nwolff', action='store_true')
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()

    if args.base.endswith('/'):
        args.base = args.base[:-1]
    
    if args.init:
        nx = 32
        ny = 32
        nz = 32

        seed = 1009

        ntherm = int(2*1e3)
        ntraj = int(1e3)

        k_samples = 11
        beta_samples = 21

        sc_k = [[0]] * len(SC_IDX)
        fcc_k = [[1]] * (len(FCC_IDX) - 1)
        fcc_k.append(np.linspace(0.9, 1.1, k_samples).round(2))
        bcc_k = [[0]] * len(BCC_IDX)

        k = sc_k + fcc_k + bcc_k
        k = [np.array(arr) for arr in k]

        beta_shape = tuple(len(ki) for ki in k) + (beta_samples,)
        beta = np.empty(beta_shape)
        for idx in np.ndindex(beta_shape):
            # TODO Beta should straddle the critical point, which will be different for each configuration
            beta[idx[:-1]] = np.linspace(0.100, 0.110, beta_shape[-1]).round(6)

        sweep = Sweep(nx, ny, nz, seed, beta, ntherm, ntraj, args.base, k, sw=args.sw)

    if args.analysis:
        sweep = Sweep.load(args.base)
        sweep.energy_plot((0,)*13, FCC_IDX[-1])

    if args.refine_nwolff:
        sweep = Sweep.load(args.base)
        sweep.refine_nwolff()

    if args.diagnostic:
        sweep = Sweep.load(args.base)
        print(sweep.nwolff)
