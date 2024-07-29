import argparse
import numpy as np
import matplotlib.pyplot as plt

KFLAGS = 'CDEFGHIJKLMNO'
# PROGRAM = '/project/affine/newQFE-KS/bin/ising_cubic'
PROGRAM = 'ising_cubic'
DATA_DIR = './data/'
FIGS_DIR = './figs/'
SCRIPT = 'sweep.sh'

# TODO Do not loop over configruations of k that are permutations of each other.

class Stat():
    def __init__(self, label, axis=None, plot=False):
        self.label=label
        self.axis=axis
        self.plot=plot

class Sweep():
    headers = [Stat('generation'), Stat('flip_metric')]
    for i in range(13):
        headers.append(Stat(f'k{i}_energy', axis=f'Direction {i} energy', plot=True))
    headers.append(Stat('magnetization', axis='Magnetization', plot=True))

    script_preamble = ('#!/bin/sh\n'
                       '#SBATCH --job-name=affine_parameter_sweep\n'
                       '#SBATCH --partition=lq1_cpu\n'
                       '#SBATCH --nodes=1\n\n\n')

    def __init__(self, nx, ny, nz, seed, beta, ntherm, ntraj, nwolff, data_dir, k):
        assert len(k) == 13
        for i, ki in enumerate(k):
            assert beta.shape[i] == len(ki)
        assert beta.shape == nwolff.shape
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.seed = seed
        self.beta = beta
        self.ntherm = ntherm
        self.ntraj = ntraj
        self.nwolff = nwolff
        self.data_dir = data_dir
        self.k = k

    def write_script(self):
        with open(f'./{SCRIPT}', 'w', newline='\n') as f:
            f.write(Sweep.script_preamble)
            count = 1
            for idx in np.ndindex(self.beta.shape):
                command = f'{PROGRAM} -S {self.seed} -d {self.data_dir}'
                command += f' -X {self.nx} -Y {self.ny} -Z {self.nz}'
                command += f' -h {self.ntherm} -t {self.ntraj}'
                for flag_idx, flag in enumerate(KFLAGS):
                    command += f' -{flag} {self.k[flag_idx][idx[flag_idx]]}'
                command += f' -B {self.beta[idx]}'
                command += f' -w {self.nwolff[idx]}'
                command += ' &'
                f.write(f'{command}\n')
                # f.write(f'echo "Completed {count} of {self.beta.size} trials"\n')

                if count >= 40:
                    f.write('\nwait\n\n')
                    count = 1
                else:
                    count += 1
            f.write('\nwait\n\n')

    def read_results(self):
        avg = np.empty(self.beta.shape + (len(Sweep.headers),))
        var = np.empty(self.beta.shape + (len(Sweep.headers),))
        for idx in np.ndindex(self.beta.shape):          
            dir = f'{DATA_DIR}/'
            for i, ki in enumerate(self.k):
                dir += f'{ki[idx[i]]:.2f}_'
            dir = dir[:-1]

            fname = f'{self.nx}_{self.ny}_{self.nz}_{self.beta[idx]:.6f}_{self.seed}.obs'
            fpath = f'{dir}/{fname}'

            data = np.genfromtxt(fpath, delimiter=' ')
            avg[idx] = data.mean(axis=0)
            var[idx] = data.var(axis=0)
        return avg, var
    
    def energy_plot(self, config_idx):
        avg, var = self.read_results()
        avg = avg[config_idx]
        var = var[config_idx]

        beta_space = self.beta[config_idx]
        
        for i, stat in enumerate(Sweep.headers):
            if not stat.plot:
                continue
            plt.figure()
            plt.plot(beta_space, avg[:, i])
            plt.xlabel(r'$\beta$')
            plt.ylabel(stat.axis)
            plt.savefig(f'{FIGS_DIR}{stat.label}.png')
            
            plt.figure()
            plt.plot(beta_space, var[:, i])
            plt.xlabel(r'$\beta$')
            plt.ylabel(f'{stat.axis} Variance')
            plt.savefig(f'{FIGS_DIR}{stat.label}_var.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', action='store_true', default=False, required=False)
    parser.add_argument('--plot', action='store_true', default=False, required=False)

    args = parser.parse_args()
    
    nx = 32
    ny = 32
    nz = 32

    seed = 1009

    ntherm = int(2*1e3)
    ntraj = int(1e3)

    k_samples = 10
    beta_samples = 10

    sc_k = [[0]] * 3
    fcc_k = [[1]] * 5
    fcc_k.append(np.linspace(0.9, 1.1, k_samples))
    bcc_k = [[0]]*4

    k = sc_k + fcc_k + bcc_k
    k = [np.array(arr) for arr in k]

    beta_shape = tuple(len(ki) for ki in k) + (beta_samples,)
    beta = np.empty(beta_shape)
    for idx in np.ndindex(beta_shape):
        # TODO Beta should straddle the critical point, which will be different for each configuration of k
        beta[idx[:-1]] = np.linspace(0.9, 1.1, beta_shape[-1])

    # TODO nwolff will depend on k configuration and beta
    nwolff = np.full(beta_shape, 100)

    sweep = Sweep(nx, ny, nz, seed, beta, ntherm, ntraj, nwolff, DATA_DIR, k)
    
    if args.script:
        sweep.write_script()

    if args.plot:
        sweep.energy_plot((0,)*13)
