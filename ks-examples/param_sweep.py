import numpy as np
import argparse

KFLAGS = 'CDEFGHIJKLMNO'
PROGRAM = '/project/affine/newQFE-KS/bin/ising_cubic'
DATA_DIR = './data/'
SCRIPT = 'sweep.sh'

# TODO Do not loop over configruations of k that are permutations of each other.

class Sweep():
    headers = ['generation', 'flip metric']
    for i in range(13):
        headers.append(f'k{i} energy')
    headers.append('magnetization')

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
                count += 1
            f.write('wait\n')

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
    
    def temp_plot(self, fig_dir):
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--make-script', action='store_true', default=False, required=False)
    parser.add_argument('--read-results', action='store_true', default=False, required=False)

    args = parser.parse_args()
    
    nx = 4
    ny = 4
    nz = 4

    seed = 1009

    ntherm = int(2*1e3)
    ntraj = int(1e3)

    sc_k = [[0]] * 3
    fcc_k = [[1]] * 5
    fcc_k.append(np.linspace(0.97, 1.03, 2))
    bcc_k = [[0]]*4

    k = sc_k + fcc_k + bcc_k
    k = [np.array(arr) for arr in k]

    beta_samples = 2
    beta_shape = tuple(len(ki) for ki in k) + (beta_samples,)
    beta = np.empty(beta_shape)
    for idx in np.ndindex(beta_shape):
        # TODO Beta should straddle the critical point, which will be different for each configuration of k
        beta[idx[:-1]] = np.linspace(0.97, 1.03, beta_shape[-1])

    # TODO nwolff will depend on k configuration and beta
    nwolff = np.full(beta_shape, 5)

    sweep = Sweep(nx, ny, nz, seed, beta, ntherm, ntraj, nwolff, DATA_DIR, k)
    
    if args.make_script:
        sweep.write_script()

    if args.read_results:
        sweep.read_results()
