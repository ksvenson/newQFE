Author: Kai Svenson
Contact: kai.svenson628@gmail.com
Github: https://github.com/ksvenson/newQFE

This directory contains one Python script `param_sweep.py`, and possibily other directories containing the results of various parameter sweeps.

`param_sweep.py` is used to perform parameter sweeps and analyses of the 3d affine-transformed Ising model. The program `ising_cubic` is used to perform the Monte Carlo simulations, and `param_sweep.py` keeps all necessary parameter and data files organized. An example workflow of performing a parameter sweep is given below. First, we initialize a sweep directory by running:

$ python param_sweep.py --init --base my_sweep

There are a few command line options to specifiy some properties of the sweep. For the most part however, edit `param_sweep.py` directly to configure all properties of the sweep. Most importantly, this includes the area of parameter space you want to sweep over.

Running the above command will create a directory named `my_sweep`. In addition, several subdirectories and files are created, described below:

`my_sweep/data/`: Stores all data files created by `ising_cubic`.
`my_sweep/figs/`: Stores all figures created by `param_sweep.py`.
`my_sweep/stdout/`: Standard output for the commands called by `my_sweep/batch.sh`.
`my_sweep/batch.sh`: Shell script to be run on the lq1 cluster. Pulls commands from `my_sweep/commands.txt` and runs them in parallel.
`my_sweep/commands.txt`: List of commands sent to the lq1 cluster.
`my_sweep/params.pkl`: A saved Python object storing all details of the parameter sweep.

Once a sweep is initialized, copy it to the lq1 cluster with `scp`, `rsync`, etc. Begin the sweep by running on lq1:

$ sbatch my_sweep/batch.sh

It's important that your working directory has `my_sweep` as an immediate subdirectory. Do NOT `cd` into `my_sweep` and then run `$ sbatch batch.sh`. You can monitor the progress of the sweep by reading the contents of `my_sweep/stdout/`.

When the sweep has completed, copy `my_sweep/` back to your local machine. Now we can use `param_sweep.py` to analyze the collected data. For a description of all the analyses, run:

$ python param_sweep.py --help

The most rudamentary analysis can be performed by running:

$ python param_sweep.py --base my_sweep --analysis

This will compute the average and variance of all observables, and make a plot placing temperature on one axis, and a coupling parameter on the other. To specify the type of plot, the configuration, and the specific coupling parameter to have on the axis, edit `param_sweep.py` directly.

Notes:
 - Some older sweeps (150824 and earlier) have an attribute `seed`, which is a single int for the rng seed. They lack the attributes `seeds` (a list of ints), 