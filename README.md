# mArch_mcs_analysis

This repository contains the replication files for sections 4 and 5 of "A fast algorithm for finding the confidence set of large model collections"

## Requirements and Installation

Running the replication files requires:
- The `fastMCS` toolbox with dependencies installed.
- The `mArch` toolbox with dependencies installed.
- Additional packages specified in `requirements_pip.txt` and `requirements_conda.txt`, to be installed using `pip` and `conda` respectively.

*NOTE*: This repo contains a copy of the `fastMCS` and `mArch` toolboxes, frozen in the state used for generating the results/figures in the release. These copies are not updated/maintained, and might get superseded by updates. For the latest versions of these toolboxes, refer to the relevant repository.

*NOTE*: the main run files were are parallelised on HPC nodes, therefore any attempt at replication should take into account this computational requirement. This is particularly the case for the Monte-Carlo analysis, which assumes 20 cores are available with 25GB per core. The files are provided for the sake of transparency and replication, and all intermediate/final results and outputs are provided in the associated release (see below).

## Release contents

The release provides zipped versions of the following folders. These contain all the intermediate results of the scripts, so that the outputs of the paper (i.e. figures /tables) can be generated directly from them, without requiring a full re-run of the entire analysis.

- `/data`: contains the empirical data used in the multivariate ARCH MCS analysis (Oxford-Man realized volatility library, Heber *et. al* 2009, version 0.3).
- `/forecasts`: contains the raw 4800 multivariate ARCH forecasts used in the MCS analysis, in two subfolders (one per empirical sample). Note that due to the large number of forecast models, the folder is broken down into 8 separate zipped files which together allow the overall `/forecasts` folder and its subfolders to be reconstructed.
- `/logs`: contains the run logs of all the parallelised scripts (MCS benchmarking, forecast generation and MCS partition analysis)
- `/losses`: contains the forecasts losses used in the MCS and partition analyses, as well as the results of the analyses.
- `/montecarlo`: contains results and outputs of the Monte-Carlo benchmarking and large-scale power analysis of the fast updating MCS algorithm (section 4 of the manuscript).
- `/outputs`: contains outputs of the multivariate ARCH MCS analysis.

## Run sequence:

The various scripts should be run in the following order, as the outputs of earlier scripts for the inputs of later ones. To run a later file (e.g. output generation) without running an earlier file (e.g. estimation), use the folders provided in the release as the source of the intermediate inputs.

### 1. Monte-Carlo benchmarking analysis (Section 4)

- `parallel_fastMCS_benchmark.py` - Run the parallelised Monte Carlo benchmarking exercise.
- `fastMCS_benchmark_outputs.py` - Generate outputs for the Monte Carlo benchmarking exercise.
- `parallel_fastMCS_power.py` - Run the parallelised Monte Carlo large-scale power exercise.
- `fastMCS_power_outputs.py` - Generate outputs for the Monte Carlo power exercise.

### 2. Multivariate ARCH forecast comparison (Section 5)

#### 2.1. Forecast generation

- `mArch_data_plots.py` - Generate the plots for the sample visualisation (figure 2 in manuscript).
- `parallel_mArch_main.py` - Run the parallelised Multivariate ARCH forecasting script (2 runs, once per sample).

#### 2.2. Loss calculation and MCS analysis

- `mArch_loss_calculation.py` - Calculate the 8 sets of forecast losses (2 samples, 2 horizons and 2 volatility proxies).
- `parallel_mArch_mcs_partition.py` - Run the parallelised 16 MCS partition analysis (2 bootstrap specifications per loss).
- `mArch_mcs_outputs.py` - Run the full MCS analysis and generate the tables for the paper (32 summary table components for main body - 16 nalaysis $\times$ 2 summary views, 16 full tables for appendix)

## Reference:

Heber, Gerd, Asger Lunde, Neil Shephard and Kevin K. Sheppard (2009) "Oxford-Man Institute's realized library", Oxford-Man Institute, University of Oxford.
