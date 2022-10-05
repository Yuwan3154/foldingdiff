# foldingdiff - Diffusion model for protein backbone generation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

We present a diffusion model for generating novel protein backbone structures. For more details, see our preprint on [arXiv](https://arxiv.org/abs/2209.15611).

![Animation of diffusion model protein folds over timesteps](plots/generated_0.gif)

## Installation

To install, clone this using `git clone` followed by `git lfs fetch`. Note that this requires [git-lfs](https://git-lfs.github.com) to be installed on your system.

This software is written in Python, notably using PyTorch, PyTorch Lightning, and the HuggingFace
transformers library. The required conda environment is defined within the `environment.yml` file. To set this up, make sure you have conda (or [mamba](https://mamba.readthedocs.io/en/latest/index.html)) installed, clone this repository, and run:

```bash
conda env create -f environment.yml
conda activate foldingdiff
pip install -e ./  # make sure ./ is the dir including setup.py
```

### Downloading data

We require some data files not packaged on Git due to their large size. These are not strictly required for sampling (as long as you are not using the `--testcomparison` option, see below); this is required for training your own model. We provide a script in the `data` dir to download requisite CATH data.

```bash
# Download the CATH dataset
cd data  # Ensure that you are in the data subdirectory within the codebase
chmod +x download_cath.sh
./download_cath.sh
```

## Training models

To train a model on the CATH dataset, use the script at `bin/train.py` in combination with one of the
json config files under `config_jsons` (or write your own). An example usage of this is as follows:

```bash
python bin/train.py config_jsons/cath_full_angles_cosine.json --dryrun
```

By default, the training script will calculate the KL divergence at each timestep before starting training, which can be quite computationally expensive with more timesteps. To skip this, append the `--dryrun` flag. The output of the model will be in the `results` folder with the following major files present:

```
results/
    - config.json           # Contains the config file for the huggingface BERT model itself
    - logs/                 # Contains the logs from training
    - models/               # Contains model checkpoints. By default we store the best 5 models by validation loss and the best 5 by training loss
    - training_args.json    # Full set of arguments, can be used to reproduce run
```

## Pre-trained models

We provide weights for a model trained on the CATH dataset. These weights are located under the `models/cath_pretrained` directory and are stored via Git LFS. The following code snippet shows how to load this model, load data, and perform a forward pass:

```python
from torch.utils.data.dataloader import DataLoader
from foldingdiff import modelling
from foldingdiff import datasets as dsets

# Load the model
m = modelling.BertForDiffusion.from_dir("models/cath_pretrained")

# Load dataset
# As part of loading, we try to compute internal angles in parallel. This may
# throw warnings like the following; this is normal.
# WARNING:root:Illegal values for omega in /home/*/projects/foldingdiff-main/data/cath/dompdb/2ebqA00 -- skipping
# After computing these once, the results are saved in a .pkl file under the
# foldingdiff source directory for faster loading in future calls.
clean_dset = dsets.CathCanonicalAnglesOnlyDataset(pad=128, trim_strategy='randomcrop')
noised_dset = dsets.NoisedAnglesDataset(clean_dset, timesteps=1000, beta_schedule='cosine')
dl = DataLoader(noised_dset, batch_size=32, shuffle=False)
x = iter(dl).next()

# Forward pass
predicted_noise = m(x['corrupted'], x['t'], x['attn_mask'])
```

Providing this path to a premade script, such as the one for sampling, is detailed below.

## Sampling protein backbones

To sample protein backbones, use the script `bin/sample.py`. Example commands to do this using the pretrained weights described above are as follows.

```bash
# To sample 10 backbones per length ranging from [50, 128) with a batch size of 512 - reproduces results in our manuscript
python ~/projects/foldingdiff/bin/sample.py -l 50 128 -n 10 -b 512 --device cuda:0
```

This will run the trained model contained in the `models/cath_pretrained` folder and generate sequences of varying lengths. If you wish to load the test dataset and include test chains in the generated plots, use the option `--testcomparison`; note that this requires downloading the CATH dataset, see above. Running `sample.py` will create the following directory structure in the diretory where it is run:

```
some_dir/
    - plots/            # Contains plots comparing the distribution of training/generated angles
    - sampled_angles/   # Contains .csv.gz files with the sampled angles
    - sampled_pdb/      # Contains .pdb files from converting the sampled angles to cartesian coordinates
    - model_snapshot/   # Contains a copy of the model used to produce results
```

Not specifying a `--device` will default to the first device `cuda:0`; use `--device cpu` to run on CPU (though this will be very slow). See the following table for runtimes from our machines.

| Device | Runtime estimates sampling 512 structures |
| --- | --- |
| Nvidia RTX 2080Ti | 7 minutes |
| i9-9960X (16 physical cores) | 2 hours |

### Maximum training similarity TM scores

After generating sequences, we can calculate TM-scores to evaluate the simliarity of the generated sequences and the original sequences. This is done using the script under `bin/tmscore_training.py`.

### Visualizing diffusion "folding" process

The above sampling code can also be run with the ``--fullhistory`` flag to write an additional subdirectory `sample_history` under each of the `sampled_angles` and `sampled_pdb` folders that contain pdb/csv files coresponding to each timestep in the sampling process. The pdb files, for example, can then be passed into the script under `foldingdiff/pymol_vis.py` to generate a gif of the folding process (as shown above). An example command to do this is:

```bash
python ~/projects/foldingdiff/foldingdiff/pymol_vis.py pdb2gif -i sampled_pdb/sample_history/generated_0/*.pdb -o generated_0.gif
```

**Note** this script lives separately from other plotting code because it depends on PyMOL; feel free to install/activate your own installation of PyMOL for this.

## Evaluating designability of generated backbones

One way to evaluate the quality of generated backbones is via their "designability". This refers to whether or not we can design an amino acid chain that will fold into the designed backbone. To evaluate this, we use the [ESM inverse folding model](https://github.com/facebookresearch/esm) to generate residues that are predicted to fold into our generated backbone, and use [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) to check whether that generated sequence actually does fold into a structure comparable to our backbone. (While prior backbone design works have used AlphaFold for their designability evaluations, this was previously done without providing AlphaFold with MSA information; OmegaFold is designed from the ground up to use sequence only, and is therefore better suited for this use case.)

### Inverse folding with ESM

We use a different conda environment for this step; see <https://colab.research.google.com/github/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb> for setup details. We found that the following command works on our machines:

```bash
mamba create -n inverse python=3.9 pytorch cudatoolkit pyg -c pytorch -c conda-forge -c pyg
conda activate inverse
mamba install -c conda-forge biotite
pip install git+https://github.com/facebookresearch/esm.git
```

After this, we `cd` into the folder that contains the `sampled_pdb` directory created by the prior step, and run:

```bash
python ~/projects/foldingdiff/bin/pdb_to_residues_esm.py sampled_pdb -o esm_residues
```

This creates a new folder, `esm_residues` that contains 10 potential residues for each of the pdb files contained in `sampled_pdb`.

### Structural prediction with OmegaFold

We use [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) to fold the amino acid sequences produced above. After creating and activating a separate conda environment and following the authors' instructions for installing OmegaFold, we use the following script to split our input amino acid fasta files across GPUs for inference, and subsequently calculate the self-consistency TM (scTM) scores.

```bash
# Fold each fasta, spreading the work over GPUs 0 and 1, outputs to omegafold_predictions folder
python ~/projects/foldingdiff/bin/omegafold_across_gpus.py esm_residues/*.fasta -g 0 1
# Calculate the scTM scores; parallelizes across all CPUs
python ~/projects/foldingdiff/bin/omegafold_self_tm.py  # Requires no arguments
```

After executing these commands, the final command produces a json file of all scTM scores, as well as various pdf files containing plots and correlations of the scTM score distribution.

## Tests

Tests are implemented through a mixture of doctests and unittests. To run unittests, run:

```bash
python -m unittest -v
```

You may see warnings like the following; these are expected.

```bash
WARNING:root:Illegal values for omega in protdiff-main/data/cath/dompdb/5a2qw00 -- skipping
```
