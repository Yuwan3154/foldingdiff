"""
Plot distribution plots, CDF, and ramachandran plots for sampled proteins
Meant as a way to re-plot the sampled proteins after sampling has been done.
Assumes directory structure (e.g., contains sampled_angles folder)

Usage: python sample_potting_only.py <dirname_with_results>
"""

from glob import glob
import os, sys
import re
import logging
import json
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

import torch

# Import data loading code from main training script
from train import get_train_valid_test_sets
from annot_secondary_structures import make_ss_cooccurrence_plot


from sample import (
    FT_NAME_MAP,
    build_datasets,
    plot_distribution_overlap,
    plot_ramachandran,
)

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))
import plotting
import custom_metrics as cm


def int_getter(x: str) -> int:
    """Fetches integer value out of a string"""
    matches = re.findall(r"[0-9]+", x)
    assert len(matches) == 1
    return int(matches.pop())


def main(dir_name: Path):
    """Run the script"""
    plotdir = dir_name / "plots"
    assert plotdir.is_dir()

    # Read in the training args
    with open(dir_name / "model_snapshot/training_args.json") as source:
        training_args = json.load(source)
    _, _, test_dset = build_datasets(training_args)

    phi_idx = test_dset.feature_names["angles"].index("phi")
    psi_idx = test_dset.feature_names["angles"].index("psi")

    # Ramachandran for test set
    select_by_attn = lambda x: x["angles"][x["attn_mask"] != 0]
    test_values = [
        select_by_attn(test_dset.dset.__getitem__(i, ignore_zero_center=True))
        for i in range(len(test_dset))
    ]
    test_values_stacked = torch.cat(test_values, dim=0).cpu().numpy()
    plot_ramachandran(
        test_values_stacked[:5000, phi_idx],
        test_values_stacked[:5000, psi_idx],
        annot_ss=True,
        fname=plotdir / "ramachandran_test_annot.pdf",
    )

    # Read in the sampled angles
    sampled_fnames = sorted(
        glob(os.path.join(dir_name, "sampled_angles/*.csv.gz")),
        key=lambda x: int_getter(os.path.basename(x)),
    )
    sampled_dfs = []
    for fname in sampled_fnames:
        df = pd.read_csv(fname, index_col=0)
        sampled_dfs.append(df)
    assert sampled_dfs
    logging.info(f"Found {len(sampled_dfs)} sets of generated angles")

    sampled_stacked = np.vstack([df.values for df in sampled_dfs])
    plot_ramachandran(
        sampled_stacked[:5000, phi_idx],
        sampled_stacked[:5000, psi_idx],
        fname=plotdir / "ramachandran_generated.pdf",
    )

    # Plot distribution overlap
    multi_fig, multi_axes = plt.subplots(dpi=300, nrows=2, ncols=3, figsize=(13, 6.5))
    step_multi_fig, step_multi_axes = plt.subplots(
        dpi=300, nrows=2, ncols=3, figsize=(14, 6.5)
    )
    for i, ft_name in enumerate(test_dset.feature_names["angles"]):
        orig_values = test_values_stacked[:, i]
        samp_values = sampled_stacked[:, i]

        kl = cm.kl_from_empirical(samp_values, orig_values, nbins=200, pseudocount=True)
        logging.info(f"Angle {ft_name} KL(generated || test) = {kl}")

        ft_name_readable = FT_NAME_MAP[ft_name]

        # Plot combo plots
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"{ft_name_readable} distribution, KL={kl:.4f}",
            edgecolor="black",
            ax=multi_axes.flatten()[i],
            show_legend=i == 0,
            alpha=0.45,
            bins=60,
        )
        plot_distribution_overlap(
            {"Test": orig_values, "Sampled": samp_values},
            title=f"{ft_name_readable} CDF",
            cumulative=True,
            histtype="step",
            ax=step_multi_axes.flatten()[i],
            show_legend=i == 0,
            alpha=0.6,
            bins=60,
        )
    for i in [3, 4, 5]:
        multi_axes.flatten()[i].set_xlabel("Angle (rad)", fontsize=14)
        step_multi_axes.flatten()[i].set_xlabel("Angle (rad)", fontsize=14)
    for i in [0, 3]:
        multi_axes.flatten()[i].set_ylabel("Normalized frequency", fontsize=14)
        step_multi_axes.flatten()[i].set_ylabel("Cumulative frequency", fontsize=14)
    # for i in range(6):
    # https://stackoverflow.com/questions/29188757/matplotlib-specify-format-of-floats-for-tick-labels
    # multi_axes.flatten()[i].yaxis.set_major_formatter(FormatStrFormatter("%.2g"))

    multi_fig.tight_layout()
    multi_fig.savefig(plotdir / "dist_combined.pdf", bbox_inches="tight")
    step_multi_fig.tight_layout()
    step_multi_fig.savefig(plotdir / "cdf_combined.pdf", bbox_inches="tight")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) == 1:
        main(Path(os.getcwd()))
    else:
        main(Path(sys.argv[1]))
