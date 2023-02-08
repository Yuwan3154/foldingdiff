"""
Run MDS on structures to create an embedding visualization

Coloring options:
* training TM similarity
* scTM
* helix/beta strand annotations
* length
"""

import os
import json
import logging
from glob import glob
import argparse


import pandas as pd
from sklearn.manifold import MDS
from matplotlib import pyplot as plt

from hclust_structures import get_pairwise_tmscores, int_getter
from annot_secondary_structures import count_structures_in_pdb

# :)
SEED = int(
    float.fromhex("2254616977616e2069732061206672656520636f756e74727922") % 10000
)


def len_pdb_structure(fname: str) -> int:
    """Return the integer length of the PDB structure"""
    with open(fname) as source:
        atom_lines = [l.strip() for l in source if l.startswith("ATOM")]
    last_line_tokens = atom_lines[-1].split()
    last_line_l = int(last_line_tokens[5])
    assert int(len(atom_lines) / 3) == last_line_l
    return last_line_l


def build_parser():
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dirname", type=str, help="Directory containing PDB files")
    parser.add_argument("--sctm", type=str, default="", help="scTM scores JSON file")
    parser.add_argument(
        "--trainingtm", type=str, default="", help="Training TM score JSON"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="tmscore_mds",
        help="PDF file prefix to write output to",
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    # Get files
    fnames = sorted(
        glob(os.path.join(args.dirname, "*.pdb")),
        key=lambda x: int_getter(os.path.basename(x)),
    )
    logging.info(f"Computing TMscore on {len(fnames)} structures")

    pdist_df = get_pairwise_tmscores(fnames, sctm_scores_json=args.sctm)
    mds = MDS(n_components=2, dissimilarity="precomputed", n_jobs=-1, random_state=SEED)
    embedding = pd.DataFrame(mds.fit_transform(pdist_df.values), index=pdist_df.index)

    format_strings = {
        "Number helices": "{x:.1f}",
    }
    # For a variety of coloring keys, compute/read the scores and color scatter
    # plot by the scores.
    for k, v in {
        "null": None,
        "Max training TM": args.trainingtm,
        "scTM": args.sctm,
        "length": lambda x: len_pdb_structure(x),
        "Number helices": lambda x: count_structures_in_pdb(x, "psea")[0],
        "Number sheets": lambda x: count_structures_in_pdb(x, "psea")[1],
    }.items():
        if v is None or v:
            logging.info(f"Coloring by {k} scores")
            if v is None:
                scores = None
            elif callable(v):
                fname_to_key = lambda f: os.path.basename(f).split(".")[0]
                scores = {
                    fname_to_key(f): v(f)
                    for f in fnames
                    if fname_to_key(f) in embedding.index
                }
                scores = embedding.index.map(scores)
            elif os.path.isfile(v):
                with open(v) as source:
                    scores = embedding.index.map(json.load(source))
            else:
                raise ValueError(f"Invalid value for {k}: {v}")

            fig, ax = plt.subplots(dpi=300)
            points = ax.scatter(
                embedding.iloc[:, 0],
                embedding.iloc[:, 1],
                s=8,
                c=scores,
                cmap="RdYlBu",
                alpha=0.9,
            )
            ax.set(
                xlabel="MDS 1",
                ylabel="MDS 2",
            )
            if not k == "null":
                ax.set(
                    xticks=[],
                    yticks=[],
                    title=k,
                )
            if scores is not None:
                cbar = plt.colorbar(
                    points,
                    ax=ax,
                    fraction=0.08,
                    pad=0.04,
                    location="right",
                    format=format_strings.get(k, None),
                )
                cbar.ax.set_ylabel(k, fontsize=12)

            fig.savefig(f"{args.output}_mds_{k}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
