"""
Run heirarchical clustering on the pairwise distance matrix between all pairs of files
"""

import os, sys
import re
import json
import logging
from glob import glob
from pathlib import Path
import itertools
import argparse
from typing import *
import multiprocessing as mp

import numpy as np
import pandas as pd
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import seaborn as sns

SRC_DIR = (Path(os.path.dirname(os.path.abspath(__file__))) / "../protdiff").resolve()
assert SRC_DIR.is_dir()
sys.path.append(str(SRC_DIR))
import tmalign


def int_getter(x: str) -> int:
    """Fetches integer value out of a string"""
    matches = re.findall(r"[0-9]+", x)
    assert len(matches) == 1
    return int(matches.pop())


def get_pairwise_tmscores(
    dirname: Collection[str], sctm_scores_json: Optional[str] = None
) -> pd.DataFrame:
    """Get the pairwise TM scores across all fnames"""
    fnames = sorted(
        glob(os.path.join(dirname, "*.pdb")),
        key=lambda x: int_getter(os.path.basename(x)),
    )
    assert fnames, f"{dirname} does not contain any pdb files"
    logging.info(f"Found {len(fnames)} pdb files to compute pairwise distances")

    bname_getter = lambda x: os.path.splitext(os.path.basename(x))[0]
    if sctm_scores_json:
        with open(sctm_scores_json) as source:
            sctm_scores = json.load(source)
        fnames = [f for f in fnames if sctm_scores[bname_getter(f)] >= 0.5]
        logging.info(f"{len(fnames)} structures have scTM scores >= 0.5")

    # for debugging
    # fnames = fnames[:50]

    pairs = list(itertools.combinations(fnames, 2))
    pool = mp.Pool(mp.cpu_count())
    values = list(pool.starmap(tmalign.run_tmalign, pairs, chunksize=25))
    pool.close()
    pool.join()

    bnames = [bname_getter(f) for f in fnames]
    retval = pd.DataFrame(1.0, index=bnames, columns=bnames)
    for (k, v), val in zip(pairs, values):
        retval.loc[bname_getter(k), bname_getter(v)] = val
        retval.loc[bname_getter(v), bname_getter(k)] = val
    assert np.allclose(retval, retval.T)
    return retval


def build_parser():
    """Build a CLI parser"""
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dirname", type=str, help="Directory of PDB files to analyze")
    parser.add_argument(
        "--sctm", type=str, required=False, default="", help="scTM scores to filter by"
    )
    return parser


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    # TMscore of 1 = perfect match --> 0 distance, so need 1.0 - tmscore
    pdist_df = 1.0 - get_pairwise_tmscores(args.dirname, sctm_scores_json=args.sctm)

    # https://stackoverflow.com/questions/38705359/how-to-give-sns-clustermap-a-precomputed-distance-matrix
    # https://stackoverflow.com/questions/57308725/pass-distance-matrix-to-seaborn-clustermap
    m = "average"  # Trippe uses average here
    linkage = hc.linkage(
        sp.distance.squareform(pdist_df), method=m, optimal_ordering=False
    )

    c = sns.clustermap(
        pdist_df,
        row_linkage=linkage,
        col_linkage=linkage,
        method=None,
        row_cluster=True,
        col_cluster=True,
        vmin=0.0,
        vmax=1.0,
        xticklabels=False,
        yticklabels=False,
    )
    c.savefig(os.path.join(args.dirname, "tmscore_hclust.pdf"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
