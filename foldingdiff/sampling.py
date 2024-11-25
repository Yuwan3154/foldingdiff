"""
Code for sampling from diffusion models
"""
import json
import os
import multiprocessing as mp
from pathlib import Path
import tempfile
import logging
from typing import *
import matplotlib.pyplot as plt
import mpl_scatter_density
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import default_collate, Dataset
from huggingface_hub import snapshot_download

from foldingdiff import datasets as dsets
from foldingdiff import beta_schedules, modelling, utils, sampling, tmalign
from foldingdiff import angles_and_coords as ac

from sklearn.preprocessing import OneHotEncoder
AMINO_ACID_LIST = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']

FT_NAME_MAP = {
    "phi": r"$\phi$",
    "psi": r"$\psi$",
    "omega": r"$\omega$",
    "tau": r"$\theta_1$",
    "CA:C:1N": r"$\theta_2$",
    "C:1N:1CA": r"$\theta_3$",
}
ANGLES_DEFINITIONS = Literal[
    "canonical", "canonical-full-angles", "canonical-minimal-angles", "cart-coords"
]

@torch.no_grad()
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    seq_lens: Sequence[int],
    betas: torch.Tensor,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, length in enumerate(seq_lens):
        attn_mask[i, :length] = 1.0

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x
        - betas_t
        * model(x, t, attention_mask=attn_mask)
        / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    is_angle: Union[bool, List[bool]] = [False, True, True, True],
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    device = next(model.parameters()).device
    b = noise.shape[0]
    img = noise.to(device)
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {noise.shape} with angularity {is_angle} and range {torch.amin(img, dim=(0, 1))} - {torch.amax(img, dim=(0, 1))} using {device}"
    )

    imgs = []

    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
        disable=disable_pbar,
    ):
        # Shape is (batch, seq_len, 4)
        img = p_sample(
            model=model,
            x=img,
            t=torch.full((b,), i, device=device, dtype=torch.long),  # time vector
            seq_lens=lengths,
            betas=betas,
        )

        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                img = utils.modulo_with_wrapped_range(
                    img, range_min=-torch.pi, range_max=torch.pi
                )
        else:
            assert len(is_angle) == img.shape[-1]
            for j in range(img.shape[-1]):
                if is_angle[j]:
                    img[:, :, j] = utils.modulo_with_wrapped_range(
                        img[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        imgs.append(img.cpu())
    return torch.stack(imgs)


def sample(
    model: nn.Module,
    train_dset: dsets.NoisedAnglesDataset,
    n: int = 10,
    sweep_lengths: Optional[Tuple[int, int]] = (50, 128),
    batch_size: int = 512,
    feature_key: str = "angles",
    disable_pbar: bool = False,
    trim_to_length: bool = True,  # Trim padding regions to reduce memory
) -> List[np.ndarray]:
    """
    Sample from the given model. Use the train_dset to generate noise to sample
    sequence lengths. Returns a list of arrays, shape (timesteps, seq_len, fts).
    If sweep_lengths is set, we generate n items per length in the sweep range

    train_dset object must support:
    - sample_noise - provided by NoisedAnglesDataset
    - timesteps - provided by NoisedAnglesDataset
    - alpha_beta_terms - provided by NoisedAnglesDataset
    - feature_is_angular - provided by *wrapped dataset* under NoisedAnglesDataset
    - pad - provided by *wrapped dataset* under NoisedAnglesDataset
    And optionally, sample_length()
    """
    # Process each batch
    if sweep_lengths is not None:
        sweep_min, sweep_max = sweep_lengths
        if not sweep_min < sweep_max:
            raise ValueError(
                f"Minimum length {sweep_min} must be less than maximum {sweep_max}"
            )
        logging.info(
            f"Sweeping from {sweep_min}-{sweep_max} with {n} examples at each length"
        )
        lengths = []
        for l in range(sweep_min, sweep_max):
            lengths.extend([l] * n)
    else:
        lengths = [train_dset.sample_length() for _ in range(n)]
    lengths_chunkified = [
        lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)
    ]

    logging.info(f"Sampling {len(lengths)} items in batches of size {batch_size}")
    retval = []
    for this_lengths in lengths_chunkified:
        batch = len(this_lengths)
        # Sample noise and sample the lengths
        noise = train_dset.sample_noise(
            torch.zeros((batch, train_dset.pad, model.n_inputs), dtype=torch.float32)
        )

        # Trim things that are beyond the length of what we are generating
        if trim_to_length:
            noise = noise[:, : max(this_lengths), :]

        # Produces (timesteps, batch_size, seq_len, n_ft)
        sampled = p_sample_loop(
            model=model,
            lengths=this_lengths,
            noise=noise,
            timesteps=train_dset.timesteps,
            betas=train_dset.alpha_beta_terms["betas"],
            is_angle=train_dset.feature_is_angular[feature_key],
            disable_pbar=disable_pbar,
        )
        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]
        retval.extend(trimmed_sampled)
    # Note that we don't use means variable here directly because we may need a subset
    # of it based on which features are active in the dataset. The function
    # get_masked_means handles this gracefully
    if (
        hasattr(train_dset, "dset")
        and hasattr(train_dset.dset, "get_masked_means")
        and train_dset.dset.get_masked_means() is not None
    ):
        logging.info(
            f"Shifting predicted values by original offset: {train_dset.dset.get_masked_means()}"
        )
        retval = [s + train_dset.dset.get_masked_means() for s in retval]
        # Because shifting may have caused us to go across the circle boundary, re-wrap
        angular_idx = np.where(train_dset.feature_is_angular[feature_key])[0]
        for s in retval:
            s[..., angular_idx] = utils.modulo_with_wrapped_range(
                s[..., angular_idx], range_min=-np.pi, range_max=np.pi
            )

    return retval


def sample_simple(
    model_dir: str, n: int = 10, sweep_lengths: Tuple[int, int] = (50, 128)
) -> List[pd.DataFrame]:
    """
    Simple wrapper on sample to automatically load in the model and dummy dataset
    Primarily for gradio integration
    """
    if utils.is_huggingface_hub_id(model_dir):
        model_dir = snapshot_download(model_dir)
    assert os.path.isdir(model_dir)

    with open(os.path.join(model_dir, "training_args.json")) as source:
        training_args = json.load(source)

    model = modelling.BertForDiffusionBase.from_dir(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    dummy_dset = dsets.AnglesEmptyDataset.from_dir(model_dir)
    dummy_noised_dset = dsets.NoisedAnglesDataset(
        dset=dummy_dset,
        dset_key="coords" if training_args == "cart-cords" else "angles",
        timesteps=training_args["timesteps"],
        exhaustive_t=False,
        beta_schedule=training_args["variance_schedule"],
        nonangular_variance=1.0,
        angular_variance=training_args["variance_scale"],
    )

    sampled = sample(
        model, dummy_noised_dset, n=n, sweep_lengths=sweep_lengths, disable_pbar=True
    )
    final_sampled = [s[-1] for s in sampled]
    sampled_dfs = [
        pd.DataFrame(s, columns=dummy_noised_dset.feature_names["angles"])
        for s in final_sampled
    ]
    return sampled_dfs


@torch.no_grad()
def seq_p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    seq_lens: Sequence[int],
    betas: torch.Tensor,
    n_outputs: int,
) -> torch.Tensor:
    """
    Sample the given timestep. Note that this _may_ fall off the manifold if we just
    feed the output back into itself repeatedly, so we need to perform modulo on it
    (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = beta_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][t_index]

    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, length in enumerate(seq_lens):
        attn_mask[i, :length] = 1.0

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x[..., :n_outputs]
        - betas_t
        * model(x, t, attention_mask=attn_mask)
        / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x[..., :n_outputs])
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def seq_p_sample_loop(
    model: nn.Module,
    lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    aa_seqs_encoded: torch.Tensor,
    is_angle: Union[bool, List[bool]] = [True, True, True, True, True, True],
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    device = next(model.parameters()).device
    b = noise.shape[0]
    n_outputs = noise.shape[-1]
    ft_set = noise.to(device)
    aa_seqs_encoded = aa_seqs_encoded.to(device)
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {noise.shape} with angularity {is_angle} and range {torch.amin(ft_set, dim=(0, 1))} - {torch.amax(ft_set, dim=(0, 1))} using {device}"
    )

    ft_sets = []

    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
        disable=disable_pbar,
    ):
        ft_set = torch.cat((ft_set, aa_seqs_encoded), dim=2)
        # Shape is (batch, seq_len, n_ft)
        ft_set = seq_p_sample(
            model=model,
            x=ft_set,
            t=torch.full((b,), i, device=device, dtype=torch.long),  # time vector
            seq_lens=lengths,
            betas=betas,
            n_outputs=n_outputs,
        )

        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                ft_set = utils.modulo_with_wrapped_range(
                    ft_set, range_min=-torch.pi, range_max=torch.pi
                )
        else:
            for j in range(ft_set.shape[-1]):
                if is_angle[j]:
                    ft_set[:, :, j] = utils.modulo_with_wrapped_range(
                        ft_set[:, :, j], range_min=-torch.pi, range_max=torch.pi
                    )
        ft_sets.append(ft_set.cpu())
    return torch.stack(ft_sets)


def sample_fragment(
    model: nn.Module,
    dset: dsets.NoisedAnglesDataset,
    timesteps: int = None,
    aa_seqs: List[List[str]] = None,
    n: int = 10,
    batch_size: int = 512,
    feature_key: str = "angles",
    disable_pbar: bool = False,
    trim_to_length: bool = True,  # Trim padding regions to reduce memory
) -> List[np.ndarray]:
    """
    Sample from the given model. Use the dset to generate noise to sample
    sequence lengths. Returns a list of arrays, shape (timesteps, seq_len, fts).
    If sweep_lengths is set, we generate n items per length in the sweep range

    dset object must support:
    - sample_noise - provided by NoisedAnglesDataset
    - timesteps - provided by NoisedAnglesDataset
    - alpha_beta_terms - provided by NoisedAnglesDataset
    - feature_is_angular - provided by *wrapped dataset* under NoisedAnglesDataset
    - pad - provided by *wrapped dataset* under NoisedAnglesDataset
    And optionally, sample_length()
    """
    # Check if aa_seqs is provided
    if aa_seqs is not None:
        lengths = [len(seq) for seq in aa_seqs for _ in range(n)]

        encoder = OneHotEncoder(categories=[AMINO_ACID_LIST])
        aa_seqs_encoded = torch.from_numpy(
            encoder.fit_transform(
                np.array([aa_seq for aa_seq in aa_seqs for _ in range(n)]).reshape(-1, 1)
            ).toarray()
        ).float()
    elif not isinstance(dset.dset, dsets.AnglesEmptyDataset):
        lengths = [dset.pad for __ in range(len(dset)) for _ in range(n)]
        aa_idx = [dset.dset.feature_names["angles"].index(aa) for aa in AMINO_ACID_LIST]
        aa_seqs_encoded = torch.cat(
            [
                dset.dset[i]["angles"].iloc[:, aa_idx]
                for i in range(len(dset.dset)) for _ in range(n)
            ], dim=0
            )
    else:
        raise ValueError("Either aa_seqs or dset must be provided")

    timesteps = dset.timesteps if timesteps is None else timesteps
    length = np.unique(lengths)
    assert length.shape[0] == 1, "All sequences must be the same length"
    length = length[0]

    lengths_chunkified = [
        lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)
    ]
    aa_seqs_encoded_chunkified = [
        aa_seqs_encoded[i : i + batch_size*length] 
        for i in range(0, len(aa_seqs_encoded), batch_size*length)
    ]
    n_outputs = dset.dset.noise_mask[dset.dset_key].count_nonzero().item()

    logging.info(f"Sampling {len(lengths)} items in batches of size {batch_size}")
    retval = []
    for this_lengths, this_aa_seqs_encoded in zip(lengths_chunkified, aa_seqs_encoded_chunkified):
        batch = len(this_lengths)
        # Sample noise and sample the lengths
        noise = dset.sample_noise(
            torch.zeros((batch, dset.pad, n_outputs), dtype=torch.float32)
        )
        # Reshape the one-hot encoded amino acid sequence to be used as an input to the model
        reshaped_aa_seqs_encoded = this_aa_seqs_encoded.reshape(batch, dset.pad, -1)

        # Trim things that are beyond the length of what we are generating
        if trim_to_length:
            noise = noise[:, : max(this_lengths), :]

        # Produces (timesteps, batch_size, seq_len, n_ft)
        sampled = seq_p_sample_loop(
            model=model,
            lengths=this_lengths,
            noise=noise,
            timesteps=timesteps,
            betas=dset.alpha_beta_terms["betas"],
            aa_seqs_encoded=reshaped_aa_seqs_encoded,
            is_angle=dset.feature_is_angular,
            disable_pbar=disable_pbar,
        )
        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]
        retval.extend(trimmed_sampled)
    # Note that we don't use means variable here directly because we may need a subset
    # of it based on which features are active in the dataset. The function
    # get_masked_means handles this gracefully
    if (
        hasattr(dset, "dset")
        and hasattr(dset.dset, "get_masked_means")
        and dset.dset.get_masked_means() is not None
    ):
        logging.info(
            f"Shifting predicted values by original offset: {dset.dset.get_masked_means()}"
        )
        retval = [s + dset.dset.get_masked_means() for s in retval]
        # Because shifting may have caused us to go across the circle boundary, re-wrap
        angular_idx = np.where(dset.feature_is_angular)[0]
        for s in retval:
            s[..., angular_idx] = utils.modulo_with_wrapped_range(
                s[..., angular_idx], range_min=-np.pi, range_max=np.pi
            )
    return retval


def load_model_and_dset(
    model_dir: Path,
    model_snapshot_dir: Path = "",
    device: str = "cpu",
    timesteps: int = None,
    load_actual: bool = True
    ):
    # Load the model
    model = modelling.BertForDiffusionBase.from_dir(
        model_dir, copy_to=model_snapshot_dir
    ).to(device)

    with open(model_dir / "training_args.json") as source:
            training_args = json.load(source)

    timesteps = training_args["timesteps"] if timesteps is None else timesteps
    # Load the dataset based on training args
    if load_actual:
        dset_args = dict(
            timesteps=timesteps,
            variance_schedule=training_args["variance_schedule"],
            max_seq_len=training_args["max_seq_len"],
            min_seq_len=training_args["min_seq_len"],
            var_scale=training_args["variance_scale"],
            syn_noiser=training_args["syn_noiser"],
            exhaustive_t=training_args["exhaustive_validation_t"],
            single_angle_debug=training_args["single_angle_debug"],
            single_time_debug=training_args["single_timestep_debug"],
            toy=training_args["subset"],
            angles_definitions=training_args["angles_definitions"],
            dataset_key=training_args["dataset_key"],
            train_only=False,
            seq_trim_strategy=training_args["trim_strategy"],
        )

        train_dset, valid_dset, test_dset = get_train_valid_test_sets(**dset_args)
        logging.info(
            f"Training dset contains features: {train_dset.feature_names} - angular {train_dset.feature_is_angular}"
        )
        return model, train_dset, valid_dset, test_dset
    else:
        # Build args based on training args
        mean_file = model_dir / "training_mean_offset.npy"
        placeholder_dset = dsets.AnglesEmptySequenceDataset(
            feature_set_key=training_args["angles_definitions"],
            pad=training_args["max_seq_len"],
            mean_offset=None if not mean_file.exists() else np.load(mean_file),
        )
        dummy_dsets = [dsets.NoisedAnglesDataset(
                dset=placeholder_dset,
                dset_key="coords"
                if training_args["angles_definitions"] == "cart-coords"
                else "angles",
                timesteps=timesteps,
                exhaustive_t=False,
                beta_schedule=training_args["variance_schedule"],
                nonangular_variance=1.0,
                angular_variance=training_args["variance_scale"],
            ) for _ in range(3)]

    return model, dummy_dsets[0], dummy_dsets[1], dummy_dsets[2]


def get_train_valid_test_sets(
    dataset_key: str = "cath",
    angles_definitions: ANGLES_DEFINITIONS = "canonical-full-angles",
    max_seq_len: int = 512,
    min_seq_len: int = 0,
    seq_trim_strategy: dsets.TRIM_STRATEGIES = "leftalign",
    timesteps: int = 250,
    variance_schedule: beta_schedules.SCHEDULES = "linear",
    var_scale: float = np.pi,
    toy: Union[int, bool] = False,
    exhaustive_t: bool = False,
    syn_noiser: str = "",
    single_angle_debug: int = -1,  # Noise and return a single angle. -1 to disable, 1-3 for omega/theta/phi
    single_time_debug: bool = False,  # Noise and return a single time
    train_only: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Get the dataset objects to use for train/valid/test

    Note, these need to be wrapped in data loaders later
    """
    assert (
        single_angle_debug != 0
    ), f"Invalid value for single_angle_debug: {single_angle_debug}"

    clean_dset_class = {
        "canonical": dsets.CathCanonicalAnglesDataset,
        "canonical-full-angles": dsets.CathCanonicalAnglesOnlyDataset,
        "canonical-minimal-angles": dsets.CathCanonicalMinimalAnglesDataset,
        "cart-coords": dsets.CathCanonicalCoordsDataset,
        "canonical-full-angles-sequence": dsets.CathCanonicalAnglesSequenceDataset,
        "idealized-dihedral-secondary-structure-sequence": dsets.IdealizedDihedralSecondaryStructureSequenceDataset,
        "idealized-dihedral-sequence": dsets.IdealizedDihedralSequenceDataset,
        "fold-balanced-dihedral-sequence": dsets.FoldBalancedDihedralSequenceDataset,
        "fold-balanced-dihedral": dsets.FoldBalancedDihedralDataset,
        "fold-balanced-dihedral-secondary-structure-sequence": dsets.FoldBalancedDihedralSecondaryStructureSequenceDataset,
    }[angles_definitions]
    logging.info(f"Clean dataset class: {clean_dset_class}")

    splits = ["train"] if train_only else ["train", "validation", "test"]
    logging.info(f"Creating data splits: {splits}")
    clean_dsets = [
        clean_dset_class(
            pdbs=dataset_key,
            split=s,
            pad=max_seq_len,
            min_length=min_seq_len,
            trim_strategy=seq_trim_strategy,
            zero_center=False if angles_definitions == "cart-coords" else True,
            toy=toy,
        )
        for s in splits
    ]
    assert len(clean_dsets) == len(splits)
    # Set the training set mean to the validation set mean
    if len(clean_dsets) > 1 and clean_dsets[0].means is not None:
        logging.info(f"Updating valid/test mean offset to {clean_dsets[0].means}")
        for i in range(1, len(clean_dsets)):
            clean_dsets[i].means = clean_dsets[0].means

    if syn_noiser != "":
        if syn_noiser == "halfhalf":
            logging.warning("Using synthetic half-half noiser")
            dset_noiser_class = dsets.SynNoisedByPositionDataset
        else:
            raise ValueError(f"Unknown synthetic noiser {syn_noiser}")
    else:
        if single_angle_debug > 0:
            logging.warning("Using single angle noise!")
            dset_noiser_class = functools.partial(
                dsets.SingleNoisedAngleDataset, ft_idx=single_angle_debug
            )
        elif single_time_debug:
            logging.warning("Using single angle and single time noise!")
            dset_noiser_class = dsets.SingleNoisedAngleAndTimeDataset
        else:
            dset_noiser_class = dsets.NoisedAnglesDataset

    logging.info(f"Using {dset_noiser_class} for noise")
    noised_dsets = [
        dset_noiser_class(
            dset=ds,
            dset_key="coords" if angles_definitions == "cart-coords" else "angles",
            timesteps=timesteps,
            exhaustive_t=(i != 0) and exhaustive_t,
            beta_schedule=variance_schedule,
            nonangular_variance=1.0,
            angular_variance=var_scale,
        )
        for i, ds in enumerate(clean_dsets)
    ]
    for dsname, ds in zip(splits, noised_dsets):
        logging.info(f"{dsname}: {ds}")

    # Pad with None values
    if len(noised_dsets) < 3:
        noised_dsets = noised_dsets + [None] * int(3 - len(noised_dsets))
    assert len(noised_dsets) == 3

    return tuple(noised_dsets)


def plot_ramachandran(
    phi_values,
    psi_values,
    fname: str,
    annot_ss: bool = False,
    title: str = "",
    plot_type: Literal["kde", "density_heatmap"] = "density_heatmap",
):
    """Create Ramachandran plot for phi_psi"""
    if plot_type == "kde":
        fig = plotting.plot_joint_kde(
            phi_values,
            psi_values,
        )
        ax = fig.axes[0]
        ax.set_xlim(-3.67, 3.67)
        ax.set_ylim(-3.67, 3.67)
    elif plot_type == "density_heatmap":
        fig = plt.figure(dpi=800)
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        norm = ImageNormalize(vmin=0.0, vmax=650, stretch=LogStretch())
        ax.scatter_density(phi_values, psi_values, norm=norm, cmap=plt.cm.Blues)
    else:
        raise NotImplementedError(f"Cannot plot type: {plot_type}")
    if annot_ss:
        # https://matplotlib.org/stable/tutorials/text/annotations.html
        ram_annot_arrows = dict(
            facecolor="black", shrink=0.05, headwidth=6.0, width=1.5
        )
        ax.annotate(
            r"$\alpha$ helix, LH",
            xy=(1.2, 0.5),
            xycoords="data",
            xytext=(1.7, 1.2),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=14,
        )
        ax.annotate(
            r"$\alpha$ helix, RH",
            xy=(-1.1, -0.6),
            xycoords="data",
            xytext=(-1.7, -1.9),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=14,
        )
        ax.annotate(
            r"$\beta$ sheet",
            xy=(-1.67, 2.25),
            xycoords="data",
            xytext=(-0.9, 2.9),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=14,
        )
    ax.set_xlabel("$\phi$ (radians)", fontsize=14)
    ax.set_ylabel("$\psi$ (radians)", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    fig.savefig(fname, bbox_inches="tight")

def plot_distribution_overlap(
    values_dicts: Dict[str, np.ndarray],
    title: str = "Sampled distribution",
    fname: str = "",
    bins: int = 50,
    ax=None,
    show_legend: bool = True,
    **kwargs,
):
    """
    Plot the distribution overlap between the training and sampled values
    Additional arguments are given to ax.hist; for example, can specify
    histtype='step', cumulative=True
    to get a CDF plot
    """
    # Plot the distribution overlap
    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    for k, v in values_dicts.items():
        if v is None:
            continue
        _n, bins, _pbatches = ax.hist(
            v,
            bins=bins,
            label=k,
            density=True,
            **kwargs,
        )
    if title:
        ax.set_title(title, fontsize=16)
    if show_legend:
        ax.legend()
    if fname:
        fig.savefig(fname, bbox_inches="tight")

def _score_angles(
    reconst_angles: pd.DataFrame, truth_angles: pd.DataFrame, truth_coords_pdb: str
) -> Tuple[float, float]:
    """
    Helper function to scores sets of angles
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        truth_path = Path(tmpdir) / "truth.pdb"
        reconst_path = Path(tmpdir) / "reconst.pdb"

        truth_pdb = ac.create_new_chain_nerf(str(truth_path), truth_angles)
        reconst_pdb = ac.create_new_chain_nerf(str(reconst_path), reconst_angles)

        # Calculate WRT the truth angles
        score = tmalign.run_tmalign(reconst_pdb, truth_pdb)

        score_coord = tmalign.run_tmalign(reconst_pdb, truth_coords_pdb)
    return score, score_coord


@torch.no_grad()
def get_reconstruction_error(
    model: nn.Module, dset, noise_timesteps: int = 250, bs: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the reconstruction error when adding <noise_timesteps> noise to the idx-th
    item in the dataset.
    """
    device = next(model.parameters()).device
    model.eval()

    recont_angle_sets = []
    truth_angle_sets = []
    truth_pdb_files = []
    for idx_batch in tqdm(utils.seq_to_groups(list(range(len(dset))), bs)):
        batch = default_collate(
            [
                {
                    k: v.to(device)
                    for k, v in dset.__getitem__(idx, use_t_val=noise_timesteps).items()
                }
                for idx in idx_batch
            ]
        )
        img = batch["corrupted"].clone()
        assert img.ndim == 3

        # Record the actual files containing raw coordinates
        for i in idx_batch:
            truth_pdb_files.append(dset.filenames[i])

        # Run the diffusion model for noise_timesteps steps
        for i in tqdm(list(reversed(list(range(0, noise_timesteps))))):
            img = sampling.p_sample(
                model=model,
                x=img,
                t=torch.full((len(idx_batch),), fill_value=i, dtype=torch.long).to(
                    device
                ),
                seq_lens=batch["lengths"],
                betas=dset.alpha_beta_terms["betas"],
            )
            img = utils.modulo_with_wrapped_range(img)

        # Finished reconstruction, subset to lengths and add to running list
        for i, l in enumerate(batch["lengths"].squeeze()):
            recont_angle_sets.append(
                pd.DataFrame(img[i, :l].cpu().numpy(), columns=ac.EXHAUSTIVE_ANGLES)
            )
            truth_angle_sets.append(
                pd.DataFrame(
                    batch["angles"][i, :l].cpu().numpy(), columns=ac.EXHAUSTIVE_ANGLES
                )
            )

    # Get the reconstruction error as a TM score
    logging.info(
        f"Calculating TM scores for reconstruction error with {mp.cpu_count()} processes"
    )
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.starmap(
        _score_angles,
        zip(recont_angle_sets, truth_angle_sets, truth_pdb_files),
        chunksize=10,
    )
    pool.close()
    pool.join()
    scores, coord_scores = zip(*results)
    return np.array(scores), np.array(coord_scores)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = sample_simple("wukevin/foldingdiff_cath", n=1, sweep_lengths=(50, 51))
    for i, x in enumerate(s):
        print(x.shape)
        print(x)
