import argparse
import os
import time
import warnings

import torch
from rl4co.data.dataset import TensorDictDataset
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.ops import batchify, unbatchify
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rrnco.baselines.routefinder.model import RouteFinderBase
from rrnco.envs.atsp import ATSPEnv
from rrnco.envs.rcvrp import RCVRPEnv
from rrnco.envs.rmtvrp import RMTVRPEnv
from rrnco.models import RRNet
from rrnco.models.utils.transforms import StateAugmentation

augment = StateAugmentation(augment_fn="dihedral8", no_aug_coords=False)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# no rankzero warning
warnings.filterwarnings("ignore", category=UserWarning)


# Tricks for faster inference
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass
torch.set_float32_matmul_precision("medium")


def get_dataloader(td, batch_size=4):
    """Get a dataloader from a TensorDictDataset"""
    # Set up the dataloader
    dataloader = DataLoader(
        TensorDictDataset(td.clone()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TensorDictDataset.collate_fn,
    )
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem", type=str, default="atsp", help="Problem name: hcvrp, omdcpdp, etc."
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate. Defaults to all under data/{problem}/ dir",
        default=None,
    )
    parser.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        help="Decoding type. Available only: greedy",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--problem_size", type=int, default=100)
    # Use load_from_checkpoint with map_location, which is handled internally by Lightning
    # Suppress FutureWarnings related to torch.load and weights_only
    warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)

    opts = parser.parse_args()
    generator_params = {"num_loc": opts.problem_size}
    batch_size = opts.batch_size
    decode_type = opts.decode_type
    checkpoint_path = opts.checkpoint
    problem = opts.problem
    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if checkpoint_path is None:
        assert (
            problem is not None
        ), "Problem must be specified if checkpoint is not provided"
        checkpoint_path = f"./checkpoints/rrnet/{problem}/epoch_199.ckpt"
    if opts.datasets is None:
        assert problem is not None, "Problem must be specified if dataset is not provided"
        data_paths = [f"./data/{problem}/{f}" for f in os.listdir(f"./data/{problem}")]
    else:
        data_paths = [opts.datasets] if isinstance(opts.datasets, str) else opts.datasets
    data_paths = sorted(data_paths)  # Sort for consistency

    if opts.no_aug:
        n_aug = 1
    else:
        n_aug = 8
    if problem == "atsp" or problem == "rcvrptw" or "routefinder" in checkpoint_path:
        n_start = 100
    else:
        n_start = 101
    # Load the checkpoint as usual
    print("Loading checkpoint from ", checkpoint_path)

    # monkey patch for RF-based models
    if "routefinder" in checkpoint_path:
        RRNet = RouteFinderBase

    model = RRNet.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=False, load_baseline=False
    )
    policy = model.policy.to(device).eval()  # Use mixed precision if supported

    for dataset in data_paths:
        costs = []
        inference_times = []

        print(f"Loading {dataset}")

        td_test = load_npz_to_tensordict(dataset)

        match problem:
            case "atsp":
                env = ATSPEnv(check_solution=False, generator_params=generator_params)
            case "rcvrptw":
                env = RMTVRPEnv(check_solution=False, generator_params=generator_params)
            case "rcvrp":
                if "routefinder" in checkpoint_path:
                    td_test["demand_linehaul"] = td_test["demand"] / td_test[
                        "capacity"
                    ].unsqueeze(-1)
                    td_test["capacity"] = torch.ones_like(td_test["capacity"])
                    td_test["locs"] = torch.cat(
                        [td_test["depot"][..., None, :], td_test["locs"]], dim=-2
                    )
                    env = RMTVRPEnv(
                        check_solution=False, generator_params=generator_params
                    )
                else:
                    td_test["demand"] = td_test["demand"] / td_test["capacity"].unsqueeze(
                        -1
                    )
                    td_test["capacity"] = torch.ones_like(td_test["capacity"])
                    env = RCVRPEnv(
                        check_solution=False, generator_params=generator_params
                    )
            case _:
                raise ValueError(f"Problem {problem} not supported")
        dataloader = get_dataloader(td_test, batch_size=batch_size)
        with (
            torch.autocast("cuda") if "cuda" in opts.device else torch.inference_mode()
        ):  # Use mixed precision if supported
            with torch.inference_mode():
                for td_test_batch in tqdm(dataloader):
                    if not opts.no_aug:
                        td_test_batch = augment(td_test_batch)
                    td_reset = env.reset(td_test_batch).to(device)

                    start_time = time.time()
                    out = policy(
                        td_reset,
                        env,
                        return_actions=True,
                        phase="val",
                        calc_reward=False,
                        num_starts=n_start,
                    )
                    td_batch = batchify(
                        td_reset, n_start
                    )  # Expand td to batch_size * num_starts to calc. reward
                    if env.normalize:
                        real_r, norm_r = env.get_reward(td_batch, out["actions"])
                        reward = real_r
                    else:
                        reward = env.get_reward(td_batch, out["actions"])
                    end_time = time.time()
                    inference_time = end_time - start_time
                    max_reward = (
                        unbatchify(reward, (n_aug, n_start)).max(dim=-1)[0].max(dim=-1)[0]
                    )
                    costs.append(max_reward.mean().item())
                    inference_times.append(inference_time)

            print(f"Average cost:\n{-sum(costs)/len(costs):.4f}")
            print(
                f"Per step inference time (s):\n{sum(inference_times)/len(inference_times):.4f}"
            )
            print(f"Total inference time (s):\n{sum(inference_times):.4f}")
