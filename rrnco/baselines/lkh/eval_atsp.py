import argparse
import os
import pprint as pp
import time
import warnings

from multiprocessing import Pool

import lkh  # Make sure you have lkh-python installed
import numpy as np
import tqdm
import tsplib95

warnings.filterwarnings("ignore")


def solve_atsp_with_lkh(data, lkh_path, max_trials):
    """Solve the ATSP using the LKH solver."""
    scale = 1e3
    num_nodes = data.shape[0]
    problem = tsplib95.models.StandardProblem()
    problem.name = "ATSP"
    problem.type = "ATSP"
    problem.dimension = num_nodes
    problem.edge_weight_type = "EXPLICIT"
    problem.edge_weight_format = (
        "FULL_MATRIX"  # Add this line to specify the edge weight format
    )

    # Define edge weights explicitly for ATSP
    problem.edge_weights = (data * scale).astype(int).tolist()

    # Solve the problem using LKH
    solution = lkh.solve(lkh_path, problem=problem, max_trials=max_trials, runs=10)

    tour = [n - 1 for n in solution[0]]  # Convert 1-based to 0-based indexing
    cost = sum(data[tour[i - 1], tour[i]] for i in range(len(tour)))

    return cost, tour


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="../../../data/atsp/atsp_n100_seed3333_out_of_distribution.npz",
    )
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--lkh_path", type=str, default="LKH-3.0.13/LKH")
    parser.add_argument("--lkh_trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--offset", type=int, default=0)
    opts = parser.parse_args()

    assert (
        opts.num_samples % opts.batch_size == 0
    ), "Number of samples must be divisible by batch size"
    np.random.seed(opts.seed)
    filename = opts.data_path.split("/")[-1].split(".")[0]
    if opts.filename is None:
        opts.filename = f"results/atsp{opts.num_nodes}_{filename}_lkh.txt"

    # Ensure results folder exists
    os.makedirs(os.path.dirname(opts.filename), exist_ok=True)

    # Pretty print the run args
    pp.pprint(vars(opts))
    cost_lst = []

    with open(opts.data_path, "rb") as f:
        data = np.load(opts.data_path)["distance_matrix"]
        assert (
            len(data) >= opts.num_samples
        ), "Dataset has fewer samples than --num_samples"
        sub_data = data[opts.offset : opts.num_samples]

    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
            batch_data = sub_data[b_idx * opts.batch_size : (b_idx + 1) * opts.batch_size]

            with Pool(opts.batch_size) as p:
                costs_and_tours = p.starmap(
                    solve_atsp_with_lkh,
                    [
                        (batch_data[idx], opts.lkh_path, opts.lkh_trials)
                        for idx in range(opts.batch_size)
                    ],
                )

            for idx, (cost, tour) in enumerate(costs_and_tours):
                f.write(
                    f"Sample {b_idx * opts.batch_size + idx}: Cost = {cost}, Tour = {tour}\n"
                )
                cost_lst.append(cost)

        end_time = time.time() - start_time

    print(f"Completed generation of {opts.num_samples} samples of ATSP{opts.num_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / (opts.num_samples // opts.batch_size):.1f}s")
    print(f"Average cost: {np.mean(cost_lst):.2f}")
