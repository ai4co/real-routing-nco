import argparse
import logging
import os

import numpy as np
import orjson

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


VARIANT_FEATURES = {
    "CVRP": {"O": False, "TW": False, "L": False, "B": False, "M": False},
    "OVRP": {"O": True, "TW": False, "L": False, "B": False, "M": False},
    "VRPB": {"O": False, "TW": False, "L": False, "B": True, "M": False},
    "VRPL": {"O": False, "TW": False, "L": True, "B": False, "M": False},
    "VRPTW": {"O": False, "TW": True, "L": False, "B": False, "M": False},
    "OVRPTW": {"O": True, "TW": True, "L": False, "B": False, "M": False},
    "OVRPB": {"O": True, "TW": False, "L": False, "B": True, "M": False},
    "OVRPL": {"O": True, "TW": False, "L": True, "B": False, "M": False},
    "VRPBL": {"O": False, "TW": False, "L": True, "B": True, "M": False},
    "VRPBTW": {"O": False, "TW": True, "L": False, "B": True, "M": False},
    "VRPLTW": {"O": False, "TW": True, "L": True, "B": False, "M": False},
    "OVRPBL": {"O": True, "TW": False, "L": True, "B": True, "M": False},
    "OVRPBTW": {"O": True, "TW": True, "L": False, "B": True, "M": False},
    "OVRPLTW": {"O": True, "TW": True, "L": True, "B": False, "M": False},
    "VRPBLTW": {"O": False, "TW": True, "L": True, "B": True, "M": False},
    "OVRPBLTW": {"O": True, "TW": True, "L": True, "B": True, "M": False},
    "VRPMB": {"O": False, "TW": False, "L": False, "B": True, "M": True},
    "OVRPMB": {"O": True, "TW": False, "L": False, "B": True, "M": True},
    "VRPMBL": {"O": False, "TW": False, "L": True, "B": True, "M": True},
    "VRPMBTW": {"O": False, "TW": True, "L": False, "B": True, "M": True},
    "OVRPMBL": {"O": True, "TW": False, "L": True, "B": True, "M": True},
    "OVRPMBTW": {"O": True, "TW": True, "L": False, "B": True, "M": True},
    "VRPMBLTW": {"O": False, "TW": True, "L": True, "B": True, "M": True},
    "OVRPMBLTW": {"O": True, "TW": True, "L": True, "B": True, "M": True},
}

# Constants
CAPACITIES = {
    10: 20.0,
    15: 25.0,
    20: 30.0,
    30: 33.0,
    40: 37.0,
    50: 40.0,
    60: 43.0,
    75: 45.0,
    100: 50.0,
    125: 55.0,
    150: 60.0,
    200: 70.0,
    500: 100.0,
    1000: 150.0,
}


# Utility Functions
def load_cities_list(data_dir, in_distribution):
    """Load cities list from the specified path."""

    data_path = os.path.join(data_dir, "splited_cities_list.json")
    with open(data_path, "r") as f:
        cities_list = orjson.loads(f.read())
        if in_distribution:
            return cities_list["train"]
        return cities_list["test"]


def sample_data(data_dir, cities_list, dataset_size, graph_size, dist_type="uniform"):
    """Sample data for a given cities list, dataset size and graph size."""

    num_cities = len(cities_list)
    dataset_size_per_city = dataset_size // num_cities
    sampled_data = {}

    for i, city in enumerate(cities_list):
        data = np.load(f"{data_dir}/{city}/{city}_data.npz", allow_pickle=True)
        data_length = len(data["points"])
        if graph_size > data_length:
            raise ValueError(
                f"graph_size ({graph_size}) exceeds the available data size ({data_length})."
            )
        if data["distance"].max() > 1e5:
            outlier_indices = np.where(data["distance"] > 1e5)
            for i in range(data_length):
                num_problem_points = (outlier_indices[0] == i).sum()
                if num_problem_points > 0 and num_problem_points < data_length // 2:
                    problem_indices_r = outlier_indices[1][outlier_indices[0] == i]
                    break

            for i in range(data_length):
                num_problem_points = (outlier_indices[1] == i).sum()
                if num_problem_points < data_length // 2:
                    problem_indices_c = outlier_indices[0][outlier_indices[1] == i]
                    break
            problem_indices = np.concatenate([problem_indices_r, problem_indices_c])
            new_indices = np.delete(np.arange(data_length), problem_indices)
            points = data["points"][new_indices]
            distance = data["distance"][new_indices][:, new_indices]
            duration = data["duration"][new_indices][:, new_indices]
            data_length = len(points)
            data = {"points": points, "distance": distance, "duration": duration}
        if dist_type == "uniform":
            indices = np.array(
                [
                    np.random.choice(data_length, graph_size, replace=False)
                    for _ in range(dataset_size_per_city)
                ]
            )
        elif dist_type == "cluster":
            indices = single_cluster_sample(
                data["points"], dataset_size_per_city, data_length, graph_size
            )
        for key, values in data.items():
            if key == "distance" or key == "duration":
                if sampled_data.get(key, None) is None:
                    sampled_data[key] = values[indices[:, :, None], indices[:, None, :]]
                else:
                    sampled_data[key] = np.concatenate(
                        [
                            sampled_data[key],
                            values[indices[:, :, None], indices[:, None, :]],
                        ],
                        axis=0,
                    )
            else:
                if sampled_data.get(key, None) is None:
                    sampled_data[key] = values[indices]
                else:
                    sampled_data[key] = np.concatenate(
                        [sampled_data[key], values[indices]], axis=0
                    )
    return sampled_data


def single_cluster_sample(points, dataset_size_per_city, data_length, graph_size):
    # dataset_size_per_city만큼 반복하면서 클러스터 샘플링 수행
    indices = np.array(
        [
            # 각 반복마다 새로운 중심점을 선택하고 클러스터 샘플링
            np.argsort(
                np.linalg.norm(points - points[np.random.choice(data_length)], axis=1)
            )[:graph_size]
            for _ in range(dataset_size_per_city)
        ]
    )
    return indices


def normalize_points(points):
    """Normalize point coordinates batch-wise."""
    points_min = np.min(points, axis=1, keepdims=True)
    points_max = np.max(points, axis=1, keepdims=True)
    return (points - points_min) / (points_max - points_min)


def normalize_duration(duration):
    """Normalize duration matrix."""
    # Compute batch-wise min and max
    duration_min = np.min(duration, axis=(1, 2), keepdims=True)  # Shape: [B, 1, 1]
    duration_max = np.max(duration, axis=(1, 2), keepdims=True)  # Shape: [B, 1, 1]

    # Avoid division by zero in case max == min
    denom = np.where(duration_max - duration_min == 0, 1, duration_max - duration_min)

    # Normalize
    normalized_duration = (duration - duration_min) / denom

    return normalized_duration


def prepare_rcvrptw_data(sampled_data, dataset_size, graph_size):
    """Prepare RCVRP-specific data."""
    locs = normalize_points(sampled_data["points"])
    normalized_duration = normalize_duration(sampled_data["duration"])
    data = generate_mtvrp_data(
        dataset_size=dataset_size,
        num_loc=graph_size,
        capacity=None,
        min_demand=1,
        max_demand=9,
        scale_demand=True,
        max_time=4.6,
        duration_matrix=normalized_duration,
        variant="VRPTW",
    )

    data.update(
        {
            "locs": locs.astype(np.float32),
            "distance_matrix": sampled_data["distance"].astype(np.float32),
            "duration_matrix": normalized_duration.astype(np.float32),
        }
    )
    return data


def prepare_rcvrp_data(sampled_data, dataset_size, graph_size):
    """Prepare RCVRP-specific data."""
    locs = normalize_points(sampled_data["points"])
    depot = locs[:, 0, :]
    locs = locs[:, 1:, :]
    demands = np.random.randint(1, 10, size=(dataset_size, graph_size))
    capacity = np.full(dataset_size, CAPACITIES[graph_size], dtype=np.float32)

    return {
        "depot": depot.astype(np.float32),
        "locs": locs.astype(np.float32),
        "demand": demands.astype(np.float32),
        "capacity": capacity,
        "distance_matrix": sampled_data["distance"].astype(np.float32),
    }


def prepare_atsp_data(sampled_data):
    """Prepare ATSP-specific data."""
    locs = normalize_points(sampled_data["points"])
    return {
        "locs": locs.astype(np.float32),
        "distance_matrix": sampled_data["distance"].astype(np.float32),
    }


def get_vehicle_capacity(num_loc):
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap


def generate_mtvrp_data(
    dataset_size,
    num_loc=100,
    min_loc=0,
    max_loc=1,
    capacity=None,
    min_demand=1,
    max_demand=9,
    scale_demand=True,
    max_time=4.6,
    max_distance_limit=2.8,  # 2sqrt(2) ~= 2.8
    speed=1.0,
    duration_matrix=None,
    variant="VRPTW",
):
    """Generate MTVRP data using NumPy for a specific variant."""

    variant = variant.upper()
    if variant not in VARIANT_FEATURES:
        raise ValueError(f"Unknown variant: {variant}")

    features = VARIANT_FEATURES[variant]

    if capacity is None:
        capacity = get_vehicle_capacity(num_loc)

    # Generate demands
    def generate_demand(size):
        return (
            np.random.randint(min_demand, max_demand + 1, size).astype(np.float32)
            / capacity
        )

    demand_linehaul = generate_demand((dataset_size, num_loc))
    demand_backhaul = None

    if features["B"]:
        demand_backhaul = np.zeros((dataset_size, num_loc))
        backhaul_mask = (
            np.random.rand(dataset_size, num_loc) < 0.2
        )  # 20% of nodes are backhaul
        demand_backhaul[backhaul_mask] = generate_demand(backhaul_mask.sum())
        demand_linehaul[backhaul_mask] = 0

    # Generate backhaul class
    backhaul_class = (
        np.full((dataset_size, 1), 2 if features["M"] else 1) if features["B"] else None
    )

    # Generate open route
    open_route = np.full((dataset_size, 1), features["O"]) if features["O"] else None

    # Generate time windows and service time
    time_windows = None
    service_time = None
    if features["TW"]:
        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * np.random.rand(dataset_size, num_loc)
        tw_length = b + (c - b) * np.random.rand(dataset_size, num_loc)
        if duration_matrix is not None:
            d_0i = duration_matrix[:, 0, 1:]
            d_i0 = duration_matrix[:, 1:, 0]
            d_max = np.maximum(d_0i, d_i0)
            h_max = (max_time - service_time - tw_length) / (d_max + 1e-6) - 1
            tw_start = d_0i + (h_max - 1) * d_max * np.random.rand(dataset_size, num_loc)
            tw_end = tw_start + tw_length
        else:
            # Generate locations
            locs = np.random.uniform(min_loc, max_loc, (dataset_size, num_loc + 1, 2))
            d_0i = np.linalg.norm(locs[:, 0:1] - locs[:, 1:], axis=2)
            h_max = (max_time - service_time - tw_length) / d_0i * speed - 1
            tw_start = (
                (1 + (h_max - 1) * np.random.rand(dataset_size, num_loc)) * d_0i / speed
            )
            tw_end = tw_start + tw_length

        time_windows = np.concatenate(
            [np.zeros((dataset_size, 1, 2)), np.stack([tw_start, tw_end], axis=-1)],
            axis=1,
        )
        time_windows[:, 0, 1] = max_time
        service_time = np.pad(service_time, ((0, 0), (1, 0)))

    # Generate distance limits: dist_lower_bound = 2 * max(depot_to_location_distance),
    # max = min(dist_lower_bound, max_distance_limit). Ensures feasible yet challenging
    # constraints, with each instance having a unique, meaningful limit.
    if features["L"]:
        # Calculate the maximum distance from depot to any location
        max_dist = np.max(np.linalg.norm(locs[:, 1:] - locs[:, 0:1], axis=2), axis=1)

        # Calculate the minimum distance limit (2 * max_distance)
        distance_lower_bound = 2 * max_dist + 1e-6  # Add epsilon to avoid zero distance

        # Ensure max_distance_limit is not exceeded
        max_distance_limit = np.maximum(max_distance_limit, distance_lower_bound + 1e-6)

        # Generate distance limits between min_distance_limits and max_distance_limit
        distance_limit = np.random.uniform(
            distance_lower_bound,
            np.full_like(distance_lower_bound, max_distance_limit),
            (dataset_size,),
        )[:, None]
    else:
        distance_limit = None

    # Generate speed
    speed = np.full((dataset_size, 1), speed)

    # Scale demand if needed
    if scale_demand:
        vehicle_capacity = np.full((dataset_size, 1), 1.0)
    else:
        vehicle_capacity = np.full((dataset_size, 1), capacity)
        if demand_backhaul is not None:
            demand_backhaul *= capacity
        demand_linehaul *= capacity

    data = {
        "demand_linehaul": demand_linehaul.astype(np.float32),
        "vehicle_capacity": vehicle_capacity.astype(np.float32),
        "speed": speed.astype(np.float32),
    }

    # Only include features that are used in the variant
    if features["B"]:
        data["demand_backhaul"] = demand_backhaul.astype(np.float32)
        data["backhaul_class"] = backhaul_class.astype(np.float32)
    if features["O"]:
        data["open_route"] = open_route
    if features["TW"]:
        data["time_windows"] = time_windows.astype(np.float32)
        data["service_time"] = service_time.astype(np.float32)
    if features["L"]:
        data["distance_limit"] = distance_limit.astype(np.float32)

    return data


# Main Dataset Generation Functions
def generate_env_data(
    env_type,
    data_dir,
    dataset_size,
    graph_size,
    in_distribution,
    dist_type="uniform",
    **kwargs,
):
    """Generate data for a given environment type."""
    cities_list = load_cities_list(data_dir, in_distribution)

    if env_type == "rcvrp":
        sampled_data = sample_data(
            data_dir, cities_list, dataset_size, graph_size + 1, dist_type
        )
        return prepare_rcvrp_data(sampled_data, dataset_size, graph_size)
    elif env_type == "atsp":
        sampled_data = sample_data(
            data_dir, cities_list, dataset_size, graph_size, dist_type
        )
        return prepare_atsp_data(sampled_data)
    elif env_type == "rcvrptw":
        sampled_data = sample_data(
            data_dir, cities_list, dataset_size, graph_size + 1, dist_type
        )
        return prepare_rcvrptw_data(sampled_data, dataset_size, graph_size)
    else:
        raise NotImplementedError(f"Environment type '{env_type}' not implemented.")


def generate_dataset(
    filename=None,
    data_dir="data/dataset",
    save_dir="data",
    problem="rcvrp",
    dataset_size=1280,
    graph_sizes=[100],
    overwrite=False,
    seed=3333,
    in_distribution=False,
    dist_type="uniform",
    disable_warning=True,
    **kwargs,
):
    """Generate and save datasets for routing problems."""
    if isinstance(graph_sizes, int):
        graph_sizes = [graph_sizes]
    data_type = "in_distribution" if in_distribution else "out_of_distribution"
    for graph_size in graph_sizes:
        save_dir = os.path.join(save_dir, problem)
        os.makedirs(save_dir, exist_ok=True)
        if dist_type == "uniform":
            fname = filename or os.path.join(
                save_dir, f"{problem}_n{graph_size}_seed{seed}_{data_type}.npz"
            )
        elif dist_type == "cluster":
            fname = filename or os.path.join(
                save_dir,
                f"{problem}_n{graph_size}_seed{seed}_{data_type}_{dist_type}.npz",
            )
        fname = check_extension(fname, ".npz")

        if not overwrite and os.path.exists(fname):
            if not disable_warning:
                log.info(f"File {fname} already exists. Skipping...")
            continue

        np.random.seed(seed)
        dataset = generate_env_data(
            problem,
            data_dir,
            dataset_size,
            graph_size,
            in_distribution,
            dist_type,
            **kwargs,
        )

        if dataset:
            log.info(f"Saving {problem} dataset to {fname}")
            np.savez(fname, **dataset)


# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create")
    parser.add_argument(
        "--data_dir", default="data/dataset", help="Base directory for dataset"
    )
    parser.add_argument("--save_dir", type=str, default="data", help="save directory")
    parser.add_argument(
        "--problems",
        type=str,
        nargs="+",
        default=["rcvrp", "atsp", "rcvrptw"],
        help="List of problem types to generate",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=1280, help="Size of the dataset"
    )
    parser.add_argument(
        "--graph_sizes", type=int, nargs="+", default=[100], help="Graph sizes"
    )
    parser.add_argument("-f", action="store_true", help="Overwrite existing datasets")
    parser.add_argument("--seed", type=int, default=3333, help="Random seed")
    parser.add_argument(
        "--disable_warning", action="store_true", help="Disable overwrite warnings"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    args.overwrite = args.f
    delattr(args, "f")

    for in_distribution in [True, False]:
        for dist_type in ["uniform", "cluster"]:
            if not in_distribution and dist_type == "cluster":
                continue
            for problem in args.problems:
                generate_dataset(
                    filename=args.filename,
                    data_dir=args.data_dir,
                    save_dir=args.save_dir,
                    problem=problem,
                    dataset_size=args.dataset_size,
                    graph_sizes=args.graph_sizes,
                    overwrite=args.overwrite,
                    seed=args.seed,
                    in_distribution=in_distribution,
                    dist_type=dist_type,
                    disable_warning=args.disable_warning,
                )
