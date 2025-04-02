import os
import random

from typing import Callable, Union

import numpy as np
import orjson
import torch

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rrnco.envs.rcvrp.sampler import Real_World_Sampler
from rrnco.envs.rcvrp.utils import get_real_world_sampler

log = get_pylogger(__name__)


# From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023
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


class RCVRPGenerator(Generator):
    """Data generator for the Capacitated Vehicle Routing Problem (CVRP).

    Args:
        num_loc: number of locations (cities) in the VRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations
        min_demand: minimum value for the demand of each customer
        max_demand: maximum value for the demand of each customer
        demand_distribution: distribution for the demand of each customer
        capacity: capacity of the vehicle

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            depot [batch_size, 2]: location of the depot
            demand [batch_size, num_loc]: demand of each customer
            capacity [batch_size]: capacity of the vehicle
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = "uniform",
        depot_distribution: Union[int, float, str, type, Callable] = None,
        min_demand: int = 1,
        max_demand: int = 10,
        demand_distribution: Union[int, float, type, Callable] = Uniform,
        vehicle_capacity: float = 1.0,
        capacity: float = None,
        num_cluster: int = 5,
        data_path: str = "../../../data/dataset",
        file_name: str = "splited_cities_list",
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = vehicle_capacity

        self.loc_sampler = get_real_world_sampler()
        self.loc_distribution = loc_distribution
        self.num_cluster = num_cluster
        self.data_path = data_path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(f"{base_dir}/{data_path}/{file_name}.json"):
            with open(f"{base_dir}/{data_path}/{file_name}.json", "r") as f:
                cities_list = orjson.loads(f.read())
                train_cities_list = cities_list["train"]
            self.train_cities_list = train_cities_list
        else:
            self.train_cities_list = None
        # # Location distribution
        # if kwargs.get("loc_sampler", None) is not None:
        #     self.loc_sampler = kwargs["loc_sampler"]
        # else:
        #     self.loc_sampler = get_sampler(
        #         "loc", loc_distribution, min_loc, max_loc, **kwargs
        #     )

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = (
                get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)
                if depot_distribution is not None
                else None
            )

        # Demand distribution
        if kwargs.get("demand_sampler", None) is not None:
            self.demand_sampler = kwargs["demand_sampler"]
        else:
            self.demand_sampler = get_sampler(
                "demand", demand_distribution, min_demand - 1, max_demand - 1, **kwargs
            )

        # Capacity
        if (
            capacity is None
        ):  # If not provided, use the default capacity from Kool et al. 2019
            capacity = CAPACITIES.get(num_loc, None)
        if (
            capacity is None
        ):  # If not in the table keys, find the closest number of nodes as the key
            closest_num_loc = min(CAPACITIES.keys(), key=lambda x: abs(x - num_loc))
            capacity = CAPACITIES[closest_num_loc]
            log.warning(
                f"The capacity capacity for {num_loc} locations is not defined. Using the closest capacity: {capacity}\
                    with {closest_num_loc} locations."
            )
        self.capacity = capacity

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations: depot and customers
        if (
            isinstance(self.loc_sampler, Real_World_Sampler)
            and self.train_cities_list is not None
        ):
            num_cities_per_epoch = 10
            cities = random.sample(self.train_cities_list, num_cities_per_epoch)
            sub_batch_size = batch_size[0] // num_cities_per_epoch
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for i, city in enumerate(cities):
                full_data_path = os.path.join(
                    base_dir, "../../../data/dataset", city, f"{city}_data.npz"
                )
                if not os.path.exists(full_data_path):
                    raise ValueError(
                        f"Data for city {city} not found in {self.data_path}"
                    )
                data = np.load(full_data_path, allow_pickle=True, mmap_mode="r")
                if i == 0:
                    sampled_data = self.loc_sampler.sample(
                        data=data,
                        batch=sub_batch_size,
                        num_sample=self.num_loc + 1,
                        loc_dist=self.loc_distribution,
                        num_cluster=self.num_cluster,
                    )
                else:
                    new_data = self.loc_sampler.sample(
                        data=data,
                        batch=sub_batch_size,
                        num_sample=self.num_loc + 1,
                        loc_dist=self.loc_distribution,
                        num_cluster=self.num_cluster,
                    )
                    sampled_data["points"] = np.concatenate(
                        (sampled_data["points"], new_data["points"]), axis=0
                    )
                    sampled_data["distance_matrix"] = np.concatenate(
                        (sampled_data["distance_matrix"], new_data["distance_matrix"]),
                        axis=0,
                    )

            points = sampled_data["points"].astype(np.float32)
            distance = torch.from_numpy(
                sampled_data["distance_matrix"].astype(np.float32)
            )

            points_min = np.min(points, axis=1, keepdims=True)
            points_max = np.max(points, axis=1, keepdims=True)
            locs = (points - points_min) / (points_max - points_min)
            points = sampled_data["points"].astype(np.float32)
            distance = torch.from_numpy(
                sampled_data["distance_matrix"].astype(np.float32)
            )

            points_min = np.min(points, axis=1, keepdims=True)
            points_max = np.max(points, axis=1, keepdims=True)
            locs = (points - points_min) / (points_max - points_min)
            depot = torch.from_numpy(locs[:, 0, :])
            locs = torch.from_numpy(locs[:, 1:, :])

        elif self.depot_sampler is not None:
            depot = self.depot_sampler.sample((*batch_size, 2))
            locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        else:
            # if depot_sampler is None, sample the depot from the locations
            locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2))
            depot = locs[..., 0, :]
            locs = locs[..., 1:, :]

        # Sample demands
        demand = self.demand_sampler.sample((*batch_size, self.num_loc))
        demand = (demand.int() + 1).float()

        # Sample capacities
        capacity = torch.full((*batch_size, 1), self.capacity)

        if isinstance(self.loc_sampler, Real_World_Sampler):
            return TensorDict(
                {
                    "locs": locs,
                    "depot": depot,
                    "demand": demand / self.capacity,
                    "capacity": capacity,
                    "distance_matrix": distance,
                },
                batch_size=batch_size,
            )

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "demand": demand / self.capacity,
                "capacity": capacity,
            },
            batch_size=batch_size,
        )
