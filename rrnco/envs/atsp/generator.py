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

from rrnco.envs.atsp.sampler import Real_World_Sampler
from rrnco.envs.atsp.utils import get_real_world_sampler

log = get_pylogger(__name__)


class ATSPGenerator(Generator):
    """Data generator for the Asymmetric Travelling Salesman Problem (ATSP)
    Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
    We satifsy the triangle inequality (TMAT class) in a batch

    Args:
        num_loc: number of locations (customers) in the TSP
        min_dist: minimum value for the distance between nodes
        max_dist: maximum value for the distance between nodes
        dist_distribution: distribution for the distance between nodes
        tmat_class: whether to generate a class of distance matrix

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
    """

    def __init__(
        self,
        num_loc: int = 10,
        min_dist: float = 0.0,
        max_dist: float = 1.0,
        dist_distribution: Union[int, float, str, type, Callable] = Uniform,
        loc_distribution: Union[int, float, str, type, Callable] = "uniform",
        tmat_class: bool = True,
        num_cluster: int = 5,
        data_path: str = "../../../data/dataset",
        file_name: str = "splited_cities_list",
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.tmat_class = tmat_class
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
        # Distance distribution
        if not tmat_class:
            self.dist_sampler = get_real_world_sampler()
        else:
            self.dist_sampler = get_sampler("dist", dist_distribution, 0.0, 1.0, **kwargs)

    def _generate(self, batch_size) -> TensorDict:
        # Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
        # We satifsy the triangle inequality (TMAT class) in a batch
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        if (
            isinstance(self.dist_sampler, Real_World_Sampler)
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
                    sampled_data = self.dist_sampler.sample(
                        data=data,
                        batch=sub_batch_size,
                        num_sample=self.num_loc,
                        loc_dist=self.loc_distribution,
                        num_cluster=self.num_cluster,
                    )
                else:
                    new_data = self.dist_sampler.sample(
                        data=data,
                        batch=sub_batch_size,
                        num_sample=self.num_loc,
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
            locs = torch.from_numpy(locs)
            return TensorDict(
                {
                    "locs": locs,
                    "distance_matrix": distance,
                },
                batch_size=batch_size,
            )
        else:
            dms = (
                self.dist_sampler.sample((batch_size + [self.num_loc, self.num_loc]))
                * (self.max_dist - self.min_dist)
                + self.min_dist
            )
            dms[..., torch.arange(self.num_loc), torch.arange(self.num_loc)] = 0
            log.info("Using TMAT class (triangle inequality): {}".format(self.tmat_class))
            if self.tmat_class:
                for i in range(self.num_loc):
                    dms = torch.minimum(dms, dms[..., :, [i]] + dms[..., [i], :])

            return TensorDict(
                {
                    "distance_matrix": dms,
                },
                batch_size=batch_size,
            )

    def __getstate__(self):
        """Pickle 시 파일 관련 데이터를 제외하여 BufferedReader 문제 방지"""
        state = self.__dict__.copy()
        # 파일 관련 데이터 제거 (pickle 시 문제 방지)
        state["train_cities_list"] = None
        return state

    def __setstate__(self, state):
        """Unpickle 시 파일 관련 데이터 초기화"""
        self.__dict__.update(state)
        self.train_cities_list = None
