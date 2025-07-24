import os
import random
from typing import Callable, Union, Optional, Tuple

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


class LazyRCVRPGenerator(Generator):
    """Memory-efficient Lazy Loading RCVRP Generator.

    This generator combines the memory efficiency of lazy loading with
    RCVRP features.
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
        chunk_size: int = 100,  # Chunk size for memory efficiency
        **kwargs,
    ):
        super().__init__()

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
        self.file_name = file_name

        # Lazy loading specific
        self.chunk_size = chunk_size

        # Lazy loading attributes
        self._cities_list = None
        self._city_data_cache = {}
        self._cache_size_limit = 5  # Limit cache size to prevent memory issues

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
        self.demand_sampler = get_sampler(
            "demand", demand_distribution, min_demand, max_demand, **kwargs
        )

        # Capacity
        if capacity is None:
            capacity = CAPACITIES.get(num_loc, 50.0)
        self.capacity = capacity

    @property
    def cities_list(self):
        """Lazy load cities list"""
        if self._cities_list is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cities_path = os.path.join(base_dir, self.data_path, f"{self.file_name}.json")

            if os.path.exists(cities_path):
                with open(cities_path, "r") as f:
                    cities_data = orjson.loads(f.read())
                    self._cities_list = cities_data.get("train", [])
            else:
                self._cities_list = []
                log.warning(f"Cities list not found at {cities_path}")

        return self._cities_list

    def _load_city_data(self, city: str) -> Optional[dict]:
        """Lazy load city data with cache management"""
        # Check cache size and clean if necessary
        if len(self._city_data_cache) >= self._cache_size_limit:
            # Remove oldest entry (simple FIFO)
            oldest_city = next(iter(self._city_data_cache))
            del self._city_data_cache[oldest_city]

        if city not in self._city_data_cache:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, self.data_path, city, f"{city}_data.npz")

            if os.path.exists(data_path):
                try:
                    # Load data fully into memory for pickle compatibility
                    data = np.load(data_path, allow_pickle=True)
                    self._city_data_cache[city] = {"path": data_path, "data": data}
                except Exception as e:
                    log.error(f"Failed to load data for city {city}: {e}")
                    return None
            else:
                log.warning(f"Data for city {city} not found at {data_path}")
                return None

        return self._city_data_cache[city]

    def __getstate__(self):
        """Pickle 시 캐시와 파일 관련 데이터를 제외하여 BufferedReader 문제 방지"""
        state = self.__dict__.copy()
        # 캐시와 파일 관련 데이터 제거 (파일 핸들러 포함 가능)
        state["_city_data_cache"] = {}
        state["_cities_list"] = None
        return state

    def __setstate__(self, state):
        """Unpickle 시 캐시와 파일 관련 데이터 초기화"""
        self.__dict__.update(state)
        self._city_data_cache = {}
        self._cities_list = None

    def _generate(self, batch_size) -> TensorDict:
        """Main generation function with chunk processing"""
        total_size = batch_size[0]
        num_chunks = (total_size + self.chunk_size - 1) // self.chunk_size

        all_tds = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min((chunk_idx + 1) * self.chunk_size, total_size)
            chunk_batch_size = [chunk_end - chunk_start]

            # Generate chunk
            chunk_td = self._generate_chunk(chunk_batch_size, chunk_idx)
            all_tds.append(chunk_td)

        # Merge chunks efficiently
        if len(all_tds) == 1:
            final_td = all_tds[0]
        else:
            # Concatenate all tensordicts
            final_td = self._merge_tensordicts(all_tds)

        return final_td

    def _generate_chunk(self, batch_size: list, chunk_idx: int) -> TensorDict:
        """Generate data for a single chunk"""
        if isinstance(self.loc_sampler, Real_World_Sampler) and self.cities_list:
            return self._generate_real_world_chunk(batch_size)
        else:
            return self._generate_synthetic_chunk(batch_size)

    def _generate_real_world_chunk(self, batch_size: list) -> TensorDict:
        """Generate chunk using real world data"""
        target_batch_size = batch_size[0]

        # Sample cities
        num_cities_per_epoch = min(10, len(self.cities_list))
        cities = random.sample(self.cities_list, num_cities_per_epoch)
        sub_batch_size = max(1, target_batch_size // num_cities_per_epoch)

        chunk_data = None
        total_generated = 0

        for i, city in enumerate(cities):
            if total_generated >= target_batch_size:
                break

            city_data = self._load_city_data(city)
            if city_data is None:
                continue

            # Calculate remaining batch size
            remaining_batch = target_batch_size - total_generated
            current_batch_size = min(sub_batch_size, remaining_batch)

            # Sample from city data
            sampled_data = self.loc_sampler.sample(
                data=city_data["data"],
                batch=current_batch_size,
                num_sample=self.num_loc + 1,
                loc_dist=self.loc_distribution,
                num_cluster=self.num_cluster,
            )

            if chunk_data is None:
                chunk_data = sampled_data
            else:
                # Efficient concatenation
                for key in ["points", "distance_matrix", "duration_matrix"]:
                    if key in sampled_data:
                        chunk_data[key] = np.concatenate(
                            (chunk_data[key], sampled_data[key]), axis=0
                        )

            total_generated += current_batch_size

        # Process chunk data
        return self._process_real_world_data(chunk_data, batch_size)

    def _generate_synthetic_chunk(self, batch_size: list) -> TensorDict:
        """Generate synthetic data chunk"""
        # Sample locations: depot and customers
        locs = torch.FloatTensor(*batch_size, self.num_loc, 2).uniform_(
            self.min_loc, self.max_loc
        )

        # Sample depot
        if self.depot_sampler is not None:
            depot = self.depot_sampler(batch_size)
        else:
            depot = torch.FloatTensor(*batch_size, 1, 2).uniform_(
                self.min_loc, self.max_loc
            )

        # Sample demand
        demand = self.demand_sampler((*batch_size, self.num_loc))

        # Sample capacity
        capacity = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "demand": demand / self.capacity,
                "capacity": capacity,
            },
            batch_size=batch_size,
        )

    def _process_real_world_data(self, chunk_data: dict, batch_size: list) -> TensorDict:
        """Process real world data into TensorDict"""
        points = chunk_data["points"].astype(np.float32)

        # Split depot and locations
        depot = torch.from_numpy(points[:, :1, :]).float()  # First point is depot
        locs = torch.from_numpy(points[:, 1:, :]).float()  # Rest are customer locations

        # Generate demand
        demand = self.demand_sampler((*batch_size, self.num_loc))

        # Generate capacity
        capacity = torch.full((*batch_size, 1), self.capacity, dtype=torch.float32)

        # Add distance matrix if available
        td_data = {
            "locs": locs,
            "depot": depot,
            "demand": demand / self.capacity,
            "capacity": capacity,
        }

        if "distance_matrix" in chunk_data:
            distance = torch.from_numpy(chunk_data["distance_matrix"].astype(np.float32))
            td_data["distance_matrix"] = distance

        return TensorDict(td_data, batch_size=batch_size)

    def _merge_tensordicts(self, tds: list) -> TensorDict:
        """Efficiently merge multiple TensorDicts"""
        # Get all keys from first tensordict
        keys = list(tds[0].keys())

        # Merge data
        merged_data = {}
        for key in keys:
            # Collect all tensors for this key
            tensors = [td[key] for td in tds if key in td.keys()]
            # Concatenate along batch dimension
            merged_data[key] = torch.cat(tensors, dim=0)

        # Calculate total batch size
        total_batch_size = [sum(td.batch_size[0] for td in tds)]

        return TensorDict(merged_data, batch_size=total_batch_size)

    def clear_cache(self):
        """Clear city data cache."""
        self._city_data_cache.clear()
        self._cities_list = None
