import os
import random
from typing import Callable, Union, Optional, Tuple

import numpy as np
import orjson
import torch

from rl4co.data.utils import save_tensordict_to_npz
from rl4co.envs.common.utils import Generator
from rl4co.utils.ops import get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict

from rrnco.envs.rmtvrp.sampler import Real_World_Sampler
from rrnco.envs.rmtvrp.utils import get_real_world_sampler

log = get_pylogger(__name__)


def get_vehicle_capacity(num_loc: int) -> int:
    """Capacity should be 30 + num_loc/5 if num_loc > 20 as described in Liu et al. 2024 (POMO-MTL).
    For every N over 1000, we add 1 of capacity every 33.3 nodes to align with Ye et al. 2024 (GLOP),
    i.e. 260 at 2K nodes, 350 at 5K nodes and 500 at 10K nodes.
    """
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap


VARIANT_GENERATION_PRESETS = {
    "all": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
    "single_feat": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5},
    "single_feat_otw": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5, "OTW": 0.5},
    "cvrp": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0},
    "ovrp": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0},
    "vrpb": {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "vrpl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0},
    "vrptw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0},
    "ovrptw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0},
    "ovrpb": {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0},
    "ovrpl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0},
    "vrpbl": {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "vrpbtw": {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "vrpltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0},
    "ovrpbl": {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0},
    "ovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0},
    "ovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0},
    "vrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0},
    "ovrpbltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0},
}


class LazyRMTVRPGenerator(Generator):
    """Memory-efficient Lazy Loading MTVRP Generator with all features.

    This generator combines the memory efficiency of lazy loading with
    all the MTVRP features from the naive generator.
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = "uniform",
        capacity: float = None,
        min_demand: int = 1,
        max_demand: int = 10,
        min_backhaul: int = 1,
        max_backhaul: int = 10,
        scale_demand: bool = True,
        max_time: float = 4.6,
        backhaul_ratio: float = 0.2,
        backhaul_class: int = 1,
        sample_backhaul_class: bool = False,
        max_distance_limit: float = 2.8,
        speed: float = 1.0,
        prob_open: float = 0.5,
        prob_time_window: float = 0.5,
        prob_limit: float = 0.5,
        prob_backhaul: float = 0.5,
        variant_preset="vrptw",
        use_combinations=False,
        subsample=True,
        num_cluster: int = 5,
        data_path: str = "../../../data/dataset",
        file_name: str = "splited_cities_list",
        chunk_size: int = 100,  # Chunk size for memory efficiency
        **kwargs,
    ) -> None:
        super().__init__()

        # Basic parameters
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.loc_distribution = loc_distribution

        # Capacity
        if capacity is None:
            capacity = get_vehicle_capacity(num_loc)
        self.capacity = capacity

        # Demand parameters
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_backhaul = min_backhaul
        self.max_backhaul = max_backhaul
        self.scale_demand = scale_demand

        # MTVRP specific parameters
        self.backhaul_ratio = backhaul_ratio
        assert backhaul_class in (1, 2), "Backhaul class must be in [1, 2]"
        self.backhaul_class = backhaul_class
        self.sample_backhaul_class = sample_backhaul_class
        self.max_time = max_time
        self.max_distance_limit = max_distance_limit
        self.speed = speed

        # Variant generation
        if variant_preset is not None:
            log.info(f"Using variant generation preset {variant_preset}")
            variant_probs = VARIANT_GENERATION_PRESETS.get(variant_preset)
            assert variant_probs is not None, f"Variant preset {variant_preset} not found"
        else:
            variant_probs = {
                "O": prob_open,
                "TW": prob_time_window,
                "L": prob_limit,
                "B": prob_backhaul,
            }

        # Validate probabilities
        for key, prob in variant_probs.items():
            assert 0 <= prob <= 1, f"Probability {key} must be between 0 and 1"

        self.variant_probs = variant_probs
        self.variant_preset = variant_preset
        if isinstance(variant_preset, str) and variant_preset != "all":
            log.info(f"{variant_preset} selected. Will not use feature combination!")
            use_combinations = False
        self.use_combinations = use_combinations
        self.subsample = subsample

        # Lazy loading specific
        self.chunk_size = chunk_size
        self.loc_sampler = get_real_world_sampler()
        self.num_cluster = num_cluster
        self.data_path = data_path
        self.file_name = file_name

        # Lazy loading attributes
        self._cities_list = None
        self._city_data_cache = {}
        self._cache_size_limit = 5  # Limit cache size to prevent memory issues

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
                    # Memory mapping for efficiency
                    data = np.load(data_path, allow_pickle=True, mmap_mode="r")
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

        # Apply subsampling if needed
        if self.subsample:
            final_td = self.subsample_problems(final_td)

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
                    chunk_data[key] = np.concatenate(
                        (chunk_data[key], sampled_data[key]), axis=0
                    )

            total_generated += current_batch_size

        # Process chunk data
        return self._process_real_world_data(chunk_data, batch_size)

    def _generate_synthetic_chunk(self, batch_size: list) -> TensorDict:
        """Generate synthetic data chunk"""
        # Generate locations
        locs = self.generate_locations(batch_size=batch_size, num_loc=self.num_loc)

        # Generate all MTVRP components
        vehicle_capacity = torch.full(
            (*batch_size, 1), self.capacity, dtype=torch.float32
        )
        capacity_original = vehicle_capacity.clone()

        # Demands
        demand_linehaul, demand_backhaul = self.generate_demands(
            batch_size=batch_size, num_loc=self.num_loc
        )

        # Other components
        backhaul_class = self.generate_backhaul_class(
            shape=(*batch_size, 1), sample=self.sample_backhaul_class
        )
        speed = self.generate_speed(shape=(*batch_size, 1))
        time_windows, service_time = self.generate_time_windows(locs=locs, speed=speed)
        open_route = self.generate_open_route(shape=(*batch_size, 1))
        distance_limit = self.generate_distance_limit(shape=(*batch_size, 1), locs=locs)

        # Scale demands if needed
        if self.scale_demand:
            demand_backhaul /= vehicle_capacity
            demand_linehaul /= vehicle_capacity
            vehicle_capacity /= vehicle_capacity

        return TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "time_windows": time_windows,
                "service_time": service_time,
                "vehicle_capacity": vehicle_capacity,
                "capacity_original": capacity_original,
                "open_route": open_route,
                "speed": speed,
            },
            batch_size=batch_size,
        )

    def _process_real_world_data(self, chunk_data: dict, batch_size: list) -> TensorDict:
        """Process real world data into TensorDict"""
        points = chunk_data["points"].astype(np.float32)
        distance = torch.from_numpy(chunk_data["distance_matrix"].astype(np.float32))
        duration = chunk_data["duration_matrix"].astype(np.float32)

        # Normalize locations
        points_min = np.min(points, axis=1, keepdims=True)
        points_max = np.max(points, axis=1, keepdims=True)
        locs = (points - points_min) / (points_max - points_min + 1e-8)
        locs = torch.from_numpy(locs).float()

        # Normalize duration
        duration_min = np.min(duration, axis=(1, 2), keepdims=True)
        duration_max = np.max(duration, axis=(1, 2), keepdims=True)
        denom = np.where(duration_max - duration_min == 0, 1, duration_max - duration_min)
        normalized_duration = (duration - duration_min) / denom
        normalized_duration = torch.from_numpy(normalized_duration).float()

        # Generate all MTVRP components
        vehicle_capacity = torch.full(
            (*batch_size, 1), self.capacity, dtype=torch.float32
        )
        capacity_original = vehicle_capacity.clone()

        # Generate demands
        demand_linehaul, demand_backhaul = self.generate_demands(
            batch_size=batch_size, num_loc=self.num_loc
        )

        # Generate other components
        backhaul_class = self.generate_backhaul_class(
            shape=(*batch_size, 1), sample=self.sample_backhaul_class
        )
        speed = self.generate_speed(shape=(*batch_size, 1))
        time_windows, service_time = self.generate_time_windows_with_duration_matrix(
            duration=normalized_duration
        )
        open_route = self.generate_open_route(shape=(*batch_size, 1))
        distance_limit = self.generate_distance_limit(shape=(*batch_size, 1), locs=locs)

        # Scale demands if needed
        if self.scale_demand:
            demand_backhaul /= vehicle_capacity
            demand_linehaul /= vehicle_capacity
            vehicle_capacity /= vehicle_capacity

        td = TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "time_windows": time_windows,
                "service_time": service_time,
                "vehicle_capacity": vehicle_capacity,
                "capacity_original": capacity_original,
                "open_route": open_route,
                "speed": speed,
                "distance_matrix": distance,
                "duration_matrix": normalized_duration,
            },
            batch_size=batch_size,
        )

        return td

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
        total_batch_size = sum(td.batch_size[0] for td in tds)

        return TensorDict(merged_data, batch_size=[total_batch_size])

    # ===== All methods from Naive Generator =====

    def generate_locations(self, batch_size, num_loc) -> torch.Tensor:
        """Generate seed locations."""
        locs = torch.FloatTensor(*batch_size, num_loc + 1, 2).uniform_(
            self.min_loc, self.max_loc
        )
        return locs

    def generate_demands(
        self, batch_size: int, num_loc: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate linehaul and backhaul demands."""
        linehaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_demand - 1, self.max_demand - 1
        )
        linehaul_demand = (linehaul_demand.int() + 1).float()

        backhaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_backhaul - 1, self.max_backhaul - 1
        )
        backhaul_demand = (backhaul_demand.int() + 1).float()

        is_linehaul = torch.rand(*batch_size, num_loc) > self.backhaul_ratio
        backhaul_demand = backhaul_demand * ~is_linehaul
        linehaul_demand = linehaul_demand * is_linehaul

        return linehaul_demand, backhaul_demand

    def generate_time_windows(
        self, locs: torch.Tensor, speed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate time windows and service times."""
        batch_size, n_loc = locs.shape[0], locs.shape[1] - 1

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)
        d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i / speed
        tw_end = tw_start + tw_length

        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),
            dim=-1,
        )
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)

        return time_windows, service_time

    def generate_time_windows_with_duration_matrix(
        self, duration: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate time windows with duration matrix."""
        batch_size, n_loc = duration.shape[0], duration.shape[1] - 1

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)

        d_0i = duration[:, 0, 1:]
        d_i0 = duration[:, 1:, 0]
        d_max = torch.max(d_0i, d_i0)
        h_max = (self.max_time - service_time - tw_length) / (d_max + 1e-6) - 1
        tw_start = d_0i + (h_max - 1) * d_max * torch.rand(batch_size, n_loc)
        tw_end = tw_start + tw_length

        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),
            dim=-1,
        )
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)

        return time_windows, service_time

    def generate_distance_limit(
        self, shape: Tuple[int, int], locs: torch.Tensor
    ) -> torch.Tensor:
        """Generate distance limits."""
        max_dist = torch.max(torch.cdist(locs[:, 0:1], locs[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist + 1e-6
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.max_distance_limit),
            dist_lower_bound + 1e-6,
        )

        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ]

    def generate_open_route(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Generate open route flags."""
        return torch.ones(shape, dtype=torch.bool)

    def generate_speed(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Generate speed."""
        return torch.full(shape, self.speed, dtype=torch.float32)

    def generate_backhaul_class(
        self, shape: Tuple[int, int], sample: bool = False
    ) -> torch.Tensor:
        """Generate backhaul class."""
        if sample:
            return torch.randint(1, 3, shape, dtype=torch.float32)
        else:
            return torch.full(shape, self.backhaul_class, dtype=torch.float32)

    def subsample_problems(self, td: TensorDict) -> TensorDict:
        """Subsample problems based on variant probabilities."""
        batch_size = td.batch_size[0]
        variant_probs = torch.tensor(list(self.variant_probs.values()))

        if self.use_combinations:
            keep_mask = torch.rand(batch_size, 4) >= variant_probs
        else:
            if self.variant_preset in list(
                VARIANT_GENERATION_PRESETS.keys()
            ) and self.variant_preset not in (
                "all",
                "cvrp",
                "single_feat",
                "single_feat_otw",
            ):
                cvrp_prob = 0
            else:
                cvrp_prob = 0.5

            if self.variant_preset in ("all", "cvrp", "single_feat", "single_feat_otw"):
                indices = torch.distributions.Categorical(
                    torch.Tensor(list(self.variant_probs.values()) + [cvrp_prob])[
                        None
                    ].repeat(batch_size, 1)
                ).sample()

                if self.variant_preset == "single_feat_otw":
                    keep_mask = torch.zeros((batch_size, 6), dtype=torch.bool)
                    keep_mask[torch.arange(batch_size), indices] = True
                    keep_mask[:, :2] |= keep_mask[:, 4:5]
                else:
                    keep_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
                    keep_mask[torch.arange(batch_size), indices] = True
            else:
                keep_mask = torch.zeros((batch_size, 4), dtype=torch.bool)
                indices = torch.nonzero(variant_probs).squeeze()
                keep_mask[:, indices] = True

        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_window(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])

        return td

    @staticmethod
    def _default_open(td: TensorDict, remove: torch.Tensor) -> TensorDict:
        """Set default for open routes."""
        td["open_route"][remove] = False
        return td

    @staticmethod
    def _default_time_window(td: TensorDict, remove: torch.Tensor) -> TensorDict:
        """Set default for time windows."""
        default_tw = torch.zeros_like(td["time_windows"])
        default_tw[..., 1] = float("inf")
        td["time_windows"][remove] = default_tw[remove]
        td["service_time"][remove] = torch.zeros_like(td["service_time"][remove])
        return td

    @staticmethod
    def _default_distance_limit(td: TensorDict, remove: torch.Tensor) -> TensorDict:
        """Set default for distance limits."""
        td["distance_limit"][remove] = float("inf")
        return td

    @staticmethod
    def _default_backhaul(td: TensorDict, remove: torch.Tensor) -> TensorDict:
        """Set default for backhauls."""
        td["demand_linehaul"][remove] = (
            td["demand_linehaul"][remove] + td["demand_backhaul"][remove]
        )
        td["demand_backhaul"][remove] = 0
        return td

    def clear_cache(self):
        """Clear city data cache."""
        self._city_data_cache.clear()
        self._cities_list = None

    @staticmethod
    def save_data(td: TensorDict, path: str, compress: bool = False):
        """Save TensorDict to file."""
        save_tensordict_to_npz(td, path)

    @staticmethod
    def print_presets():
        """Print available variant presets."""
        for key, value in VARIANT_GENERATION_PRESETS.items():
            print(f"{key}: {value}")

    @staticmethod
    def available_variants(*args, **kwargs):
        """Get available variants."""
        return list(VARIANT_GENERATION_PRESETS.keys())[3:]
