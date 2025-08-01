import os
import random

from typing import Callable, Tuple, Union

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
    Note that this serves as a demand scaler.
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


class RMTVRPGenerator(Generator):
    """MTVRP Generator.
    Class to generate instances of the MTVRP problem.

    Args:
        num_loc: Number of locations to generate
        min_loc: Minimum location value
        max_loc: Maximum location value
        loc_distribution: Distribution to sample locations from
        capacity: Vehicle capacity. If None, get value based on `get_vehicle_capacity`
        min_demand: Minimum demand value
        max_demand: Maximum demand value
        min_backhaul: Minimum backhaul value
        max_backhaul: Maximum backhaul value
        scale_demand: Scale demand values (by default, generate between 1 and 10)
        max_time: Maximum time window value (at depot)
        backhaul_ratio: Fraction of backhauls (e.g. 0.2 means 20% of nodes are backhaul)
        backhaul_class: which type of backhaul to use:
                0: no backhaul (note: we don't use 0 since we can efficiently generate CVRP by just transforming backhauls to linehauls)
                1: classic backhaul (VRPB), linehauls must be served before backhauls in a route (every customer is either, not both)
                2: mixed backhaul (VRPMPD or VRPMB), linehauls and backhauls can be served in any order (every customer is either, not both)
        distance_limit: Distance limit
        speed: Speed of vehicle. Defaults to 1
        **kwargs: Additional keyword arguments
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
        max_distance_limit: float = 2.8,  # 2sqrt(2) ~= 2.8
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
        **kwargs,
    ) -> None:
        # Location distribution
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
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
        # if kwargs.get("loc_sampler", None) is not None:
        #     self.loc_sampler = kwargs["loc_sampler"]
        # else:
        #     self.loc_sampler = get_sampler(
        #         "loc", loc_distribution, min_loc, max_loc, **kwargs
        #     )

        if capacity is None:
            capacity = get_vehicle_capacity(num_loc)
        self.capacity = capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_backhaul = min_backhaul
        self.max_backhaul = max_backhaul
        self.scale_demand = scale_demand
        self.backhaul_ratio = backhaul_ratio
        assert backhaul_class in (
            1,
            2,
        ), "Backhaul class must be in [1, 2]. We don't use class 0 for efficiency since it is a subset"
        self.backhaul_class = backhaul_class
        self.sample_backhaul_class = sample_backhaul_class
        self.max_time = max_time
        self.max_distance_limit = max_distance_limit
        self.speed = speed

        if variant_preset is not None:
            log.info(f"Using variant generation preset {variant_preset}")
            variant_probs = VARIANT_GENERATION_PRESETS.get(variant_preset)
            assert (
                variant_probs is not None
            ), f"Variant generation preset {variant_preset} not found. \
                Available presets are {VARIANT_GENERATION_PRESETS.keys()} with probabilities {VARIANT_GENERATION_PRESETS.values()}"
        else:
            variant_probs = {
                "O": prob_open,
                "TW": prob_time_window,
                "L": prob_limit,
                "B": prob_backhaul,
            }
        # check probabilities
        for key, prob in variant_probs.items():
            assert 0 <= prob <= 1, f"Probability {key} must be between 0 and 1"
        self.variant_probs = variant_probs
        self.variant_preset = variant_preset
        if isinstance(variant_preset, str) and variant_preset != "all":
            log.info(f"{variant_preset} selected. Will not use feature combination!")
            use_combinations = False
        self.use_combinations = use_combinations
        self.subsample = subsample

    def _generate(self, batch_size) -> TensorDict:
        # Locations
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
                    sampled_data["duration_matrix"] = np.concatenate(
                        (sampled_data["duration_matrix"], new_data["duration_matrix"]),
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

            locs = torch.from_numpy(locs).float()
            duration = sampled_data["duration_matrix"].astype(np.float32)
            # Compute batch-wise min and max
            duration_min = np.min(
                duration, axis=(1, 2), keepdims=True
            )  # Shape: [B, 1, 1]
            duration_max = np.max(
                duration, axis=(1, 2), keepdims=True
            )  # Shape: [B, 1, 1]

            # Avoid division by zero in case max == min
            denom = np.where(
                duration_max - duration_min == 0, 1, duration_max - duration_min
            )

            # Normalize
            normalized_duration = (duration - duration_min) / denom
            normalized_duration = torch.from_numpy(normalized_duration).float()

            speed = self.generate_speed(shape=(*batch_size, 1))
            # Sample demands
            time_windows, service_time = self.generate_time_windows_with_duration_matrix(
                duration=normalized_duration,
            )

        else:
            locs = self.generate_locations(batch_size=batch_size, num_loc=self.num_loc)
            # Vehicle capacity (C, B) - applies to both linehaul and backhaul

            # Time windows (TW)
            speed = self.generate_speed(shape=(*batch_size, 1))
            time_windows, service_time = self.generate_time_windows(
                locs=locs,
                speed=speed,
            )

        vehicle_capacity = torch.full(
            (*batch_size, 1), self.capacity, dtype=torch.float32
        )
        capacity_original = vehicle_capacity.clone()

        # linehaul demand / delivery (C) and backhaul / pickup demand (B)
        demand_linehaul, demand_backhaul = self.generate_demands(
            batch_size=batch_size, num_loc=self.num_loc
        )

        backhaul_class = self.generate_backhaul_class(
            shape=(*batch_size, 1), sample=self.sample_backhaul_class
        )

        # Open (O)
        open_route = self.generate_open_route(shape=(*batch_size, 1))

        # Distance limit (L)
        distance_limit = self.generate_distance_limit(shape=(*batch_size, 1), locs=locs)

        # scaling
        if self.scale_demand:
            demand_backhaul /= vehicle_capacity
            demand_linehaul /= vehicle_capacity
            vehicle_capacity /= vehicle_capacity

            # Put all variables together
        td = TensorDict(
            {
                "locs": locs,
                "demand_backhaul": demand_backhaul,  # (C)
                "demand_linehaul": demand_linehaul,  # (B)
                "backhaul_class": backhaul_class,  # (B)
                "distance_limit": distance_limit,  # (L)
                "time_windows": time_windows,  # (TW)
                "service_time": service_time,  # (TW)
                "vehicle_capacity": vehicle_capacity,  # (C)
                "capacity_original": capacity_original,  # unscaled capacity (C)
                "open_route": open_route,  # (O)
                "speed": speed,  # common
            },
            batch_size=batch_size,
        )

        if self.subsample:
            # Subsample problems based on given instructions
            td = self.subsample_problems(td)
            if isinstance(self.loc_sampler, Real_World_Sampler):
                td.update(
                    {
                        "distance_matrix": distance,
                        "duration_matrix": normalized_duration,
                    }
                )
            return td
        else:
            # Not subsampling problems, i.e. return tensordict with all attributes
            return td

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

    def subsample_problems(self, td):
        """Create subproblems starting from seed probabilities depending on their variant.
        If random seed sampled in [0, 1] in batch is greater than prob, remove the constraint
        thus, if prob high, it is less likely to remove the constraint (i.e. prob=0.9, 90% chance to keep constraint)
        """
        batch_size = td.batch_size[0]

        variant_probs = torch.tensor(list(self.variant_probs.values()))

        if self.use_combinations:
            # in a batch, multiple variants combinations can be picked
            keep_mask = torch.rand(batch_size, 4) >= variant_probs  # O, TW, L, B
        else:
            # in a batch, only a variant can be picked.
            # we assign a 0.5 prob to the last variant (which is normal cvrp)
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

                    # If keep_mask[:, 4] is True, make both keep_mask[:, 0] and keep_mask[:, 1] True
                    keep_mask[:, :2] |= keep_mask[:, 4:5]
                else:
                    keep_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
                    keep_mask[torch.arange(batch_size), indices] = True
            else:
                # if the variant is specified, we keep the attributes with probability > 0
                keep_mask = torch.zeros((batch_size, 4), dtype=torch.bool)
                indices = torch.nonzero(variant_probs).squeeze()
                keep_mask[:, indices] = True

        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_window(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])

        return td

    @staticmethod
    def _default_open(td, remove):
        td["open_route"][remove] = False
        return td

    @staticmethod
    def _default_time_window(td, remove):
        default_tw = torch.zeros_like(td["time_windows"])
        default_tw[..., 1] = float("inf")
        td["time_windows"][remove] = default_tw[remove]
        td["service_time"][remove] = torch.zeros_like(td["service_time"][remove])
        return td

    @staticmethod
    def _default_distance_limit(td, remove):
        td["distance_limit"][remove] = float("inf")
        return td

    @staticmethod
    def _default_backhaul(td, remove):
        # by default, where there is a backhaul, linehaul is 0. therefore, we add backhaul to linehaul
        # and set backhaul to 0 where we want to remove backhaul
        td["demand_linehaul"][remove] = (
            td["demand_linehaul"][remove] + td["demand_backhaul"][remove]
        )
        td["demand_backhaul"][remove] = 0
        return td

    def generate_locations(self, batch_size, num_loc) -> torch.Tensor:
        """Generate seed locations.

        Returns:
            locs: [B, N+1, 2] where the first location is the depot.
        """
        locs = torch.FloatTensor(*batch_size, num_loc + 1, 2).uniform_(
            self.min_loc, self.max_loc
        )
        return locs

    def generate_demands(self, batch_size: int, num_loc: int) -> torch.Tensor:
        """Classical lineahul demand / delivery from depot (C) and backhaul demand / pickup to depot (B) generation.
        Initialize the demand for nodes except the depot, which are added during reset.
        Demand sampling Following Kool et al. (2019), demands as integers between 1 and 10.
        Generates a slightly different distribution than using torch.randint.

        Returns:
            linehaul_demand: [B, N]
            backhaul_demand: [B, N]
        """
        linehaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_demand - 1, self.max_demand - 1
        )
        linehaul_demand = (linehaul_demand.int() + 1).float()
        # Backhaul demand sampling
        backhaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_backhaul - 1, self.max_backhaul - 1
        )
        backhaul_demand = (backhaul_demand.int() + 1).float()
        is_linehaul = torch.rand(*batch_size, num_loc) > self.backhaul_ratio
        backhaul_demand = (
            backhaul_demand * ~is_linehaul
        )  # keep only values where they are not linehauls
        linehaul_demand = linehaul_demand * is_linehaul
        return linehaul_demand, backhaul_demand

    def generate_time_windows(
        self,
        locs: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time windows (TW) and service times for each location including depot.
        We refer to the generation process in "Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization"
        (Liu et al., 2024). Note that another way to generate is from "Learning to Delegate for Large-scale Vehicle Routing" (Li et al, 2021) which
        is used in "MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024). Note that however, in that case
        the distance limit would have no influence when time windows are present, since the tw for depot is the same as distance with speed=1.
        This function can be overridden for that implementation.
        See also https://github.com/RoyalSkye/Routing-MVMoE

        Args:
            locs: [B, N+1, 2] (depot, locs)
            speed: [B]

        Returns:
            time_windows: [B, N+1, 2]
            service_time: [B, N+1]
        """

        batch_size, n_loc = locs.shape[0], locs.shape[1] - 1  # no depot

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)
        d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i / speed
        tw_end = tw_start + tw_length

        # Depot tw is 0, max_time
        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),  # start
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),  # en
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)
        return time_windows, service_time  # [B, N+1, 2], [B, N+1]

    def generate_time_windows_with_duration_matrix(
        self,
        duration: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time windows (TW) and service times for each location including depot.
        We refer to the generation process in "Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization"
        (Liu et al., 2024). Note that another way to generate is from "Learning to Delegate for Large-scale Vehicle Routing" (Li et al, 2021) which
        is used in "MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024). Note that however, in that case
        the distance limit would have no influence when time windows are present, since the tw for depot is the same as distance with speed=1.
        This function can be overridden for that implementation.
        See also https://github.com/RoyalSkye/Routing-MVMoE

        Args:
            locs: [B, N+1, 2] (depot, locs)
            speed: [B]

        Returns:
            time_windows: [B, N+1, 2]
            service_time: [B, N+1]
        """

        batch_size, n_loc = duration.shape[0], duration.shape[1] - 1  # no depot

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)

        # d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        d_0i = duration[:, 0, 1:]
        d_i0 = duration[:, 1:, 0]
        d_max = torch.max(d_0i, d_i0)
        h_max = (self.max_time - service_time - tw_length) / (d_max + 1e-6) - 1
        # tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i
        # duration matrix is assymmetric, so we need to use the max of d_0i and d_i0 for always feasible time windows (vehicle should always return to depot)
        tw_start = d_0i + (h_max - 1) * d_max * torch.rand(batch_size, n_loc)
        tw_end = tw_start + tw_length

        # Depot tw is 0, max_time
        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),  # start
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),  # en
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)
        return time_windows, service_time  # [B, N+1, 2], [B, N+1]

    def generate_distance_limit(
        self, shape: Tuple[int, int], locs: torch.Tensor
    ) -> torch.Tensor:
        """Generates distance limits (L).
        The distance lower bound is dist_lower_bound = 2 * max(depot_to_location_distance),
        then the max can be max_lim = min(max_distance_limit, dist_lower_bound + EPS). Ensures feasible yet challenging
        constraints, with each instance having a unique, meaningful limit

        Returns:
            distance_limit: [B, 1]
        """
        max_dist = torch.max(torch.cdist(locs[:, 0:1], locs[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist + 1e-6
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.max_distance_limit),
            dist_lower_bound + 1e-6,
        )

        # We need to sample from the `distribution` module to get the same distribution with a tensor as input
        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ]

    def generate_open_route(self, shape: Tuple[int, int]):
        """Generate open route flags (O). Here we could have a sampler but we simply return True here so all
        routes are open. Afterwards, we subsample the problems.
        """
        return torch.ones(shape, dtype=torch.bool)

    def generate_speed(self, shape: Tuple[int, int]):
        """We simply generate the speed as constant here"""
        # in this version, the speed is constant but this class may be overridden
        return torch.full(shape, self.speed, dtype=torch.float32)

    def generate_backhaul_class(self, shape: Tuple[int, int], sample: bool = False):
        """Generate backhaul class (B) for each node. If sample is True, we sample the backhaul class
        otherwise, we return the same class for all nodes.
        - Backhaul class 1: classic backhaul (VRPB), linehauls must be served before backhauls in a route (every customer is either, not both)
        - Backhaul class 2: mixed backhaul (VRPMPD or VRPMB), linehauls and backhauls can be served in any order (every customer is either, not both)
        """
        if sample:
            return torch.randint(1, 3, shape, dtype=torch.float32)
        else:
            return torch.full(shape, self.backhaul_class, dtype=torch.float32)

    @staticmethod
    def save_data(td: TensorDict, path, compress: bool = False):
        save_tensordict_to_npz(td, path)

    @staticmethod
    def print_presets():
        for key, value in VARIANT_GENERATION_PRESETS.items():
            print(f"{key}: {value}")

    @staticmethod
    def available_variants(*args, **kwargs):
        # remove 'all', 'single_feat' from the list
        return list(VARIANT_GENERATION_PRESETS.keys())[3:]
