import os

from typing import List, Optional, Union

import numpy as np
import torch

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from .generator import RMTVRPGenerator
from .generator_lazy import LazyRMTVRPGenerator
from .selectstartnodes import get_select_start_nodes_fn

log = get_pylogger(__name__)


class RMTVRPEnv(RL4COEnvBase):
    r"""MTVRPEnv is a Multi-Task VRP environment which can take any combination of the following constraints:
    Note: "Real" stands for actual distance and duration matrix

    Features:

    - *Capacity (C)*
        - Each vehicle has a maximum capacity :math:`Q`, restricting the total load that can be in the vehicle at any point of the route.
        - The route must be planned such that the sum of demands and pickups for all customers visited does not exceed this capacity.
    - *Time Windows (TW)*
        - Every node :math:`i` has an associated time window :math:`[e_i, l_i]` during which service must commence.
        - Additionally, each node has a service time :math:`s_i`. Vehicles must reach node :math:`i` within its time window; early arrivals must wait at the node location until time :math:`e_i`.
    - *Open Routes (O)*
        - Vehicles are not required to return to the depot after serving all customers.
        - Note that this does not need to be counted as a constraint since it can be modelled by setting zero costs on arcs returning to the depot :math:`c_{i0} = 0` from any customer :math:`i \in C`, and not counting the return arc as part of the route.
    - *Backhauls (B)*
        - Backhauls generalize demand to also account for return shipments. Customers are either linehaul or backhaul customers.
        - Linehaul customers require delivery of a demand :math:`q_i > 0` that needs to be transported from the depot to the customer, whereas backhaul customers need a pickup of an amount :math:`p_i > 0` that is transported from the client back to the depot.
        - It is possible for vehicles to serve a combination of linehaul and backhaul customers in a single route, but then any linehaul customers must precede the backhaul customers in the route.
    - *Duration Limits (L)*
        - Imposes a limit on the total travel duration (or length) of each route, ensuring a balanced workload across vehicles.
    - *Mixed (M) Backhaul (M)*
        - This is a variant of the backhaul constraint where the vehicle can pick up and deliver linehaul customers in any order.
        - However, we need to ensure that the vehicle has enough capacity to deliver the linehaul customers and that the vehicle can pick up backhaul customers only if it has enough capacity to deliver the linehaul customers.

    The environment covers the following 16 variants depending on the data generation:

    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRP Variant  || Capacity (C) | Open Route (O) | Backhaul (B) | Duration Limit (L) | Time Window (TW) |
    +==============++==============+================+==============+====================+==================+
    | CVRP         || ✔            |                |              |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRP         || ✔            | ✔              |              |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPB         || ✔            |                | ✔            |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPL         || ✔            |                |              | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPTW        || ✔            |                |              |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPTW       || ✔            | ✔              |              |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPB        || ✔            | ✔              | ✔            |                    |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPL        || ✔            | ✔              |              | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBL        || ✔            |                | ✔            | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBTW       || ✔            |                | ✔            |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPLTW       || ✔            |                |              | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBL       || ✔            | ✔              | ✔            | ✔                  |                  |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBTW      || ✔            | ✔              | ✔            |                    | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPLTW      || ✔            | ✔              |              | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | VRPBLTW      || ✔            |                | ✔            | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+
    | OVRPBLTW     || ✔            | ✔              | ✔            | ✔                  | ✔                |
    +--------------++--------------+----------------+--------------+--------------------+------------------+

    Additionally, with the mixed backhaul (M) variant, we obtain 24 variants.

    You may also check out `"Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization" (Liu et al., 2024) <https://arxiv.org/abs/2402.16891>`_
    and `"MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024) <https://arxiv.org/abs/2405.01029>`_.


    Note:
        Have a look at https://pyvrp.org/ for more information about VRP and its variants and their solutions. Kudos to their help and great job!

    Args:
        generator: Generator for the environment, see :class:`RMTVRPGenerator`.
        generator_params: Parameters for the generator.
    """

    name = "rcvrptw"

    def __init__(
        self,
        generator: Union[RMTVRPGenerator, LazyRMTVRPGenerator] = None,
        generator_params: Union[dict, RMTVRPGenerator, LazyRMTVRPGenerator] = {},
        select_start_nodes_fn: Union[str, callable] = "all",
        normalize: bool = True,
        check_solution: bool = False,
        load_solutions: bool = True,
        solution_fname: str = "_sol_pyvrp.npz",
        **kwargs,
    ):
        super().__init__(check_solution=check_solution, **kwargs)
        if generator is None:
            # generator_params가 이미 인스턴스화된 객체인지 확인
            if isinstance(generator_params, (RMTVRPGenerator, LazyRMTVRPGenerator)):
                generator = generator_params
            elif isinstance(generator_params, dict):
                # 딕셔너리인 경우 _target_을 확인하여 적절한 생성기 선택
                if (
                    generator_params.get("_target_")
                    == "rrnco.envs.rmtvrp.generator_lazy.LazyRMTVRPGenerator"
                ):
                    # _target_ 키를 제거하고 LazyRMTVRPGenerator 사용
                    generator_params_copy = generator_params.copy()
                    generator_params_copy.pop("_target_", None)
                    generator = LazyRMTVRPGenerator(**generator_params_copy)
                else:
                    generator = RMTVRPGenerator(**generator_params)
            else:
                # 기본 생성기 사용
                generator = RMTVRPGenerator()

        if check_solution:
            log.warning(
                "Solution checking is enabled. This may slow down the environment."
                " We recommend disabling this for training by passing `check_solution=False`."
            )

        self.generator = generator
        if isinstance(select_start_nodes_fn, str):
            self.select_start_nodes_fn = get_select_start_nodes_fn(select_start_nodes_fn)
        else:
            self.select_start_nodes_fn = select_start_nodes_fn

        self.solution_fname = solution_fname
        self.load_solutions = load_solutions
        self.normalize = normalize
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        prev_node, curr_node = td["current_node"], td["action"]
        b_idx = torch.arange(td.batch_size[0])
        distance = td["distance_matrix"][b_idx, prev_node, curr_node]
        duration = td["duration_matrix"][b_idx, prev_node, curr_node]

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        # we cannot start before we arrive and we should start at least at start times
        curr_time = (curr_node[:, None] != 0) * (
            torch.max(td["current_time"] + duration[:, None], start_times) + service_time
        )

        # Update current route length (reset at depot)
        curr_route_length = (curr_node[:, None] != 0) * (
            td["current_route_length"] + distance[:, None]
        )

        # Linehaul (delivery) demands
        selected_demand_linehaul = gather_by_index(
            td["demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["demand_backhaul"], curr_node, dim=1, squeeze=False
        )

        # Backhaul (pickup) demands
        # this holds for backhaul_classes 0, 1, and 2:
        used_capacity_linehaul = (curr_node[:, None] != 0) * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = (curr_node[:, None] != 0) * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        # Done when all customers are visited
        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(
            done
        ).float()  # we use the `get_reward` method to compute the reward

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict],
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        # Demands: linehaul (C) and backhaul (B). Backhaul defaults to 0
        demand_linehaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), td["demand_linehaul"]],
            dim=1,
        )
        demand_backhaul = td.get(
            "demand_backhaul",
            torch.zeros_like(td["demand_linehaul"]),
        )
        demand_backhaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), demand_backhaul], dim=1
        )
        # Backhaul class (MB). 1 is the default backhaul class
        backhaul_class = td.get(
            "backhaul_class",
            torch.full((*batch_size, 1), 1, dtype=torch.int32),
        )

        # Time windows (TW). Defaults to [0, inf] and service time to 0
        time_windows = td.get("time_windows", None)
        if time_windows is None:
            time_windows = torch.zeros_like(td["locs"])
            time_windows[..., 1] = float("inf")
        service_time = td.get("service_time", torch.zeros_like(demand_linehaul))

        # Open (O) route. Defaults to 0
        open_route = td.get(
            "open_route", torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool)
        )

        # Distance limit (L). Defaults to inf
        distance_limit = td.get(
            "distance_limit", torch.full_like(demand_linehaul[..., :1], float("inf"))
        )

        #######################################
        # Real-routing part
        # Distances and durations are matrices
        #######################################

        # Distance matrix
        if "distance_matrix" in td:
            distance_matrix = td["distance_matrix"]
        else:
            distance_matrix = torch.cdist(td["locs"], td["locs"], p=2)

        if self.normalize:
            # Calculate min and max distances for normalization
            min_distance = distance_matrix.amin(dim=(-2, -1), keepdim=True)
            max_distance = distance_matrix.amax(dim=(-2, -1), keepdim=True)

            # # Normalize distance using min-max scaling
            distance_matrix = (distance_matrix - min_distance) / (
                max_distance - min_distance + 1e-6
            )
            distance_matrix = distance_matrix.to(dtype=torch.float32)

        # Duration matrix
        speed = td.get("speed", torch.ones_like(demand_linehaul[..., :1]))
        if "duration_matrix" in td:
            duration_matrix = td["duration_matrix"]
        else:
            duration_matrix = distance_matrix / speed[:, None]

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "distance_matrix": distance_matrix,
                "duration_matrix": duration_matrix,
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "service_time": service_time,
                "open_route": open_route,
                "time_windows": time_windows,
                "speed": speed,
                "vehicle_capacity": td.get(
                    "vehicle_capacity", torch.ones_like(demand_linehaul[..., :1])
                ),
                "capacity_original": td.get(
                    "capacity_original", torch.ones_like(demand_linehaul[..., :1])
                ),
                "current_node": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_route_length": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),  # for distance limits
                "current_time": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),  # for time windows
                "used_capacity_backhaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),  # for capacity constraints in backhaul
                "used_capacity_linehaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),  # for capacity constraints in linehaul
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ),
            },
            batch_size=batch_size,
            device=device,
        )
        if self.normalize:
            td_reset.update(
                {
                    "min_distance": min_distance.squeeze(-1).squeeze(-1),
                    "max_distance": max_distance.squeeze(-1).squeeze(-1),
                }
            )

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # Collect distance and duration matrices to next nodes and from next nodes to depot
        curr_node = td["current_node"]
        b_idx = torch.arange(td.batch_size[0])
        dist_ij = td["distance_matrix"][b_idx, curr_node, :]
        dist_j0 = td["distance_matrix"][:, :, 0]  # distance as ..., current, next
        dur_ij = td["duration_matrix"][b_idx, curr_node, :]
        dur_j0 = td["duration_matrix"][:, :, 0]

        # Time constraint (TW):
        early_tw, late_tw = (
            td["time_windows"][..., 0],
            td["time_windows"][..., 1],
        )
        arrival_time = td["current_time"] + dur_ij
        # can reach in time -> only need to *start* in time
        can_reach_customer = arrival_time < late_tw
        # we must ensure that we can return to depot in time *if* route is closed
        # i.e. start time + service time + time back to depot < late_tw
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td["service_time"] + (dur_j0)
        ) * ~td["open_route"] < late_tw[..., 0:1]

        # Distance limit (L): do not add distance to depot if open route (O)
        exceeds_dist_limit = (
            td["current_route_length"] + dist_ij + (dist_j0 * ~td["open_route"])
            > td["distance_limit"]
        )

        # Capacity constraints linehaul (C) and backhaul (B)
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )

        # Backhaul class 1 (classical backhaul) (B)
        # every customer is either backhaul or linehaul, all linehauls are visited before backhauls
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[
            ..., None
        ]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["demand_backhaul"],
                idx=td["current_node"],
                dim=1,
                squeeze=False,
            )
            > 0
        )
        meets_demand_constraint_backhaul_1 = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))

        # Backhaul class 2 (mixed pickup and delivery / mixed backhaul) (MB)
        # to serve linehaul customers we additionally need to check the remaining capacity in the vehicle
        # capacity is vehicle_capacity-used_capacity_backhauls, as all used_capacity_linehaul at this point have already been *delivered*
        cannot_serve_linehaul = (
            td["demand_linehaul"] > td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )

        # Now we merge the constraints of backhaul class 1 and 2 depending on the backhaul class
        meets_demand_constraint = (
            (td["backhaul_class"] == 1) & meets_demand_constraint_backhaul_1
        ) | ((td["backhaul_class"] == 2) & meets_demand_constraint_backhaul_2)

        # Condense constraints
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

        # Mask depot: don't visit depot if coming from there and there are still customer nodes I can visit
        can_visit[:, 0] = ~((td["current_node"] == 0) & (can_visit[:, 1:].sum(-1) > 0))
        return can_visit

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        cost_matrix = td["distance_matrix"]  # [b, n + 1, n + 1] (batch, from, to)
        # set last dim of cost matrix back to depot to 0 if route is open (as per Niels' good suggestion)
        cost_matrix[:, :, 0] = cost_matrix[:, :, 0] * ~td["open_route"]

        # append depot to actions
        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)  # [b, seq_len]

        # gather cost matrix in order of tour
        distances = gather_by_index(
            gather_by_index(cost_matrix, go_from, dim=1, squeeze=False),
            go_to,
            dim=2,
            squeeze=False,
        ).squeeze(-1)

        if self.normalize:
            normalized_distances = -distances.sum(-1)  # reward is negative cost
            real_distances = (
                normalized_distances * (td["max_distance"] - td["min_distance"] + 1e-6)
                + td["min_distance"]
            )
            return real_distances, normalized_distances

        return -distances.sum(-1)  # reward is negative cost

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        raise NotImplementedError(
            "This method is not implemented yet. Please modify considering the dist and dur matrices."
        )

        batch_size, n_loc = td["demand_linehaul"].size()
        dist_ij, dist_j0, dur_ij, dur_j0 = RMTVRPEnv._get_distance_durations_ij_j0(td)

        n_loc -= 1  # exclude depot
        sorted_pi = actions.data.sort(1)[0]

        # all customer nodes visited exactly once
        assert (
            torch.arange(1, n_loc + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, n_loc)
            == sorted_pi[:, -n_loc:]
        ).all() and (sorted_pi[:, :-n_loc] == 0).all(), "Invalid tour"

        # Distance limits (L)
        assert (td["distance_limit"] >= 0).all(), "Distance limits must be non-negative."

        # Time windows (TW)
        # d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(td["service_time"] >= 0.0), "Service time must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "there are unfeasible time windows"
        assert torch.all(
            td["time_windows"][..., :, 0] + dur_j0 + td["service_time"]
            <= td["time_windows"][..., 0, 1, None]
        ), "vehicle cannot perform service and get back to depot in time."
        # check individual time windows
        curr_time = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros(batch_size, dtype=torch.int64, device=td.device)
        curr_length = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            # curr_loc = gather_by_index(td["locs"], curr_node)
            # next_loc = gather_by_index(td["locs"], next_node)
            # dist = get_distance(curr_loc, next_loc)\
            dist = dist_ij[:, ii]

            # distance limit (L)
            curr_length = curr_length + dist * ~(
                td["open_route"].squeeze(-1) & (next_node == 0)
            )  # do not count back to depot for open route
            assert torch.all(
                curr_length <= td["distance_limit"].squeeze(-1)
            ), "Route exceeds distance limit"
            curr_length[next_node == 0] = 0.0  # reset length for depot

            curr_time = torch.max(
                curr_time + dist, gather_by_index(td["time_windows"], next_node)[..., 0]
            )
            assert torch.all(
                curr_time <= gather_by_index(td["time_windows"], next_node)[..., 1]
            ), "vehicle cannot start service before deadline"
            curr_time = curr_time + gather_by_index(td["service_time"], next_node)
            curr_node = next_node
            curr_time[curr_node == 0] = 0.0  # reset time for depot

        # Demand constraints (C) and (B) and (MB)
        # we keep track of the current picked up linehaul and backhaul
        # and the used capacity of both
        demand_l = td["demand_linehaul"].gather(dim=1, index=actions)
        demand_b = td["demand_backhaul"].gather(dim=1, index=actions)
        used_cap_l = torch.zeros_like(td["demand_linehaul"][:, 0])
        used_cap_b = torch.zeros_like(td["demand_backhaul"][:, 0])
        for ii in range(actions.size(1)):
            # reset at depot
            used_cap_l = used_cap_l * (actions[:, ii] != 0)
            used_cap_b = used_cap_b * (actions[:, ii] != 0)
            # increase counters
            used_cap_l += demand_l[:, ii]
            used_cap_b += demand_b[:, ii]

            # For backhaul_class 1 (B), we must ensure that if we are carrying backhaul, we are not picking up linehaul
            assert (
                (td["backhaul_class"] == 2)
                | (used_cap_b == 0)
                | ((td["backhaul_class"] == 1) & ~(demand_l[:, ii] > 0))
            ).all(), "Cannot pick up linehaul while carrying backhaul due to precedence constraints"

            # For backhaul_class 2 (MB), we cannot pick up linehaul if the used capacity of backhaul is already at the vehicle capacity
            # also, cannot pick up other backhauls if we are full
            assert (
                (td["backhaul_class"] == 1)
                | (used_cap_b == 0)
                | (
                    (td["backhaul_class"] == 2)
                    & (used_cap_b + demand_l[:, ii] <= td["vehicle_capacity"])
                )
            ).all(), "Cannot deliver linehaul, not enough load"

            # Assertions: total used linehaul and backhaul capacity should not exceed vehicle capacity
            assert (
                used_cap_l <= td["vehicle_capacity"]
            ).all(), "Used more linehaul than capacity: {} / {}".format(
                used_cap_l, td["vehicle_capacity"]
            )
            assert (
                used_cap_b <= td["vehicle_capacity"]
            ).all(), "Used more backhaul than capacity: {} / {}".format(
                used_cap_b, td["vehicle_capacity"]
            )

    def get_num_starts(self, td):
        return self.select_start_nodes_fn.get_num_starts(td)

    def select_start_nodes(self, td, num_starts):
        return self.select_start_nodes_fn(td, num_starts, self.get_num_starts(td))

    @staticmethod
    def render(*args, **kwargs):
        """Simple wrapper for render function"""
        from .render import render

        return render(*args, **kwargs)

    def _make_spec(self, td_params: TensorDict):
        # TODO: include extra vars (but we don't really need them for now)
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=self.generator.min_loc,
                high=self.generator.max_loc,
                shape=(self.generator.num_loc + 1, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
                device=self.device,
            ),
            demand_linehaul=BoundedTensorSpec(
                low=-self.generator.capacity,
                high=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),  # demand is only for customers
                dtype=torch.float32,
                device=self.device,
            ),
            demand_backhaul=BoundedTensorSpec(
                low=-self.generator.capacity,
                high=self.generator.max_demand,
                shape=(self.generator.num_loc, 1),  # demand is only for customers
                dtype=torch.float32,
                device=self.device,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.generator.num_loc + 1, 1),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            low=0,
            high=self.generator.num_loc + 1,
            shape=(1,),
            dtype=torch.int64,
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), dtype=torch.float32, device=self.device
        )
        self.done_spec = UnboundedDiscreteTensorSpec(
            shape=(1,), dtype=torch.bool, device=self.device
        )

    @staticmethod
    def check_variants(td):
        """Check if the problem has the variants"""
        has_open = td["open_route"].squeeze(-1)
        has_tw = (td["time_windows"][..., 1] != float("inf")).any(-1)
        has_limit = (td["distance_limit"] != float("inf")).squeeze(-1)
        has_backhaul = (td["demand_backhaul"] != 0).any(-1)
        backhaul_class = td.get("backhaul_class", torch.full_like(has_open, 1))
        return has_open, has_tw, has_limit, has_backhaul, backhaul_class

    @staticmethod
    def get_variant_names(td: TensorDict) -> Union[str, List[str]]:
        (
            has_open,
            has_time_window,
            has_duration_limit,
            has_backhaul,
            backhaul_class,
        ) = RMTVRPEnv.check_variants(td)

        def _name(o, b, bc, l_, tw):
            if not o and not b and not l_ and not tw:
                instance_name = "CVRP"
            else:
                instance_name = "VRP"
                if o:
                    instance_name = "O" + instance_name
                if b:
                    if bc == 2:  # mixed backhaul
                        instance_name += "M"
                    instance_name += "B"
                if l_:
                    instance_name += "L"
                if tw:
                    instance_name += "TW"
            return instance_name

        if len(has_open.shape) == 0:
            return _name(
                has_open,
                has_backhaul,
                backhaul_class,
                has_duration_limit,
                has_time_window,
            )
        else:
            return [
                _name(o, b, bc, l_, tw)
                for o, b, bc, l_, tw in zip(
                    has_open,
                    has_backhaul,
                    backhaul_class,
                    has_duration_limit,
                    has_time_window,
                )
            ]

    def print_presets(self):
        self.generator.print_presets()

    def available_variants(self):
        return self.generator.available_variants()

    def load_data(self, fpath, batch_size=[]):
        """Dataset loading from file"""
        td = load_npz_to_tensordict(fpath)
        if self.load_solutions:
            # Load solutions if they exist depending on the file name
            solution_fpath = fpath.replace(".npz", self.solution_fname)
            if os.path.exists(solution_fpath):
                sol = np.load(solution_fpath)
                sol_dict = {}
                for key, value in sol.items():
                    if isinstance(value, np.ndarray) and len(value.shape) > 0:
                        if value.shape[0] == td.batch_size[0]:
                            key = "costs_bks" if key == "costs" else key
                            key = "actions_bks" if key == "actions" else key
                            sol_dict[key] = torch.tensor(value)
                td.update(sol_dict)
            else:
                log.warning(f"No solution file found at {solution_fpath}")
        return td
