from typing import Optional, Union

import torch

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from .generator import RCVRPGenerator
from .generator_lazy import LazyRCVRPGenerator
from .render import render

try:
    from .local_search import local_search
except Exception:
    # In case some dependencies are not installed (e.g., pyvrp)
    local_search = None
log = get_pylogger(__name__)


class RCVRPEnv(RL4COEnvBase):
    """Capacitated Vehicle Routing Problem (CVRP) environment.
    At each step, the agent chooses a customer to visit depending on the current location and the remaining capacity.
    When the agent visits a customer, the remaining capacity is updated. If the remaining capacity is not enough to
    visit any customer, the agent must go back to the depot. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - location of the depot.
        - locations and demand of each customer.
        - current location of the vehicle.
        - the remaining customer of the vehicle,

    Constraints:
        - the tour starts and ends at the depot.
        - each customer must be visited exactly once.
        - the vehicle cannot visit customers exceed the remaining capacity.
        - the vehicle can return to the depot to refill the capacity.

    Finish Condition:
        - the vehicle has visited all customers and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "rcvrp"

    def __init__(
        self,
        generator: Union[RCVRPGenerator, LazyRCVRPGenerator] = None,
        generator_params: Union[dict, RCVRPGenerator, LazyRCVRPGenerator] = {},
        cdist_compute_mode: str = "use_mm_for_euclid_dist",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            # generator_params가 이미 인스턴스화된 객체인지 확인
            if isinstance(generator_params, (RCVRPGenerator, LazyRCVRPGenerator)):
                generator = generator_params
            elif isinstance(generator_params, dict):
                # 딕셔너리인 경우 _target_을 확인하여 적절한 생성기 선택
                if (
                    generator_params.get("_target_")
                    == "rrnco.envs.rcvrp.generator_lazy.LazyRCVRPGenerator"
                ):
                    # _target_ 키를 제거하고 LazyRCVRPGenerator 사용
                    generator_params_copy = generator_params.copy()
                    generator_params_copy.pop("_target_", None)
                    generator = LazyRCVRPGenerator(**generator_params_copy)
                else:
                    generator = RCVRPGenerator(**generator_params)
            else:
                # 기본 생성기 사용
                generator = RCVRPGenerator()

        self.generator = generator
        self.cdist_compute_mode = cdist_compute_mode
        self.normalize = normalize
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]  # Add dimension for step
        n_loc = td["demand"].size(-1)  # Excludes depot

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node, 1)

        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device
        # if we don't have a distance in td, we need to generate it
        if "distance_matrix" not in td:
            assert "locs" in td, "locs must be provided to generate distance"
            distance = torch.cdist(
                td["locs"], td["locs"], p=2, compute_mode=self.cdist_compute_mode
            )
        else:
            distance = td["distance_matrix"]
        if self.normalize:
            # Calculate min and max distances for normalization
            min_distance = distance.amin(dim=(-2, -1), keepdim=True)
            max_distance = distance.amax(dim=(-2, -1), keepdim=True)

            # Normalize distance using min-max scaling
            distance = (distance - min_distance) / (max_distance - min_distance + 1e-6)
            distance = distance.to(dtype=torch.float32)

        # Common fields for TensorDict
        td_reset_data = {
            "locs": torch.cat((td["depot"][:, None, :], td["locs"]), dim=-2),
            "distance_matrix": distance,
            "demand": td["demand"],
            "current_node": torch.zeros(*batch_size, 1, dtype=torch.long, device=device),
            "used_capacity": torch.zeros((*batch_size, 1), device=device),
            "vehicle_capacity": torch.full(
                (*batch_size, 1), self.generator.vehicle_capacity, device=device
            ),
            "visited": torch.zeros(
                (*batch_size, td["locs"].shape[-2] + 1), dtype=torch.uint8, device=device
            ),
        }

        # Add normalization metadata if applicable
        if self.normalize:
            td_reset_data.update(
                {
                    "min_distance": min_distance.squeeze(-1).squeeze(-1),
                    "max_distance": max_distance.squeeze(-1).squeeze(-1),
                }
            )

        # Create TensorDict and set action mask
        td_reset = TensorDict(td_reset_data, batch_size=batch_size)
        td_reset.set("action_mask", self.get_action_mask(td_reset))

        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"]

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)[
            :, None
        ]
        return ~torch.cat((mask_depot, mask_loc), -1)

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Append depot to actions and get sequence of locations
        distance = td["distance_matrix"]  # [b, n + 1, n + 1] (batch, from, to)

        # append depot to actions
        go_from = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=1)
        go_to = torch.roll(go_from, -1, dims=1)  # [b, seq_len]

        # gather cost matrix in order of tour
        distances = gather_by_index(
            gather_by_index(distance, go_from, dim=1, squeeze=False),
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
        """Check that solution is valid: nodes are not visited twice except depot and capacity is not exceeded"""
        # Check if tour is valid, i.e. contain 0 to n-1
        batch_size, graph_size = td["demand"].size()
        sorted_pi = actions.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat((-td["vehicle_capacity"], td["demand"]), 1)
        d = demand_with_depot.gather(1, actions)

        used_cap = torch.zeros_like(td["demand"][:, 0])
        for i in range(actions.size(1)):
            used_cap += d[
                :, i
            ]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (
                used_cap <= td["vehicle_capacity"] + 1e-5
            ).all(), "Used more than capacity"

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        td_load = load_npz_to_tensordict(fpath)
        td_load.set("demand", td_load["demand"] / td_load["capacity"][:, None])
        return td_load

    def _make_spec(self, generator: RCVRPGenerator):
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            demand=Bounded(
                low=-generator.capacity,
                high=generator.max_demand,
                shape=(generator.num_loc + 1, 1),
                dtype=torch.float32,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def replace_selected_actions(
        self,
        cur_actions: torch.Tensor,
        new_actions: torch.Tensor,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace selected current actions with updated actions based on `selection_mask`.

        Args:
            cur_actions [batch_size, num_loc]
            new_actions [batch_size, num_loc]
            selection_mask [batch_size,]
        """
        diff_length = cur_actions.size(-1) - new_actions.size(-1)
        if diff_length > 0:
            new_actions = torch.nn.functional.pad(
                new_actions, (0, diff_length, 0, 0), mode="constant", value=0
            )
        elif diff_length < 0:
            cur_actions = torch.nn.functional.pad(
                cur_actions, (0, -diff_length, 0, 0), mode="constant", value=0
            )
        cur_actions[selection_mask] = new_actions[selection_mask]
        return cur_actions

    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        assert (
            local_search is not None
        ), "Cannot import local_search module. Check if `pyvrp` is installed."
        return local_search(td, actions, **kwargs)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
