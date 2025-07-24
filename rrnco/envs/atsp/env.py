from typing import Optional, Union

import torch

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from .generator import ATSPGenerator
from .generator_lazy import LazyATSPGenerator
from .render import render

log = get_pylogger(__name__)


class ATSPEnv(RL4COEnvBase):
    """Asymmetric Traveling Salesman Problem (ATSP) environment
    At each step, the agent chooses a customer to visit. The reward is 0 unless the agent visits all the customers.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.
    Unlike the TSP, the distance matrix is asymmetric, i.e., the distance from A to B is not necessarily the same as the distance from B to A.

    Observations:
        - distance matrix between customers
        - the current customer
        - the first customer (for calculating the reward)
        - the remaining unvisited customers

    Constraints:
        - the tour starts and ends at the same customer.
        - each customer must be visited exactly once.

    Finish Condition:
        - the agent has visited all customers.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: ATSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "atsp"

    def __init__(
        self,
        generator: Union[ATSPGenerator, LazyATSPGenerator] = None,
        generator_params: Union[dict, ATSPGenerator, LazyATSPGenerator] = {},
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            # generator_params가 이미 인스턴스화된 객체인지 확인
            if isinstance(generator_params, (ATSPGenerator, LazyATSPGenerator)):
                generator = generator_params
            elif isinstance(generator_params, dict):
                # 딕셔너리인 경우 _target_을 확인하여 적절한 생성기 선택
                if (
                    generator_params.get("_target_")
                    == "rrnco.envs.atsp.generator_lazy.LazyATSPGenerator"
                ):
                    # _target_ 키를 제거하고 LazyATSPGenerator 사용
                    generator_params_copy = generator_params.copy()
                    generator_params_copy.pop("_target_", None)
                    generator = LazyATSPGenerator(**generator_params_copy)
                else:
                    generator = ATSPGenerator(**generator_params)
            else:
                # 기본 생성기 사용
                generator = ATSPGenerator()

        self.generator = generator
        self._make_spec(self.generator)
        self.normalize = normalize

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]

        # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) <= 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize distance matrix
        distance = td["distance_matrix"]
        device = td.device
        if self.normalize:
            # Calculate min and max distances for normalization
            min_distance = distance.amin(dim=(-2, -1), keepdim=True)
            max_distance = distance.amax(dim=(-2, -1), keepdim=True)

            # Normalize distance using min-max scaling
            distance = (distance - min_distance) / (max_distance - min_distance + 1e-6)
            distance = distance.to(dtype=torch.float32)

        # Other variables
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, self.generator.num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Common fields for TensorDict
        td_reset_data = {
            "distance_matrix": distance,
            "first_node": current_node,
            "current_node": current_node,
            "i": i,
            "action_mask": available,
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

        return td_reset

    def _make_spec(self, generator: ATSPGenerator):
        self.observation_spec = Composite(
            distance_matrix=Bounded(
                low=generator.min_dist,
                high=generator.max_dist,
                shape=(generator.num_loc, generator.num_loc),
                dtype=torch.float32,
            ),
            first_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        distance_matrix = td["distance_matrix"]

        # Get indexes of tour edges
        nodes_src = actions
        nodes_tgt = torch.roll(actions, -1, dims=1)
        batch_idx = torch.arange(
            distance_matrix.shape[0], device=distance_matrix.device
        ).unsqueeze(1)
        # return negative tour length
        if self.normalize:
            normalized_distances = -distance_matrix[batch_idx, nodes_src, nodes_tgt].sum(
                -1
            )
            real_distances = (
                normalized_distances * (td["max_distance"] - td["min_distance"] + 1e-6)
                + td["min_distance"]
            )
            return real_distances, normalized_distances
        return -distance_matrix[batch_idx, nodes_src, nodes_tgt].sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    @staticmethod
    def render(td, actions=None, ax=None):
        return render(td, actions, ax)
