import numpy as np
import torch

from tensordict import TensorDict
from torch import Tensor


def mtvrp2anyvrp(td: TensorDict) -> TensorDict:
    """
    We simply transform the data to the format the current PyVRP API expects
    """
    td_ = td.clone().cpu()
    td_.set("durations", td_["service_time"])
    if "distance_matrix" not in td_:
        dist_mat = torch.cdist(td_["locs"], td_["locs"])
        num_loc = dist_mat.shape[-1]
        # note: if we don't do this, PyVRP may complain diagonal is not 0.
        # i guess it is because of some conversion from floating point to integer
        dist_mat[:, torch.arange(num_loc), torch.arange(num_loc)] = 0
    else:
        dist_mat = td_["distance_matrix"]
    if "duration_matrix" in td_:
        dur_mat = td_["duration_matrix"]
    else:
        dur_mat = dist_mat.clone()
    td_.set("distance_matrix", dist_mat)
    td_.set("duration_matrix", dur_mat)
    backhaul_class = td.get("backhaul_class", torch.ones(td_.batch_size[0], 1))
    td_.set("backhaul_class", backhaul_class)
    return td_


def scale(data: Tensor, scaling_factor: int):
    """
    Scales ands rounds data to integers so PyVRP can handle it.
    """
    array = (data * scaling_factor).numpy().round()
    array = np.where(array == np.inf, np.iinfo(np.int32).max, array)
    array = array.astype(int)

    if array.size == 1:
        return array.item()

    return array
