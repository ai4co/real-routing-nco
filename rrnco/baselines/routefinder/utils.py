import torch

from rl4co.utils.ops import gather_by_index, unbatchify


def evaluate(
    model,
    td,
    num_augment=8,
    num_starts=None,
):
    with torch.inference_mode():
        n_start = model.env.get_num_starts(td) if num_starts is None else num_starts

        if num_augment > 1:
            td = model.augment(td)

        # Evaluate policy
        out = model.policy(
            td, model.env, phase="test", num_starts=n_start, return_actions=True
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (num_augment, n_start))

        if n_start > 1:
            # max multi-start reward
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})

            if out.get("actions", None) is not None:
                # Reshape batch to [batch_size, num_augment, num_starts, ...]
                actions = unbatchify(out["actions"], (num_augment, n_start))
                out.update(
                    {
                        "best_multistart_actions": gather_by_index(
                            actions, max_idxs, dim=max_idxs.dim()
                        )
                    }
                )
                out["actions"] = actions

        # Get augmentation score only during inference
        if num_augment > 1:
            # If multistart is enabled, we use the best multistart rewards
            reward_ = max_reward if n_start > 1 else reward
            max_aug_reward, max_idxs = reward_.max(dim=1)
            out.update({"max_aug_reward": max_aug_reward})

            if out.get("actions", None) is not None:
                actions_ = (
                    out["best_multistart_actions"] if n_start > 1 else out["actions"]
                )
                out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

        return out
