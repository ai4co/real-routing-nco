from typing import Callable, Optional, Union

import torch.nn as nn

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.autoregressive import AutoregressivePolicy
from rl4co.utils.decoding import get_log_likelihood
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from rrnco.baselines.MatNet.decoder import MatNetDecoder
from rrnco.baselines.MatNet.encoder import MatNetEncoder

from .decoding import DecodingStrategy, get_decoding_strategy

log = get_pylogger(__name__)


class MatNetPolicy(AutoregressivePolicy):
    """MatNet Policy from Kwon et al., 2021.
    Reference: https://arxiv.org/abs/2106.11113

    Warning:
        This implementation is under development and subject to change.

    Args:
        env_name: Name of the environment used to initialize embeddings
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        **kwargs: keyword arguments passed to the `AutoregressivePolicy`

    Default paarameters are adopted from the original implementation.
    """

    def __init__(
        self,
        env_name: str = "atsp",
        embed_dim: int = 256,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        sdpa_fn_decoder: Callable = None,
        mask_inner: bool = True,
        num_encoder_layers: int = 5,
        num_heads: int = 16,
        init_embedding_kwargs: dict = {},
        normalization: str = "instance",
        use_graph_context: bool = False,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        if encoder is None:
            encoder = MatNetEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                env_name=env_name,
                init_embedding_kwargs=init_embedding_kwargs,
                bias=bias,
            )

        if decoder is None:
            decoder = MatNetDecoder(
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                embed_dim=embed_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,
                sdpa_fn=sdpa_fn_decoder,
                mask_inner=mask_inner,
            )

        super(MatNetPolicy, self).__init__(
            env_name=env_name,
            encoder=encoder,
            decoder=decoder,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        calc_scaled_reward: bool = True,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            if env.normalize:
                real_distances, normalized_distances = env.get_reward(td, actions)
                td.set("reward", real_distances)
            else:
                td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }
        if env.normalize:
            outdict["normalized_reward"] = normalized_distances
        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict
