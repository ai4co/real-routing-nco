import torch.nn as nn

from rl4co.models.nn.env_embeddings.context import TSPContext, VRPContext, VRPTWContext
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding

from .atsp import ATSPInitEmbedding
from .context import MTVRPContextEmbedding
from .rcvrp import RVRPInitEmbedding
from .rcvrptw import RVRPTWInitEmbedding


def env_embedding_register(
    env_name: str, config: dict, registry_default: dict, registry_custom: dict = None
) -> nn.Module:
    # Merge dictionaries if registry is not None
    embedding_registry = (
        {**registry_default, **registry_custom}
        if registry_custom is not None
        else registry_default
    )
    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available context embeddings: {embedding_registry.keys()}"
        )
    return embedding_registry[env_name](**config)


def env_init_embedding(env_name: str, config: dict, registry: dict = None) -> nn.Module:
    """Register init embedding of the environment"""

    emb_registry = {
        "atsp": ATSPInitEmbedding,
        "rcvrp": RVRPInitEmbedding,
        "rcvrptw": RVRPTWInitEmbedding,
    }
    return env_embedding_register(env_name, config, emb_registry, registry)


def env_context_embedding(
    env_name: str, config: dict, registry: dict = None
) -> nn.Module:
    """Register context of the environment"""
    emb_registry = {
        "atsp": TSPContext,
        "rcvrp": VRPContext,
        "rcvrptw": MTVRPContextEmbedding,
    }
    return env_embedding_register(env_name, config, emb_registry, registry)


def env_dynamic_embedding(
    env_name: str, config: dict, registry: dict = None
) -> nn.Module:
    """Register dynamic embedding of the environment.
    The problem in our case does not change, but this can be easily extended
    for stochastic environments.
    """
    emb_registry = {
        "atsp": StaticEmbedding,
        "rcvrp": StaticEmbedding,
        "rcvrptw": StaticEmbedding,
    }
    # if not in key, just return static embedding
    if env_name not in emb_registry.keys():
        return StaticEmbedding(**config)
    return env_embedding_register(env_name, config, emb_registry, registry)
