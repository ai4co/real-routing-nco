def patch_torchrl_specs():

    import torchrl.data.tensor_specs as _tspecs

    # TorchRL renamed Spec classes; patch old names for checkpoint compatibility
    for _old, _new in [
        ("CompositeSpec", "Composite"),
        ("BoundedTensorSpec", "Bounded"),
        ("UnboundedContinuousTensorSpec", "Unbounded"),
        ("UnboundedDiscreteTensorSpec", "Unbounded"),
        ("DiscreteTensorSpec", "Categorical"),
        ("OneHotDiscreteTensorSpec", "OneHot"),
        ("MultiDiscreteTensorSpec", "MultiCategorical"),
        ("MultiOneHotDiscreteTensorSpec", "MultiOneHot"),
        ("BinaryDiscreteTensorSpec", "Binary"),
        ("NdBoundedTensorSpec", "Bounded"),
        ("NdUnboundedContinuousTensorSpec", "Unbounded"),
    ]:
        if not hasattr(_tspecs, _old) and hasattr(_tspecs, _new):
            setattr(_tspecs, _old, getattr(_tspecs, _new))