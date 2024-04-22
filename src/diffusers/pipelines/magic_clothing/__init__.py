from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import PIL
from PIL import Image

from ...utils import (
    DIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_torch_available,
    is_transformers_available,
)

_dummy_objects = {}
_import_structure = {}

try:
    if not (is_transformers_available() and is_torch_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_torch_and_transformers_objects  # noqa F403
    _dummy_objects.update(get_objects_from_module(dummy_torch_and_transformers_objects))

else :
    _import_structure["oms_animate_diffusion_pipeline"] = ["OmsAnimateDiffusionPipeline"]
    _import_structure["oms_diffusion_controlnet_pipeline"] = ["OmsDiffusionControlNetPipeline"]
    _import_structure["oms_diffusion_inpaint_pipeline"] = ["OmsDiffusionInpaintPipeline"]
    _import_structure["oms_diffusion_pipeline"] = ["OmsDiffusionPipeline"]

if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_transformers_available() and is_torch_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_torch_and_transformers_objects import *
    else:
        from .OmsAnimateDiffusionPipeline import OmsAnimateDiffusionPipeline
        from .OmsDiffusionControlNetPipeline import OmsDiffusionControlNetPipeline
        from .OmsDiffusionInpaintPipeline import OmsDiffusionInpaintPipeline
        from .OmsDiffusionPipeline import OmsDiffusionPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)