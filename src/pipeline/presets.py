"""Registry of reusable pipeline presets and model building blocks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List


@dataclass
class PipelinePreset:
    name: str
    builder: Callable[[], Any]
    description: str = ""
    tags: Iterable[str] = field(default_factory=tuple)


_REGISTRY: Dict[str, PipelinePreset] = {}


def register_preset(preset: PipelinePreset, *, override: bool = False) -> None:
    key = preset.name
    if key in _REGISTRY and not override:
        raise ValueError(f"Preset '{key}' already registered")
    _REGISTRY[key] = preset


def get_preset(name: str) -> PipelinePreset:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown pipeline preset '{name}'")
    return _REGISTRY[name]


def build_pipeline(name: str) -> Any:
    preset = get_preset(name)
    return preset.builder()


def available_presets() -> List[str]:
    return sorted(_REGISTRY.keys())


def preset_details() -> Dict[str, Dict[str, Any]]:
    return {
        name: {
            "description": preset.description,
            "tags": list(preset.tags),
        }
        for name, preset in sorted(_REGISTRY.items())
    }


def _default_builder() -> Any:
    try:
        from .pipeline import build_default_pipeline
    except ImportError:
        from pipeline import build_default_pipeline

    return build_default_pipeline()


def _transformer_builder() -> Any:
    from .models.transformer_absa import TransformerBaselinePipeline

    return TransformerBaselinePipeline()


def _register_builtin_presets() -> None:
    register_preset(
        PipelinePreset(
            name="full_stack",
            builder=_default_builder,
            description="Default graph-based ETSA pipeline",
            tags=("graph", "default"),
        ),
        override=True,
    )

    if importlib.util.find_spec("transformers") is not None:
        register_preset(
            PipelinePreset(
                name="transformer_baseline",
                builder=_transformer_builder,
                description="Yangheng DeBERTa-v3 end2end ABSA baseline",
                tags=("transformer", "baseline"),
            ),
            override=True,
        )


_register_builtin_presets()
