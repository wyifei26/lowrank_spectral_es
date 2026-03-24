from dataclasses import dataclass

import torch.nn as nn


MODULE_ALIASES = {
    "q_proj": ("self_attn.q_proj",),
    "k_proj": ("self_attn.k_proj",),
    "v_proj": ("self_attn.v_proj",),
    "o_proj": ("self_attn.o_proj", "linear_attn.out_proj"),
    "gate_proj": ("mlp.gate_proj",),
    "up_proj": ("mlp.up_proj",),
    "down_proj": ("mlp.down_proj",),
}


@dataclass
class LayerSelection:
    full_name: str
    module_key: str
    block_index: int
    parent_module: nn.Module
    child_name: str
    module: nn.Linear


def _resolve_attr_chain(root: nn.Module, attr_path: str) -> tuple[nn.Module, str, nn.Module]:
    parts = attr_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    child_name = parts[-1]
    return parent, child_name, getattr(parent, child_name)


def _resolve_first_existing_attr_chain(
    root: nn.Module,
    attr_paths: tuple[str, ...],
    *,
    module_key: str,
    block_index: int,
) -> tuple[str, nn.Module, str, nn.Module]:
    last_error: AttributeError | None = None
    for attr_path in attr_paths:
        try:
            parent, child_name, module = _resolve_attr_chain(root, attr_path)
            return attr_path, parent, child_name, module
        except AttributeError as exc:
            last_error = exc
    path_summary = ", ".join(attr_paths)
    raise AttributeError(
        f"could not resolve {module_key} at block {block_index}; tried {path_summary}"
    ) from last_error


def select_target_layers(
    model: nn.Module,
    *,
    target_blocks: list[int],
    target_modules: list[str],
) -> list[LayerSelection]:
    selections: list[LayerSelection] = []
    for block_index in target_blocks:
        block = model.model.layers[block_index]
        for module_key in target_modules:
            attr_path, parent, child_name, module = _resolve_first_existing_attr_chain(
                block,
                MODULE_ALIASES[module_key],
                module_key=module_key,
                block_index=block_index,
            )
            if not isinstance(module, nn.Linear):
                raise TypeError(f"{module_key} at block {block_index} is not nn.Linear: {type(module)}")
            full_name = f"model.layers.{block_index}.{attr_path}"
            selections.append(
                LayerSelection(
                    full_name=full_name,
                    module_key=module_key,
                    block_index=block_index,
                    parent_module=parent,
                    child_name=child_name,
                    module=module,
                )
            )
    return selections
