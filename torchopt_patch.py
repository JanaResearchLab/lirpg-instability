# Modified from torchopt/optim/meta/base.py (torchopt 0.7.3), Copyright 2022-2024 MetaOPT Team.
# Licensed under the Apache License, Version 2.0; see licenses/torchopt-LICENSE.txt.
# Changes: step() takes max_grad_norm, clips the gradients to it and returns their norm.
"""Patch torchopt 0.7.3 MetaOptimizer.step to clip gradients to max_grad_norm and return the norm.

The clipping scales gradient values in place outside autograd, so the meta-gradient ignores the clip factor.
"""

import warnings

import torch
import torchopt
from torchopt import pytree
from torchopt.base import UninitializedState
from torchopt.optim.meta.base import MetaOptimizer
from torchopt.update import apply_updates

if torchopt.__version__ != "0.7.3":
    warnings.warn(
        f"torchopt_patch was written for torchopt 0.7.3, found {torchopt.__version__}"
    )


def step(self, loss: torch.Tensor, max_grad_norm: float | None = None):
    total_norm = 0.0
    for i, (param_container, state) in enumerate(
        zip(self.param_containers_groups, self.state_groups),
    ):
        flat_params, container_treespec = pytree.tree_flatten_as_tuple(param_container)
        if isinstance(state, UninitializedState):
            state = self.impl.init(flat_params)
        grads = torch.autograd.grad(
            loss,
            flat_params,
            create_graph=True,
            allow_unused=True,
        )
        if max_grad_norm:
            device = grads[0].device
            norm_type = 2.0
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(grad.detach(), norm_type).to(device) for grad in grads]
                ),
                norm_type,
            )
            clip_coef = max_grad_norm / (total_norm + 1e-6)
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            for grad in grads:
                grad.detach().mul_(clip_coef_clamped.to(grad.device))

        updates, new_state = self.impl.update(
            grads,
            state,
            params=flat_params,
            inplace=False,
        )
        self.state_groups[i] = new_state
        flat_new_params = apply_updates(flat_params, updates, inplace=False)
        new_params = pytree.tree_unflatten(container_treespec, flat_new_params)
        for container, new_param in zip(param_container, new_params):
            container.update(new_param)
    return total_norm


MetaOptimizer.step = step
