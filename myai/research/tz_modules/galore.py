# torchzero/modules/projections/galore.py

from typing import Literal, Callable, Any, List, Tuple, Dict
from operator import itemgetter
import warnings

import torch

from torchzero.core import Chainable, Module, Vars
from torchzero.modules.projections import Projection
from torchzero.utils import TensorList, set_storage_

# Default filter: Apply GaLore only to layers with 2 or more dimensions
# and more than one element along each of the first two dimensions.
def default_galore_filter(param: torch.Tensor) -> bool:
    return param.ndim >= 2 and param.shape[0] > 1 and param.shape[1] > 1

class GaLoreProjection(Projection):
    """
    Implements GaLore (Gradient Low-Rank Projection) optimization strategy.

    Projects gradients of high-dimensional weight matrices onto a low-rank subspace
    defined by matrices P and Q. Optimization proceeds in this low-rank space,
    potentially saving memory and computation, especially for large layers.
    The full update is reconstructed after the inner optimizer step.

    Reference:
        * GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
          (https://arxiv.org/abs/2403.03507)

    Args:
        modules (Chainable):
            Inner optimization module(s) to apply to the projected low-rank gradients (dP, dQ).
        rank (int):
            The rank 'r' for the low-rank projection.
        update_freq (int, optional):
            Frequency (in steps) for updating the projection matrices P and Q using SVD.
            Defaults to 1 (update every step).
        scale (float, optional):
            Scaling factor applied to the projected gradients before passing them to the
            inner optimizer. Defaults to 1.0.
        layer_filter (Callable[[torch.Tensor], bool] | None, optional):
            A function that takes a parameter tensor and returns True if GaLore should be
            applied to it. If None, uses a default filter that applies GaLore to
            parameters with ndim >= 2 and size > 1 in the first two dimensions.
            Defaults to None.
        svd_dtype (torch.dtype, optional):
            dtype to use for SVD computation for stability. Defaults to torch.float32.
        project_update (bool): Must be True for GaLore.
        project_params (bool): Whether to project parameters (needed for closures).
        project_grad (bool): Whether to project gradients separately.
    """

    def __init__(
        self,
        modules: Chainable,
        rank: int,
        update_freq: int = 1,
        scale: float = 1.0,
        layer_filter: Callable[[torch.Tensor], bool] | None = None,
        svd_dtype: torch.dtype = torch.float32,
        # GaLore primarily projects the update/gradient
        project_update: bool = True,
        project_params: bool = False,
        project_grad: bool = False,
    ):
        if not project_update:
            warnings.warn("GaLoreProjection typically requires project_update=True.", UserWarning)

        defaults = dict(
            rank=rank,
            update_freq=update_freq,
            scale=scale,
            layer_filter=layer_filter or default_galore_filter,
            svd_dtype=svd_dtype,
        )
        super().__init__(
            modules=modules,
            project_update=project_update,
            project_params=project_params,
            project_grad=project_grad,
            defaults=defaults,
        )
        # Clear temporary state at the start of each step
        self.global_state['galore_map'] = {}


    def _should_apply_galore(self, param: torch.Tensor, settings: Dict[str, Any]) -> bool:
        """Check if GaLore should be applied based on filter and dimensions."""
        layer_filter = settings['layer_filter']
        rank = settings['rank']
        if not layer_filter(param):
            return False
        # Check if rank is smaller than dimensions
        if rank >= min(param.shape[0], param.shape[1]):
             warnings.warn(f"GaLore rank {rank} is >= min(shape) {min(param.shape[:2])} for param {param.shape}. Skipping GaLore for this parameter.", UserWarning)
             return False
        return True

    @torch.no_grad
    def project(self, tensors: List[torch.Tensor], vars: Vars) -> List[torch.Tensor]:
        """Projects gradients onto low-rank subspaces P and Q."""
        projected_gradients_flat: List[torch.Tensor] = []
        galore_map: Dict[int, Dict[str, Any]] = {} # Store mapping info

        flat_idx_counter = 0

        # Ensure state is initialized for all parameters on the first pass
        # And also retrieve settings efficiently
        params_settings = {p: self.settings[p] for p in vars.params}

        for i, (param, grad) in enumerate(zip(vars.params, tensors)):
            settings = params_settings[param]
            state = self.state[param]

            # Initialize state on first encounter
            if 'step' not in state:
                state['step'] = 0
                state['galore_applied'] = False # Track if GaLore is ever applied

            apply_galore = self._should_apply_galore(param, settings)

            if apply_galore:
                state['galore_applied'] = True # Mark that GaLore logic is used for this param
                rank = settings['rank']
                update_freq = settings['update_freq']
                svd_dtype = settings['svd_dtype']
                scale = settings['scale']

                # --- Update P and Q via SVD if needed ---
                if state['step'] % update_freq == 0:
                    original_dtype = grad.dtype
                    matrix = grad.to(svd_dtype)
                    needs_transpose = False
                    if matrix.shape[0] < matrix.shape[1]:
                        matrix = matrix.T
                        needs_transpose = True

                    try:
                        # Use full_matrices=False for efficiency
                        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
                        # P is U[:, :rank]
                        # Q is Vh[:rank, :].T  (which is V[:,:rank])
                        P = U[:, :rank].to(original_dtype)
                        Q = Vh[:rank, :].T.to(original_dtype)

                        if needs_transpose:
                            # If we transposed, swap P and Q
                            P, Q = Q, P

                        # Store P, Q, and transpose status
                        state['P'] = P.contiguous()
                        state['Q'] = Q.contiguous()
                        state['svd_needs_transpose'] = needs_transpose # Store if grad was transposed for SVD

                    except torch.linalg.LinAlgError:
                         warnings.warn(f"SVD failed for parameter {i} with shape {grad.shape}. Skipping GaLore update for this step.", UserWarning)
                         # Fallback: Don't project, treat as non-GaLore for this step?
                         # Or reuse old P, Q? Let's reuse old P, Q if available.
                         if 'P' not in state or 'Q' not in state:
                             # If no P, Q exist yet, we cannot proceed with GaLore
                             apply_galore = False


                # --- Project the gradient if GaLore is applicable (and SVD succeeded/P,Q exist) ---
                if apply_galore and 'P' in state and 'Q' in state:
                    P = state['P']
                    Q = state['Q']

                    # Project: gP = P^T @ g, gQ = g @ Q
                    gP = P.T @ grad
                    gQ = grad @ Q

                    # Apply scaling
                    gP.mul_(scale)
                    gQ.mul_(scale)

                    projected_gradients_flat.extend([gP, gQ])
                    galore_map[i] = {
                        'is_galore': True,
                        'indices': (flat_idx_counter, flat_idx_counter + 1),
                        'original_shape': param.shape,
                        # No need to store svd_needs_transpose here, P/Q are already correct
                    }
                    flat_idx_counter += 2
                else:
                    # GaLore not applied (filter, rank issue, or SVD failed without prior P/Q)
                    projected_gradients_flat.append(grad) # Pass through original gradient
                    galore_map[i] = {
                        'is_galore': False,
                        'indices': flat_idx_counter,
                        'original_shape': param.shape,
                    }
                    flat_idx_counter += 1

            else: # GaLore filter returned False initially
                projected_gradients_flat.append(grad) # Pass through original gradient
                galore_map[i] = {
                    'is_galore': False,
                    'indices': flat_idx_counter,
                    'original_shape': param.shape,
                }
                flat_idx_counter += 1

            # Increment step counter only if GaLore was ever applied to this param
            if state['galore_applied']:
                state['step'] += 1


        self.global_state['galore_map'] = galore_map # Store map for unproject
        return projected_gradients_flat

    @torch.no_grad
    def unproject(self, tensors: List[torch.Tensor], vars: Vars) -> List[torch.Tensor]:
        """Reconstructs the full update from low-rank updates dP and dQ."""
        reconstructed_updates: List[torch.Tensor] = [torch.empty(0)] * len(vars.params)
        galore_map = self.global_state.get('galore_map', {})
        if not galore_map:
             # This might happen if project was never called or state was cleared
             warnings.warn("GaLore map not found during unproject. Returning input tensors.", UserWarning)
             if len(tensors) == len(vars.params):
                 # Best guess: assume no projection happened
                  return tensors
             else:
                  # Cannot reliably reconstruct
                  raise RuntimeError("Cannot unproject GaLore updates: mapping information is missing.")


        for i, param in enumerate(vars.params):
            map_info = galore_map.get(i)
            if map_info is None:
                 raise RuntimeError(f"Missing map info for parameter index {i} during GaLore unprojection.")

            state = self.state[param]

            if map_info['is_galore'] and state.get('galore_applied', False) and 'P' in state and 'Q' in state:
                # Retrieve dP and dQ from the flat list
                idx_p, idx_q = map_info['indices']
                dP = tensors[idx_p]
                dQ = tensors[idx_q]

                # Retrieve P and Q used for projection
                P = state['P']
                Q = state['Q']

                # Reconstruct: update = P @ dQ^T + dP @ Q^T
                # Note: P and Q shapes are already adjusted for potential SVD transpose
                update = P @ dQ.T + dP @ Q.T
                reconstructed_updates[i] = update

            elif map_info['is_galore']:
                 # GaLore was intended but skipped (e.g., SVD error on first step)
                 # We expect the original gradient/update to be in the flat list
                 idx = map_info['indices']
                 reconstructed_updates[i] = tensors[idx]
                 warnings.warn(f"Unprojecting GaLore parameter {i} that was skipped during projection. Using passthrough value.", UserWarning)

            else: # Not a GaLore parameter
                idx = map_info['indices']
                reconstructed_updates[i] = tensors[idx]

        # Clear the map after use
        self.global_state.pop('galore_map', None)
        return reconstructed_updates