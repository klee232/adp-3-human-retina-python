# Created by Kuan-Min Lee
# Created date: Nov, 4th 2025 
# All rights reserved to Leelab.ai
# passed

# Brief User Introducttion:
# This script is constructed to conduct checking the validation of neural network



import torch # for model construction



# Brief User Introduction:
# this function is used to check if any of the layer output NaN or Inf 
def register_nan_inf_hooks(model):
    """
    Attach forward hooks to every module in the model.
    Detects if that layer outputs NaN or Inf.
    """
    handles = []

    for name, module in model.named_modules():
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                # Support tuple/list outputs
                if isinstance(output, (tuple, list)):
                    output = output[0]

                if output is None:
                    return

                if torch.isnan(output).any():
                    layer_finite_status[layer_name] = "NaN"
                elif torch.isinf(output).any():
                    layer_finite_status[layer_name] = "Inf"
                else:
                    layer_finite_status[layer_name] = "OK"

            return hook_fn

        handles.append(module.register_forward_hook(make_hook(name)))
    return handles



# Brief User Introduction:
# this function is used to check if any of the layer has NaN or Inf parameters
def must_be_finite(name, tensor, step=None):
    """
    Detect BOTH Inf and NaN in a tensor.
    If found, print details + which layer produced it.
    """
    if tensor is None:
        return

    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if not has_nan and not has_inf:
        return  # Safe

    # Build error message
    msg = f"\n❌ [Non-finite detected in '{name}']"
    if step is not None:
        msg += f" at step {step}"
    msg += f"\n   - Contains NaN: {bool(has_nan)}"
    msg += f"\n   - Contains Inf: {bool(has_inf)}"
    msg += f"\n   - 🔍 Offending layers:"

    for layer_name, status in layer_finite_status.items():
        if status in ["NaN", "Inf"]:
            msg += f"\n     ➤ {layer_name} → {status}"

    raise RuntimeError(msg)