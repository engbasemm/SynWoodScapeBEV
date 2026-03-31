import torch
import torch.nn as nn
# import torch.nn.functional as F # No longer directly needed here
# import math # No longer directly needed here

class _TaskSelfAttention(nn.Module):
    """
    Internal self-attention module to weigh task contributions.
    It expects contributions for ALL tasks defined in the main loss module.
    Adds Layer Normalization for stability.
    """
    def __init__(self, num_tasks, embed_dim, num_heads=1):
        super().__init__()
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        self.num_heads = num_heads # Store num_heads as well

        if self.num_tasks == 0:
            self.input_projection = nn.Identity()
            self.input_norm = nn.Identity() # Add Identity for norm
            self.attention = nn.Identity()
            self.attention_output_norm = nn.Identity() # Add Identity for norm
            self.output_transform = nn.Identity()
            print("Warning: _TaskSelfAttention initialized with num_tasks=0")
            return

        # Handle case where embed_dim is 1 (no projection needed if input is already 1D)
        self.input_projection = nn.Linear(1, embed_dim) if embed_dim > 1 else nn.Identity()

        # Add LayerNorm after input projection
        # Apply LayerNorm only if there's an embedding dimension to normalize over (> 1)
        # If embed_dim is 1 and no projection, input_norm is Identity
        self.input_norm = nn.LayerNorm(embed_dim) if embed_dim > 1 or not isinstance(self.input_projection, nn.Identity) else nn.Identity()


        # Ensure embed_dim is compatible with num_heads if num_heads > 1
        if num_heads > 1 and embed_dim % num_heads != 0:
             print(f"Warning: embed_dim ({embed_dim}) is not divisible by num_heads ({num_heads}). Adjusting embed_dim to be divisible.")
             # Adjust embed_dim to be divisible by num_heads
             self.embed_dim = (embed_dim // num_heads) * num_heads
             if self.embed_dim == 0: # Ensure it doesn't become 0 if original was small
                 self.embed_dim = num_heads
             print(f"Adjusted embed_dim to {self.embed_dim}")
             # Re-initialize projection and norm with adjusted embed_dim if they were not Identity
             if not isinstance(self.input_projection, nn.Identity):
                 self.input_projection = nn.Linear(1, self.embed_dim)
                 self.input_norm = nn.LayerNorm(self.embed_dim)
             self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
             self.attention_output_norm = nn.LayerNorm(self.embed_dim)
             self.output_transform = nn.Linear(self.embed_dim, 1)
        else:
            self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
            # Add LayerNorm after attention output
            self.attention_output_norm = nn.LayerNorm(self.embed_dim)
            self.output_transform = nn.Linear(self.embed_dim, 1)


        # --- Initialization ---
        with torch.no_grad():
            # Initialize bias to encourage initial scales closer to 1.0 (sigmoid(3.0) ~ 0.95)
            self.output_transform.bias.fill_(3.0)

            # Initialize weights to be small to let the bias dominate initially
            self.output_transform.weight.data.uniform_(-0.01, 0.01)

            # Standard Kaiming init for input projection weights if it's a Linear layer
            if isinstance(self.input_projection, nn.Linear):
                nn.init.kaiming_uniform_(self.input_projection.weight, a=math.sqrt(5))
                if self.input_projection.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.input_projection.weight)
                    if fan_in > 0:
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(self.input_projection.bias, -bound, bound)
                    else:
                        nn.init.constant_(self.input_projection.bias, 0)

            # Initialize LayerNorms
            if not isinstance(self.input_norm, nn.Identity):
                self.input_norm.weight.fill_(1.0)
                self.input_norm.bias.fill_(0.0)
            if not isinstance(self.attention_output_norm, nn.Identity):
                self.attention_output_norm.weight.fill_(1.0)
                self.attention_output_norm.bias.fill_(0.0)


    def forward(self, task_contributions):
        """
        Parameters:
        - task_contributions (Tensor): A tensor of shape (num_tasks,) representing
                                       the unscaled total contributions of each task
                                       (uncertainty_weighted_loss + sigma_regularization_term).
        Returns:
        - attention_scales (Tensor): A tensor of shape (num_tasks,) representing
                                     the attention-based scales for each task.
                                     These are sigmoid outputs, between 0 and 1.
        """
        if self.num_tasks == 0:
            return torch.ones_like(task_contributions)

        # Ensure task_contributions is on the same device as the module's parameters
        expected_device = next(self.parameters()).device
        if task_contributions.device != expected_device:
            task_contributions = task_contributions.to(expected_device)

        if self.embed_dim == 1 and isinstance(self.input_projection, nn.Identity):
            x = task_contributions.unsqueeze(0).unsqueeze(-1)
        else:
            x = self.input_projection(task_contributions.unsqueeze(-1)).unsqueeze(0) # (1, num_tasks, embed_dim)

        # Apply input normalization
        x = self.input_norm(x)

        attn_output, _ = self.attention(x, x, x)

        # Apply normalization to attention output
        attn_output = self.attention_output_norm(attn_output)

        # Transform the attention output back to a single modulation value per task
        # Squeeze batch dim (0), apply linear transform, squeeze last dim (to get shape (num_tasks,))
        modulators = self.output_transform(attn_output.squeeze(0)).squeeze(-1) # (num_tasks, 1) -> (num_tasks,)

        # Apply sigmoid to get scales between 0 and 1
        attention_scales = torch.sigmoid(modulators)

        return attention_scales

import torch
import torch.nn as nn
import torch.nn.functional as F  # Not strictly used here but good for nn modules
import math  # For sqrt if needed, though torch.sqrt is preferred


class UncertaintyLoss(nn.Module):
    def __init__(self, tasks=None, enable_compression=False, uncertainty_regularization_strength=0.1,
                 # New optional parameters for self-attention
                 enable_self_attention=False,
                 attention_embed_dim=8,  # Sensible default
                 attention_heads=1):  # Sensible default
        """
        Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.
        Compatible with the original interface.

        Parameters:
        - tasks (str, optional): A string of tasks separated by '_'. Example: "semantic_object_detection_compression".
        - enable_compression (bool): Whether to enable the compression task.
        - uncertainty_regularization_strength (float): Regularization for uncertainty parameters.
        - enable_self_attention (bool, optional): If True, enables a self-attention mechanism
                                                 to further modulate task losses. Defaults to True.
        - attention_embed_dim (int, optional): Embedding dimension for tasks within the
                                              self-attention mechanism. Defaults to 8.
        - attention_heads (int (>=1), optional): Number of heads for the self-attention mechanism.
                                           Defaults to 1. Must be >= 1.
        """
        super().__init__()

        # Parse the tasks from the input string (e.g., "semantic_object_detection_compression")
        # Filter out empty strings that might result from split
        self.parsed_tasks = [task for task in tasks.split('_') if task] if tasks and tasks.strip() else []
        self.num_tasks = len(self.parsed_tasks)

        # Initialize learnable uncertainty parameters (log of sigma squared for stability)
        # sigma^2 = exp(log_sigma_sq)
        # sigma = exp(0.5 * log_sigma_sq)
        if self.num_tasks > 0:
            self.log_sigma_sq = nn.Parameter(torch.zeros(self.num_tasks), requires_grad=True)
        else:
            # Register as buffer if no tasks, so state_dict is consistent
            self.register_buffer('log_sigma_sq', torch.empty(0), persistent=False)

        self.uncertainty_regularization_strength = uncertainty_regularization_strength
        self.enable_compression = enable_compression

        # Self-Attention specific initialization
        self.enable_self_attention = enable_self_attention and self.num_tasks > 0
        if self.enable_self_attention:
             if attention_heads < 1:
                 raise ValueError("attention_heads must be at least 1.")
             # _TaskSelfAttention now handles embed_dim divisibility by num_heads internally
             self.attention_module = _TaskSelfAttention(
                num_tasks=self.num_tasks,
                embed_dim=attention_embed_dim, # Pass the requested embed_dim
                num_heads=attention_heads
            )
             # Update the actual embed_dim used in case _TaskSelfAttention adjusted it
             self.attention_embed_dim = self.attention_module.embed_dim
        else:
            self.attention_module = None
            self.attention_embed_dim = 0


    def forward(self, losses):
        """
        Calculate the total weighted loss with uncertainty-based weighting
        and optional self-attention.

        Parameters:
        - losses (dict): A dictionary containing the individual task losses
                         (e.g., 'semantic_loss', 'object_detection_loss', 'compression_loss').
                         This dictionary will be modified in-place to add sigma and weightage logs.

        Returns:
        - total_loss (Tensor): The total weighted loss for multi-task learning.
        """
        if self.num_tasks == 0:
            # If no tasks are defined, check if there's a default "loss" key to return
            # or return 0. Returning 0 is safer if no tasks means no loss.
            device = 'cpu'
            for loss_val in losses.values():
                if isinstance(loss_val, torch.Tensor):
                    device = loss_val.device
                    break
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Collect device from an existing loss tensor for creating new tensors
        device = self.log_sigma_sq.device # Default to sigma device if num_tasks > 0

        # Tensor to store per-task contributions for attention module and summation
        # (uncertainty_weighted_loss + sigma_regularization_term)
        # This needs to be of size self.num_tasks, initialized to 0
        per_task_contributions = torch.zeros(self.num_tasks, device=device)

        for idx, current_task in enumerate(self.parsed_tasks):
            # Skip compression task if not enabled
            if current_task == "compression" and not self.enable_compression:
                # Log a default sigma if needed, though it won't contribute to loss
                if self.num_tasks > 0 and idx < self.log_sigma_sq.size(0):
                    # Use a dummy sigma for logging purposes if the parameter exists
                    # Log detached value as it's not used in loss calculation for this task
                    sigma_sq_val = torch.exp(self.log_sigma_sq[idx].detach())
                    sigma_val = torch.sqrt(sigma_sq_val)
                    losses[f"sigma/{current_task}"] = sigma_val
                    losses[f"sigma/{current_task}_weightage"] = (0.5 / sigma_sq_val)
                continue

            task_loss_key = f"{current_task}_loss"
            task_loss = losses.get(task_loss_key, 0.0)

            # Handle the compression task's specific loss structure as in original
            # Handle the compression task: Ensure correct loss structure
            if current_task == "compression":
                if isinstance(losses.get("compression_loss"), dict):
                    losses["compression_loss"] = losses["compression_loss"].get("compression_loss", 0.0)
            if current_task == "compression":
                if isinstance(task_loss, dict):  # losses.get("compression_loss") can be a dict
                    task_loss = task_loss.get("compression_loss", 0.0)

            # Ensure task_loss is a tensor and on the correct device
            if not isinstance(task_loss, torch.Tensor):
                 # If losses dict is empty or contains non-tensors, use self.log_sigma_sq.device as default
                 task_loss = torch.tensor(float(task_loss), device=device, requires_grad=True if isinstance(task_loss, float) else task_loss.requires_grad)
            else:
                 # Move to correct device if necessary
                 task_loss = task_loss.to(device)


            # Check if task loss is effectively zero and does not require gradient
            # (e.g., padding or disabled task loss that somehow got included)
            # We check requires_grad to differentiate from a task loss that is legitimately zero
            # but part of the computational graph and needs gradients (though this is rare).
            if task_loss.item() == 0.0 and not task_loss.requires_grad:
                 # Log sigma even if loss is zero, but it won't contribute to loss sum here
                 if self.num_tasks > 0 and idx < self.log_sigma_sq.size(0):
                    sigma_sq_val = torch.exp(self.log_sigma_sq[idx].detach())
                    sigma_val = torch.sqrt(sigma_sq_val)
                    losses[f"sigma/{current_task}"] = sigma_val
                    losses[f"sigma/{current_task}_weightage"] = (0.5 / sigma_sq_val)
                 # The contribution for attention will remain 0 for this task as per initialization
                 continue

            # Calculate sigma squared and sigma for the current task
            if self.num_tasks > 0 and idx < self.log_sigma_sq.size(0):
                log_sigma_sq_param = self.log_sigma_sq[idx]
            else:
                 # This case should ideally not happen if num_tasks > 0 and idx is valid
                 raise IndexError(f"Task index {idx} out of bounds for log_sigma_sq with size {self.log_sigma_sq.size(0)}")


            sigma_sq = torch.exp(log_sigma_sq_param)
            sigma = torch.sqrt(sigma_sq)  # sigma = exp(0.5 * log_sigma_sq)

            # Uncertainty-weighted loss component: task_loss / (2 * sigma^2)
            uncertainty_weighted_loss = task_loss * (0.5 / sigma_sq)

            # Regularization term for this task's sigma: log(1 + sigma^2)
            # This matches the original formulation: torch.log(1 + self.sigma[idx].pow(2))
            sigma_regularization_term = torch.log(1 + sigma_sq)

            # The total contribution of this task to the unscaled sum
            current_task_total_contribution = uncertainty_weighted_loss + sigma_regularization_term
            per_task_contributions[idx] = current_task_total_contribution

            # For backward compatibility of logging
            losses[f"sigma/{current_task}"] = sigma.detach() # Log detached value
            losses[f"sigma/{current_task}_weightage"] = (0.5 / sigma_sq).detach() # Log detached value

        # --- Apply Self-Attention (if enabled) ---
        if self.enable_self_attention and self.attention_module is not None:
            # The attention module learns to scale the `per_task_contributions`.
            # We pass the gradients of `per_task_contributions` through the attention module
            # so that the attention weights can be learned based on these contributions.
            attention_scales = self.attention_module(per_task_contributions)

            # Modulate the contributions using the attention scales
            final_contributions = per_task_contributions * attention_scales
            total_weighted_loss = final_contributions.sum()

            # Log attention scales (optional, but good practice)
            for idx, current_task in enumerate(self.parsed_tasks):
                 # Log scales for all tasks defined, even if their initial contribution was zero
                 if idx < attention_scales.size(0): # Check index bounds for safety
                     losses[f"attention_scale/{current_task}"] = attention_scales[idx].detach()
        else:
            # If self-attention is not enabled, the total loss is simply the sum of
            # the per-task contributions (uncertainty_weighted_loss + sigma_regularization_term).
            total_weighted_loss = per_task_contributions.sum()

        # Add overall L2 regularization for all sigmas (sigma^2)
        # This is self.uncertainty_regularization_strength * self.sigma.pow(2).sum()
        if self.uncertainty_regularization_strength > 0 and self.num_tasks > 0:
            # sigma.pow(2) is exp(log_sigma_sq)
            sigma_sq_sum_regularization = self.uncertainty_regularization_strength * torch.exp(self.log_sigma_sq).sum()
            total_weighted_loss += sigma_sq_sum_regularization
            # Log the regularization term
            losses["sigma_L2_reg"] = sigma_sq_sum_regularization.detach()

        # Add the total loss to the losses dictionary for logging
        losses["total_loss"] = total_weighted_loss.detach() # Log detached total loss

        return total_weighted_loss