import torch
import torch.nn as nn

class UncertaintyLoss(nn.Module):
    def __init__(self, tasks=None, enable_compression=False, uncertainty_regularization_strength=0.001):
        """
        Multi-task learning using uncertainty to weigh losses for scene geometry and semantics.

        Parameters:
        - tasks (str): A string of tasks separated by '_'. For example, "semantic_object_detection_compression".
        - enable_compression (bool): Whether to enable the compression task. If False, the compression loss will not be used.
        - uncertainty_regularization_strength (float): A regularization term for uncertainty parameters to prevent them from growing too large.
        """
        super().__init__()

        # Parse the tasks from the input string (e.g., "semantic_object_detection_compression")
        self.tasks = tasks.split('_') if tasks else []

        # Initialize learnable uncertainty parameters (sigma) for each task
        self.sigma = nn.Parameter(torch.ones(len(self.tasks)), requires_grad=True)

        # Optional regularization strength for uncertainty parameters to avoid large values of sigma
        self.uncertainty_regularization_strength = uncertainty_regularization_strength

        # Flag to enable or disable compression task
        self.enable_compression = enable_compression

    def forward(self, losses):
        """
        Calculate the total weighted loss with uncertainty-based weighting.

        Parameters:
        - losses (dict): A dictionary containing the individual task losses (e.g., 'semantic_loss', 'object_detection_loss', 'compression_loss').

        Returns:
        - loss (Tensor): The total weighted loss for multi-task learning.
        """
        total_loss = 0

        # Loop over each task and compute the weighted loss based on its uncertainty
        for idx, current_task in enumerate(self.tasks):
            # Skip compression task if not enabled
            if current_task == "compression" and not self.enable_compression:
                continue

            # Handle the compression task: Ensure correct loss structure
            if current_task == "compression":
                if isinstance(losses.get("compression_loss"), dict):
                    losses["compression_loss"] = losses["compression_loss"].get("compression_loss", 0.0)

            # Compute the weighted loss for the current task
            task_loss = losses.get(f"{current_task}_loss", 0.0)
            if task_loss == 0.0:
                continue  # Skip if task loss is zero or not present

            uncertainty_weight = 1 / (2 * self.sigma[idx].pow(2))
            total_loss += uncertainty_weight * task_loss + torch.log(1 + self.sigma[idx].pow(2))

            # Log uncertainty and weightage for monitoring purposes
            losses[f"sigma/{current_task}"] = self.sigma[idx]
            losses[f"sigma/{current_task}_weightage"] = uncertainty_weight

        # Add regularization term for uncertainty parameters to prevent large sigma values
        total_loss += self.uncertainty_regularization_strength * self.sigma.pow(2).sum()

        return total_loss
