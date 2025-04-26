from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from typing import Optional, Tuple, Dict, List


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)



class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class JEPA(nn.Module):
    """
    Improved Joint Embedding Predictive Architecture (JEPA) for the two-room environment.
    
    This model:
    1. Encodes observations (states) into representations
    2. Predicts future representations based on current representation and actions
    3. Handles uncertainty through latent variables
    4. Uses VICReg to prevent collapse and redundancy in representations
    """
    def __init__(
        self, 
        input_channels: int = 2,         # 2 channels (agent and walls)
        repr_dim: int = 256,             # Dimension of representation
        latent_dim: int = 16,            # Dimension of latent variable
        hidden_dim: int = 128,           # Hidden dimension in networks
        use_latent: bool = True,         # Whether to use latent variables
        use_vicreg: bool = True,         # Whether to use VICReg regularization
        device: str = "cuda"
    ):
        super().__init__()
        self.repr_dim = repr_dim
        self.latent_dim = latent_dim
        self.use_latent = use_latent
        self.use_vicreg = use_vicreg
        self.device = device
        
        # Encoder: takes states (images) and encodes them into representations
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 65x65
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 33x33
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 17x17
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 9x9
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 5x5
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 5 * 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, repr_dim)
        )
        
        # Expander: projects representation to higher dimension for VICReg
        # This amplifies small differences in representations to help prevent collapse
        if use_vicreg:
            self.expander = nn.Sequential(
                nn.Linear(repr_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim * 2)
            )
        
        # Predictor: predicts next representation based on current repr and action
        action_input_dim = 2  # (delta_x, delta_y)
        
        # If using latent variables, we need to include them in the predictor input
        predictor_input_dim = repr_dim + action_input_dim
        if use_latent:
            predictor_input_dim += latent_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, repr_dim)
        )
        
        if use_latent:
            # Latent predictor: predicts latent variable distribution parameters
            # This implements amortized inference for the latent variable
            self.latent_predictor = nn.Sequential(
                nn.Linear(repr_dim + action_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log_std
            )
    
    def encode(self, states: torch.Tensor) -> torch.Tensor:
        """Encode states to representations."""
        # Handle different input shapes
        if states.dim() == 5:  # [B, T, C, H, W]
            batch_size, seq_len = states.shape[0], states.shape[1]
            states_flat = states.reshape(-1, *states.shape[2:])
            encodings_flat = self.encoder(states_flat)
            encodings = encodings_flat.reshape(batch_size, seq_len, self.repr_dim)
            return encodings
        elif states.dim() == 4:  # [B, C, H, W]
            return self.encoder(states)
        else:
            raise ValueError(f"Unexpected states shape: {states.shape}")
    
    def sample_latent(self, 
                     states: torch.Tensor, 
                     actions: torch.Tensor, 
                     deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Sample latent variables based on the state and action."""
        if not self.use_latent:
            # Return zero latent if not using latents
            batch_size, seq_len = actions.shape[0], actions.shape[1]
            return torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device), {}
        
        batch_size, seq_len = actions.shape[0], actions.shape[1]
        
        # Get current state representation (handle different input formats)
        if states.dim() == 5:  # [B, T, C, H, W]
            current_repr = self.encode(states[:, 0:1])[:, 0]  # [B, repr_dim]
        elif states.dim() == 4:  # [B, C, H, W]
            current_repr = self.encoder(states)  # [B, repr_dim]
        else:
            raise ValueError(f"Unexpected states shape: {states.shape}")
        
        # For each timestep, predict the latent distribution parameters
        latents = []
        log_probs = []
        means = []
        log_stds = []
        
        for t in range(seq_len):
            action = actions[:, t]  # [B, 2]
            
            # Predict latent distribution parameters
            latent_params = self.latent_predictor(torch.cat([current_repr, action], dim=1))
            
            # Split into mean and log_std
            mean, log_std = torch.chunk(latent_params, 2, dim=1)
            means.append(mean)
            log_stds.append(log_std)
            
            # Clamp log_std for numerical stability
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            
            # Sample latent
            if deterministic:
                z = mean
                log_prob = None
            else:
                eps = torch.randn_like(mean)
                z = mean + eps * std
                
                # Calculate log probability of the sample
                log_prob = -0.5 * ((z - mean) / std).pow(2) - log_std - 0.5 * np.log(2 * np.pi)
                log_prob = log_prob.sum(dim=1)
                log_probs.append(log_prob)
            
            latents.append(z)
            
            # Update the representation for the next timestep using the predictor
            pred_input = torch.cat([current_repr, action, z], dim=1)
            current_repr = self.predictor(pred_input)
        
        latents = torch.stack(latents, dim=1)  # [B, T, latent_dim]
        
        info = {}
        if not deterministic and log_probs:
            info['log_probs'] = torch.stack(log_probs, dim=1)  # [B, T]
        
        # Store means and log_stds for potential use in regularization
        info['means'] = torch.stack(means, dim=1) if means else None  # [B, T, latent_dim]
        info['log_stds'] = torch.stack(log_stds, dim=1) if log_stds else None  # [B, T, latent_dim]
        
        return latents, info
    
    def forward(self, 
                states: torch.Tensor, 
                actions: torch.Tensor, 
                latents: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            states: [B, T+1, C, H, W] tensor of states or [B, 1, C, H, W] initial state
            actions: [B, T, 2] tensor of actions
            latents: Optional [B, T, latent_dim] tensor of latent variables
            deterministic: Whether to sample latents deterministically
            
        Returns:
            predictions: [B, T+1, repr_dim] tensor of predicted representations
        """
        batch_size, seq_len = actions.shape[0], actions.shape[1]
        
        # Ensure states has the right shape
        if states.dim() == 4:  # [B, C, H, W]
            states = states.unsqueeze(1)  # [B, 1, C, H, W]
        
        # Encode initial state
        initial_repr = self.encode(states[:, 0:1])[:, 0]  # [B, repr_dim]
        
        # Sample latent variables if not provided
        if latents is None and self.use_latent:
            latents, _ = self.sample_latent(states, actions, deterministic)
        
        # Initialize list with the initial representation
        predictions = [initial_repr]
        
        # Recurrently predict future representations
        current_repr = initial_repr
        
        for t in range(seq_len):
            action = actions[:, t]  # [B, 2]
            
            # Prepare predictor input
            if self.use_latent:
                z = latents[:, t]  # [B, latent_dim]
                pred_input = torch.cat([current_repr, action, z], dim=1)
            else:
                pred_input = torch.cat([current_repr, action], dim=1)
            
            # Predict next representation
            next_repr = self.predictor(pred_input)
            predictions.append(next_repr)
            
            # Update current representation for next step
            current_repr = next_repr
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [B, T+1, repr_dim]
        
        return predictions
    
    def compute_vicreg_loss(self, 
                           representations: torch.Tensor, 
                           lambda_var: float = 25.0, 
                           lambda_cov: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute VICReg regularization loss to prevent representation collapse.
        
        Args:
            representations: [B, D] tensor of representations
            lambda_var: Weight for variance loss
            lambda_cov: Weight for covariance loss
            
        Returns:
            loss: VICReg loss
            info: Dictionary with detailed loss components
        """
        if not self.use_vicreg:
            return torch.tensor(0.0).to(self.device), {"var_loss": 0.0, "cov_loss": 0.0}
        
        # Project representations to higher dimension using expander
        projections = self.expander(representations)  # [B, 2*hidden_dim]
        
        # Center the projections along the batch dimension
        projections = projections - projections.mean(dim=0)
        
        # Variance loss: ensure each dimension has sufficient variance
        std = torch.sqrt(projections.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std))
        
        # Covariance loss: ensure dimensions are decorrelated
        batch_size = projections.size(0)
        cov = (projections.T @ projections) / (batch_size - 1)
        
        # Zero out the diagonal (variances) to only penalize covariances
        mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
        cov_loss = (cov[mask]**2).mean()
        
        # Total VICReg loss
        loss = lambda_var * var_loss + lambda_cov * cov_loss
        
        return loss, {"var_loss": var_loss.item(), "cov_loss": cov_loss.item()}
    
    def compute_latent_regularization(self, latent_info: Dict) -> torch.Tensor:
        """
        Compute regularization for latent variables to encourage minimal information.
        
        Args:
            latent_info: Dictionary with latent information from sample_latent
            
        Returns:
            reg_loss: Regularization loss for latent variables
        """
        if not self.use_latent or 'log_probs' not in latent_info:
            return torch.tensor(0.0).to(self.device)
        
        # KL divergence to standard normal prior
        reg_loss = -latent_info['log_probs'].mean()
        
        return reg_loss


class JEPAWorldModel(nn.Module):
    """
    Wrapper around JEPA model that provides a forward pass interface compatible with the evaluator.
    """
    def __init__(self, jepa_model: JEPA):
        super().__init__()
        self.jepa = jepa_model
        self.repr_dim = jepa_model.repr_dim
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that takes states and actions and returns predicted representations.
        
        Args:
            states: Tensor of shape (batch_size, 1, channels, height, width) - initial state
            actions: Tensor of shape (batch_size, seq_len, action_dim) - sequence of actions
            
        Returns:
            Tensor of shape (seq_len+1, batch_size, repr_dim) - sequence of predicted representations
        """
        batch_size = states.shape[0]
        seq_len = actions.shape[1]
        
        # Get initial state representation
        initial_repr = self.jepa.encode(states[:, 0])  # [B, repr_dim]
        
        # Initialize predictions with initial representation
        predictions = [initial_repr]
        
        # Sample latent variables for the entire sequence
        if self.jepa.use_latent:
            latents, _ = self.jepa.sample_latent(states, actions, deterministic=False)
        
        # Recurrently predict future representations
        current_repr = initial_repr
        
        for t in range(seq_len):
            action = actions[:, t]  # [B, action_dim]
            
            # Prepare predictor input
            if self.jepa.use_latent:
                z = latents[:, t]  # [B, latent_dim]
                pred_input = torch.cat([current_repr, action, z], dim=1)
            else:
                pred_input = torch.cat([current_repr, action], dim=1)
            
            # Predict next representation
            next_repr = self.jepa.predictor(pred_input)
            predictions.append(next_repr)
            
            # Update current representation for next step
            current_repr = next_repr
        
        # Stack predictions and transpose to match evaluator's expected shape
        predictions = torch.stack(predictions, dim=0)  # [T+1, B, repr_dim]
        
        return predictions