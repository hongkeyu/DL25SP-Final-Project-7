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
        layers.append(nn.GELU())  # Changed from ReLU to GELU
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return self.act(out)


class CoordinateEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2, channels, 1)
    
    def forward(self, x):
        batch, _, h, w = x.shape
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).repeat(batch, 1, 1, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).repeat(batch, 1, h, 1)
        coords = torch.cat([x_coords, y_coords], dim=1)
        return self.conv(coords)


class WallDoorDetector(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Horizontal edge detector
        self.horizontal_conv = nn.Conv2d(in_channels, out_channels//2, kernel_size=(1, 5), padding=(0, 2))
        # Vertical edge detector
        self.vertical_conv = nn.Conv2d(in_channels, out_channels//2, kernel_size=(5, 1), padding=(2, 0))
        
    def forward(self, x):
        h_edges = self.horizontal_conv(x)
        v_edges = self.vertical_conv(x)
        return torch.cat([h_edges, v_edges], dim=1)


class PositionAwareHead(nn.Module):
    def __init__(self, repr_dim, hidden_dim=128):
        super().__init__()
        self.position_decoder = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # Output (x, y) coordinates
        )
    
    def forward(self, representations):
        return self.position_decoder(representations)


class RoomDetector(nn.Module):
    def __init__(self, repr_dim):
        super().__init__()
        self.room_classifier = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # Binary: room 0 or room 1
        )
    
    def forward(self, representations):
        return self.room_classifier(representations)


class TrajectoryEncoder(nn.Module):
    def __init__(self, repr_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(repr_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, repr_dim)
    
    def forward(self, repr_sequence):
        _, (h_n, _) = self.lstm(repr_sequence)
        return self.proj(h_n[-1])


class AttentivePredictor(nn.Module):
    def __init__(self, repr_dim, action_dim, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.repr_proj = nn.Linear(repr_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, repr_dim)
    
    def forward(self, repr, action):
        repr_proj = self.repr_proj(repr).unsqueeze(0)
        action_proj = self.action_proj(action).unsqueeze(0)
        
        # Cross-attention between representation and action
        attended, _ = self.attention(repr_proj, action_proj, action_proj)
        return self.output_proj(attended.squeeze(0))


class JEPA(nn.Module):
    """
    Improved Joint Embedding Predictive Architecture (JEPA) for the two-room environment.
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
        
        # Coordinate embedding for spatial awareness
        self.coord_embedding = CoordinateEmbedding(32)
        
        # Wall/Door detector
        self.wall_door_detector = WallDoorDetector(input_channels, 32)
        
        # Encoder: takes states (images) and encodes them into representations
        self.encoder = nn.Sequential(
            ResidualBlock(input_channels + 32 + 32, 64, stride=1),  # +32 for coord embedding, +32 for wall/door
            ResidualBlock(64, 128, stride=2),  # 33x33
            ResidualBlock(128, 256, stride=2),  # 17x17
            ResidualBlock(256, 512, stride=2),  # 9x9
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 5x5
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(512 * 5 * 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim)
        )
        
        # Position-aware head
        self.position_head = PositionAwareHead(repr_dim, hidden_dim)
        
        # Room detector
        self.room_detector = RoomDetector(repr_dim)
        
        # Trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(repr_dim, hidden_dim)
        
        # Expander: projects representation to higher dimension for VICReg
        if use_vicreg:
            self.expander = nn.Sequential(
                nn.Linear(repr_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim * 2)
            )
        
        # Attentive predictor
        action_input_dim = 2  # (delta_x, delta_y)
        self.attentive_predictor = AttentivePredictor(repr_dim, action_input_dim, hidden_dim)
        
        # Predictor with conditional branches for still vs moving
        predictor_input_dim = repr_dim + action_input_dim
        if use_latent:
            predictor_input_dim += latent_dim
        
        self.move_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim)
        )
        
        self.still_predictor = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, repr_dim)
        )
        
        if use_latent:
            # Latent predictor with beta-VAE style regularization
            self.latent_predictor = nn.Sequential(
                nn.Linear(repr_dim + action_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log_std
            )
    
    def augment_state(self, state: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations while preserving wall structure."""
        if self.training:
            # Only augment agent position, not walls
            agent_channel = state[:, 0:1]  # Assuming agent is in channel 0
            wall_channel = state[:, 1:2]   # Assuming walls are in channel 1
            
            # Random agent position noise
            agent_noise = torch.randn_like(agent_channel) * 0.05
            agent_channel = agent_channel + agent_noise
            
            # Recombine channels
            state = torch.cat([agent_channel, wall_channel], dim=1)
        
        return state
    
    def encode(self, states: torch.Tensor) -> torch.Tensor:
        """Encode states to representations with spatial awareness."""
        # Handle different input shapes
        if states.dim() == 5:  # [B, T, C, H, W]
            batch_size, seq_len = states.shape[0], states.shape[1]
            states_flat = states.reshape(-1, *states.shape[2:])
            
            # Apply augmentation if training
            states_flat = self.augment_state(states_flat)
            
            # Add coordinate embedding
            coord_emb = self.coord_embedding(states_flat)
            # Add wall/door detection features
            wall_door_features = self.wall_door_detector(states_flat)
            
            # Concatenate all features
            enhanced_states = torch.cat([states_flat, coord_emb, wall_door_features], dim=1)
            
            encodings_flat = self.encoder(enhanced_states)
            encodings = encodings_flat.reshape(batch_size, seq_len, self.repr_dim)
            return encodings
        elif states.dim() == 4:  # [B, C, H, W]
            states = self.augment_state(states)
            coord_emb = self.coord_embedding(states)
            wall_door_features = self.wall_door_detector(states)
            enhanced_states = torch.cat([states, coord_emb, wall_door_features], dim=1)
            return self.encoder(enhanced_states)
        else:
            raise ValueError(f"Unexpected states shape: {states.shape}")
    
    def sample_latent(self, 
                     states: torch.Tensor, 
                     actions: torch.Tensor, 
                     deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Sample latent variables with beta-VAE style regularization."""
        if not self.use_latent:
            batch_size, seq_len = actions.shape[0], actions.shape[1]
            return torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device), {}
        
        batch_size, seq_len = actions.shape[0], actions.shape[1]
        
        # Get current state representation
        if states.dim() == 5:
            current_repr = self.encode(states[:, 0:1])[:, 0]
        elif states.dim() == 4:
            current_repr = self.encoder(states)
        else:
            raise ValueError(f"Unexpected states shape: {states.shape}")
        
        latents = []
        log_probs = []
        means = []
        log_stds = []
        
        for t in range(seq_len):
            action = actions[:, t]
            
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
                
                # Calculate log probability
                log_prob = -0.5 * ((z - mean) / std).pow(2) - log_std - 0.5 * np.log(2 * np.pi)
                log_prob = log_prob.sum(dim=1)
                log_probs.append(log_prob)
            
            latents.append(z)
            
            # Update representation using attention-based predictor
            current_repr = self.attentive_predictor(current_repr, action)
        
        latents = torch.stack(latents, dim=1)
        
        info = {}
        if not deterministic and log_probs:
            info['log_probs'] = torch.stack(log_probs, dim=1)
        
        info['means'] = torch.stack(means, dim=1) if means else None
        info['log_stds'] = torch.stack(log_stds, dim=1) if log_stds else None
        
        return latents, info
    
    def forward(self, 
                states: torch.Tensor, 
                actions: torch.Tensor, 
                latents: Optional[torch.Tensor] = None,
                deterministic: bool = False) -> torch.Tensor:
        """Forward pass with action-conditioned prediction."""
        batch_size, seq_len = actions.shape[0], actions.shape[1]
        
        # Ensure states has the right shape
        if states.dim() == 4:
            states = states.unsqueeze(1)
        
        # Encode initial state
        initial_repr = self.encode(states[:, 0:1])[:, 0]
        
        # Sample latent variables if not provided
        if latents is None and self.use_latent:
            latents, _ = self.sample_latent(states, actions, deterministic)
        
        predictions = [initial_repr]
        current_repr = initial_repr
        
        for t in range(seq_len):
            action = actions[:, t]
            
            # Detect if action is near-zero (stationary)
            is_still = torch.norm(action, dim=-1) < 0.01
            
            # Prepare predictor input
            if self.use_latent:
                z = latents[:, t]
                pred_input = torch.cat([current_repr, action, z], dim=1)
                move_output = self.move_predictor(pred_input)
            else:
                pred_input = torch.cat([current_repr, action], dim=1)
                move_output = self.move_predictor(pred_input)
            
            still_output = self.still_predictor(current_repr)
            
            # Apply appropriate predictor based on movement
            if is_still.any():
                still_mask = is_still.unsqueeze(-1).expand_as(move_output)
                next_repr = torch.where(still_mask, still_output, move_output)
            else:
                next_repr = move_output
            
            predictions.append(next_repr)
            current_repr = next_repr
        
        predictions = torch.stack(predictions, dim=1)
        return predictions
    
    def compute_vicreg_loss(self, 
                           representations: torch.Tensor, 
                           lambda_var: float = 25.0, 
                           lambda_cov: float = 1.0,
                           lambda_invariance: float = 1.0,
                           augmented_representations: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """Enhanced VICReg loss with invariance."""
        if not self.use_vicreg:
            return torch.tensor(0.0).to(self.device), {"var_loss": 0.0, "cov_loss": 0.0, "invariance_loss": 0.0}
        
        # Project representations to higher dimension
        projections = self.expander(representations)
        
        # Center the projections
        projections = projections - projections.mean(dim=0)
        
        # Variance loss
        std = torch.sqrt(projections.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std))
        
        # Covariance loss
        batch_size = projections.size(0)
        cov = (projections.T @ projections) / (batch_size - 1)
        mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
        cov_loss = (cov[mask]**2).mean()
        
        # Invariance loss if augmented representations are provided
        invariance_loss = torch.tensor(0.0).to(self.device)
        if augmented_representations is not None:
            aug_projections = self.expander(augmented_representations)
            invariance_loss = F.mse_loss(projections, aug_projections)
        
        # Total loss
        loss = lambda_var * var_loss + lambda_cov * cov_loss + lambda_invariance * invariance_loss
        
        return loss, {
            "var_loss": var_loss.item(), 
            "cov_loss": cov_loss.item(),
            "invariance_loss": invariance_loss.item()
        }
    
    def compute_latent_regularization(self, latent_info: Dict, beta: float = 1.0) -> torch.Tensor:
        """Beta-VAE style regularization for latent variables."""
        if not self.use_latent or 'means' not in latent_info:
            return torch.tensor(0.0).to(self.device)
        
        means = latent_info['means']
        log_stds = latent_info['log_stds']
        
        # KL divergence to standard normal prior
        kl_loss = -0.5 * torch.sum(1 + 2 * log_stds - means.pow(2) - torch.exp(2 * log_stds))
        return beta * kl_loss / means.size(0)
    
    def position_aware_loss(self, pred_repr, target_positions):
        """Position prediction loss for better (x,y) coordinate estimation."""
        pred_pos = self.position_head(pred_repr)
        pos_loss = F.mse_loss(pred_pos, target_positions)
        return pos_loss
    
    def room_consistency_loss(self, representations, actions):
        """Ensure smooth transitions between rooms."""
        room_probs = self.room_detector(representations)
        
        # Calculate room transition consistency
        # This is a simplified version - you'd need to implement proper logic
        # based on whether the agent is crossing through a door
        consistency_loss = torch.tensor(0.0).to(self.device)
        
        return consistency_loss
    
    def layout_contrastive_loss(self, repr1, repr2, same_layout):
        """Contrastive loss to distinguish different layouts."""
        similarity = F.cosine_similarity(repr1, repr2)
        
        if same_layout:
            loss = 1 - similarity
        else:
            loss = F.relu(similarity - 0.2)
        
        return loss.mean()


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
            
            # Detect if action is near-zero (stationary)
            is_still = torch.norm(action, dim=-1) < 0.01
            
            # Prepare predictor input
            if self.jepa.use_latent:
                z = latents[:, t]  # [B, latent_dim]
                pred_input = torch.cat([current_repr, action, z], dim=1)
                move_output = self.jepa.move_predictor(pred_input)
            else:
                pred_input = torch.cat([current_repr, action], dim=1)
                move_output = self.jepa.move_predictor(pred_input)
            
            still_output = self.jepa.still_predictor(current_repr)
            
            # Apply appropriate predictor based on movement
            if is_still.any():
                still_mask = is_still.unsqueeze(-1).expand_as(move_output)
                next_repr = torch.where(still_mask, still_output, move_output)
            else:
                next_repr = move_output
            
            predictions.append(next_repr)
            current_repr = next_repr
        
        # Stack predictions and transpose to match evaluator's expected shape
        predictions = torch.stack(predictions, dim=0)  # [T+1, B, repr_dim]
        
        return predictions