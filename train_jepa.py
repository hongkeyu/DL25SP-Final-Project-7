import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# Import your JEPA model
from models import JEPAWorldModel, JEPA

# Import dataset and other components
from dataset import create_wall_dataloader
from schedulers import LRSchedule, Scheduler
from normalizer import Normalizer
from evaluator import ProbingEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train a JEPA model for the wall environment')
    parser.add_argument('--data_path', type=str, default='.', help='Path to data directory')
    parser.add_argument('--base_save_dir', type=str, default='./outputs', help='Base directory to save runs')
    parser.add_argument('--run_name', type=str, default=None, help='Optional name for the run (overrides timestamp)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--repr_dim', type=int, default=256, help='Representation dimension')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--no_latent', action='store_true', help='Disable latent variables')
    parser.add_argument('--no_vicreg', action='store_true', help='Disable VICReg regularization')
    parser.add_argument('--reg_weight', type=float, default=0.1, help='Weight for latent regularization')
    parser.add_argument('--vicreg_weight', type=float, default=1.0, help='Weight for VICReg loss')
    parser.add_argument('--lambda_var', type=float, default=25.0, help='Weight for variance term in VICReg')
    parser.add_argument('--lambda_cov', type=float, default=1.0, help='Weight for covariance term in VICReg')
    parser.add_argument('--log_interval', type=int, default=100, help='Frequency for logging batch metrics to TensorBoard')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def create_optimizer_and_scheduler(model, args, train_loader):
    """Create optimizer and scheduler for training the model."""
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs
    
    scheduler = Scheduler(
        schedule=LRSchedule.Cosine,
        base_lr=args.lr,
        data_loader=train_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        batch_size=args.batch_size
    )
    
    return optimizer, scheduler


def train_jepa(args, train_loader, val_loader, model, device, save_dir):
    """Train the JEPA model with VICReg regularization."""
    model.to(device)
    
    # Initialize TensorBoard writer using the unique save_dir
    # The writer will create subdirectories within save_dir if needed
    writer = SummaryWriter(log_dir=save_dir)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, train_loader)
    
    # Define model save paths within the unique save_dir
    best_model_save_path = os.path.join(save_dir, 'model_weights_best.pth')
    final_model_save_path = os.path.join(save_dir, 'model_weights_final.pth')
    checkpoints_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    step = 0
    
    jepa = model.jepa  # Access the JEPA model inside JEPAWorldModel
    
    # Wrap epoch loop with tqdm
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Training
        model.train()
        train_loss = 0
        pred_loss_total = 0
        reg_loss_total = 0
        vicreg_loss_total = 0
        
        # Wrap batch loop with tqdm
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training", leave=False)
        for batch_idx, batch in enumerate(batch_iterator):
            states = batch.states.to(device)  # [B, T+1, C, H, W]
            actions = batch.actions.to(device)  # [B, T, 2]
            
            # Forward pass
            # Get initial state and encode it
            init_state = states[:, 0:1]  # [B, 1, C, H, W]
            
            # Get predicted representations
            pred_reprs = model(init_state, actions)  # [T+1, B, repr_dim]
            
            # Get ground truth representations
            true_reprs = jepa.encode(states)  # [B, T+1, repr_dim]
            true_reprs = true_reprs.transpose(0, 1)  # [T+1, B, repr_dim]
            
            # Calculate prediction loss - MSE between predicted and true representations
            pred_loss = F.mse_loss(pred_reprs, true_reprs)
            
            # Calculate regularization losses
            
            # 1. Latent variable regularization
            if jepa.use_latent:
                _, latent_info = jepa.sample_latent(init_state, actions, deterministic=False)
                reg_loss = jepa.compute_latent_regularization(latent_info)
            else:
                reg_loss = torch.tensor(0.0).to(device)
            
            # 2. VICReg regularization to prevent collapse
            if not args.no_vicreg:
                # Apply VICReg to final predicted representations in batch
                last_reprs = pred_reprs[-1]  # [B, repr_dim]
                vicreg_loss, vicreg_info = jepa.compute_vicreg_loss(
                    last_reprs, 
                    lambda_var=args.lambda_var,
                    lambda_cov=args.lambda_cov
                )
            else:
                vicreg_loss = torch.tensor(0.0).to(device)
                vicreg_info = {"var_loss": 0.0, "cov_loss": 0.0}
            
            # Total loss
            loss = pred_loss + args.reg_weight * reg_loss + args.vicreg_weight * vicreg_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.adjust_learning_rate(step)
            
            # Log progress and write to TensorBoard
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, step)
            
            train_loss += loss.item()
            pred_loss_total += pred_loss.item()
            reg_loss_item = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0
            vicreg_loss_item = vicreg_loss.item() if isinstance(vicreg_loss, torch.Tensor) else 0
            reg_loss_total += reg_loss_item
            vicreg_loss_total += vicreg_loss_item
            
            # Removed interval print, but keep TensorBoard logging interval
            if step % args.log_interval == 0:
                # Write batch losses to TensorBoard
                writer.add_scalar('Loss/train_batch', loss.item(), step)
                writer.add_scalar('Loss/pred_batch', pred_loss.item(), step)
                writer.add_scalar('Loss/reg_batch', reg_loss_item, step)
                writer.add_scalar('Loss/vicreg_batch', vicreg_loss_item, step)
                if not args.no_vicreg:
                    writer.add_scalar('VICReg/var_loss_batch', vicreg_info['var_loss'], step)
                    writer.add_scalar('VICReg/cov_loss_batch', vicreg_info['cov_loss'], step)
                
                # Update tqdm description with current loss
                batch_iterator.set_postfix(loss=f"{loss.item():.4f}", pred=f"{pred_loss.item():.4f}", reg=f"{reg_loss_item:.4f}", vic=f"{vicreg_loss_item:.4f}")

            step += 1
        
        avg_train_loss = train_loss / len(train_loader)
        avg_pred_loss = pred_loss_total / len(train_loader)
        avg_reg_loss = reg_loss_total / len(train_loader)
        avg_vicreg_loss = vicreg_loss_total / len(train_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} completed. "
              f"Avg Train Loss: {avg_train_loss:.4f}, "
              f"Avg Pred Loss: {avg_pred_loss:.4f}, "
              f"Avg Reg Loss: {avg_reg_loss:.4f}, "
              f"Avg VICReg Loss: {avg_vicreg_loss:.4f}")
        
        # Write average epoch training losses to TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/pred_epoch', avg_pred_loss, epoch)
        writer.add_scalar('Loss/reg_epoch', avg_reg_loss, epoch)
        writer.add_scalar('Loss/vicreg_epoch', avg_vicreg_loss, epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        
        # Wrap validation loop with tqdm (optional, but consistent)
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Validation", leave=False)
        with torch.no_grad():
            for batch in val_iterator:
                states = batch.states.to(device)
                actions = batch.actions.to(device)
                
                # Forward pass
                init_state = states[:, 0:1]
                pred_reprs = model(init_state, actions)
                
                # Get ground truth representations
                true_reprs = jepa.encode(states)
                true_reprs = true_reprs.transpose(0, 1)
                
                # Calculate prediction loss
                loss = F.mse_loss(pred_reprs, true_reprs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Write validation loss to TensorBoard
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # --- Save checkpoint for the current epoch ---
        epoch_save_path = os.path.join(checkpoints_dir, f'model_epoch_{epoch+1:04d}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, epoch_save_path)
        # --- End save checkpoint ---

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save to the renamed best model path
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, best_model_save_path)
            print(f"New best model saved to {best_model_save_path}")
    
    # Save final model (using the previously defined final_model_save_path)
    torch.save({
        'epoch': args.epochs - 1, # Correct epoch number for final save
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss, # Last validation loss
    }, final_model_save_path)
    print(f"Final model saved to {final_model_save_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    return model


def main():
    args = parse_args()
    
    # --- Create unique save directory ---
    base_save_dir = args.base_save_dir
    if args.run_name:
        run_identifier = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_identifier = f"run_{timestamp}"
    
    save_dir = os.path.join(base_save_dir, run_identifier)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving outputs to: {save_dir}")
    # --- End create unique save directory ---

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Create data loaders
    train_loader = create_wall_dataloader(
        data_path=f"{args.data_path}/train",
        probing=False,
        device=device,
        batch_size=args.batch_size,
        train=True,
    )
    
    val_loader = create_wall_dataloader(
        data_path=f"{args.data_path}/probe_normal/val",
        probing=False,
        device=device,
        batch_size=args.batch_size,
        train=False,
    )
    
    # Create JEPA model first
    jepa_model = JEPA(
        input_channels=2,
        repr_dim=args.repr_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        use_latent=not args.no_latent,
        use_vicreg=not args.no_vicreg,
        device=device
    )
    
    # Then create JEPAWorldModel with the JEPA model
    model = JEPAWorldModel(jepa_model=jepa_model)
    
    # Print model architecture and parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Train model
    trained_model = train_jepa(args, train_loader, val_loader, model, device, save_dir)


if __name__ == "__main__":
    main()