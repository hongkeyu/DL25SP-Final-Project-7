from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import JEPA, JEPAWorldModel
import glob


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "."  # Using current directory since data is in project root

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    device = get_device()
    
    # First create the JEPA model
    jepa_model = JEPA(
        input_channels=2,
        repr_dim=256,
        latent_dim=16,
        hidden_dim=128,
        use_latent=True,
        use_vicreg=True,
        device=device
    )
    
    # Then wrap it in JEPAWorldModel
    model = JEPAWorldModel(jepa_model=jepa_model)
    
    # Load weights if they exist
    try:
        model.load_state_dict(torch.load("model_weights.pth")['model_state_dict'])
    except FileNotFoundError:
        print("No saved model weights found. Starting with fresh model.")
    
    model = model.to(device)
    
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    model = load_model()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
