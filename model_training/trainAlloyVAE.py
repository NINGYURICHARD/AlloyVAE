from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset import MapDataset
from CVAEWC import CVAE


RUN_NAME = "cvae_wc_run"
BATCH_SIZE = 50
LEARNING_RATE = 1e-4
EPOCHS = 800
TRAIN_RATIO = 0.9
LATENT_DIM = 300
LR_STEP = 500
NUM_WORKERS = 0

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "dataset"
CHECKPOINT_ROOT = SCRIPT_DIR / "checkpoints"

X_DATA_PATH = DATA_DIR / "X_WC_3500.npy"
Y_DATA_PATH = DATA_DIR / "Y_WC_3500.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_training_data():
    conditions = np.load(X_DATA_PATH)
    stress = np.load(Y_DATA_PATH)

    # Keep the same channel selection logic as the original script.
    conditions = np.concatenate((conditions[:, :2, :, :, :], conditions[:, 3:, :, :, :]), axis=1)
    n_samples = len(conditions)
    split_idx = int(n_samples * TRAIN_RATIO)

    conditions_train = torch.tensor(conditions[:split_idx], dtype=torch.float32)
    stress_train = torch.tensor(stress[:split_idx], dtype=torch.float32)
    return conditions_train, stress_train


def create_run_dir(run_name: str) -> Path:
    run_dir = CHECKPOINT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ("trainCVAEWC2_github.py", "CVAEWC.py"):
        src = SCRIPT_DIR / file_name
        if src.exists():
            shutil.copy2(src, run_dir / file_name)

    return run_dir


def loss_function(recon_x, x, mu, logvar, nconcen_p, nconcen_t, nsro_p, nsro_t, theta, beta, gamma, alpha):
    mse = nn.MSELoss()
    mse_recon = mse(recon_x, x)
    mse_concen = mse(nconcen_p, nconcen_t)
    mse_sro = mse(nsro_p, nsro_t)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = theta * mse_recon + beta * mse_concen + gamma * mse_sro + alpha * kld
    return total, mse_recon, mse_concen, mse_sro, kld


def cyclic_beta_scheduler(epoch: int):
    if epoch < 300:
        return 30000, 1, 1, 0.0001
    return 30000, 10000, 10000, 0.0001


def init_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight, gain=1.2)
        init.constant_(module.bias, 0)


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading data from: {X_DATA_PATH}")
    print(f"Loading data from: {Y_DATA_PATH}")

    conditions_train, stress_train = load_training_data()
    train_dataset = MapDataset(conditions_train, stress_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    run_dir = create_run_dir(RUN_NAME)
    print(f"Saving outputs to: {run_dir}")

    model = CVAE(6, 2, 9, LATENT_DIM).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=0.1)

    mse_recon_store = []
    mse_concen_store = []
    mse_sro_store = []
    kld_store = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        theta, beta, gamma, alpha = cyclic_beta_scheduler(epoch)

        for condition, target in train_loader:
            condition = condition.to(DEVICE)
            target = target.to(DEVICE)

            reconstruction, mu, logvar, nconcen_p, nconcen_t, nsro_p, nsro_t = model(
                target,
                condition[:, :2, :, :, :],
                condition[:, 2:, :, :, :],
            )

            loss, mse_recon, mse_concen, mse_sro, kld = loss_function(
                reconstruction,
                target,
                mu,
                logvar,
                nconcen_p,
                nconcen_t,
                nsro_p,
                nsro_t,
                theta,
                beta,
                gamma,
                alpha,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.6f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        print(
            "MSE_recon=",
            mse_recon.detach().cpu().item(),
            "MSE_concen=",
            mse_concen.detach().cpu().item(),
            "MSE_SRO=",
            mse_sro.detach().cpu().item(),
            "KLD=",
            kld.detach().cpu().item(),
            "beta=",
            beta,
            "alpha=",
            alpha,
        )

        mse_recon_store.append(mse_recon.detach().cpu().item())
        mse_concen_store.append(mse_concen.detach().cpu().item())
        mse_sro_store.append(mse_sro.detach().cpu().item())
        kld_store.append(kld.detach().cpu().item())

    np.save(run_dir / "MseReconStore.npy", np.array(mse_recon_store))
    np.save(run_dir / "MseConcenStore.npy", np.array(mse_concen_store))
    np.save(run_dir / "MseSROStore.npy", np.array(mse_sro_store))
    np.save(run_dir / "KLDStore.npy", np.array(kld_store))

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, run_dir / f"{RUN_NAME}.tar")


if __name__ == "__main__":
    main()
