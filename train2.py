import argparse
import os
from pathlib import Path
from typing import Any, Dict, Union, cast

import matplotlib.figure
import matplotlib.pyplot as plt
import randomname
import torch
import torch.nn.functional as F
import torch.utils.tensorboard.writer
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from gamm23.config import add_arpgarse_arguments, load_config, process_arpgarse_arguments, save_config
from gamm23.data import load_datasets
from gamm23.networks import ItNet, Tiramisu, UNet
from gamm23.operators import DifferentialOperator, MatrixOperator
from gamm23.utils import seed


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ":16:8"

plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif", "font.weight": "bold", "font.size": 22})

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)


config = load_config(Path(__file__).parent.joinpath("config.yaml"))
parser = argparse.ArgumentParser()
add_arpgarse_arguments(parser, config)
args = parser.parse_args()
process_arpgarse_arguments(args, config)


seed(config.seed)
OUTPUT_PATH = Path(f"/work/ws-tmp/aa609734-GAMM/runs/tmp/{randomname.get_name()}")  # Path(__file__).parent.joinpath("runs", randomname.get_name())
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
LOG_IMG_COUNT = 4


matrix = torch.load(Path(__file__).parent.joinpath("src", "gamm23", "operators", "matrix.pth"))
A = MatrixOperator(matrix).to(config.device)
__D = DifferentialOperator().to_matrix(A.input_size).to(config.device)
__ATApaD2iAT = (A.to_matrix().mT @ A.to_matrix() + config.tik_alpha * __D.mT @ __D).inverse() @ A.to_matrix().mT
A_T = A.mT
A_Tik = MatrixOperator(__ATApaD2iAT).to(config.device)


train_dataset, val_dataset, test_dataset = load_datasets(Path(__file__).parent.joinpath("src", "gamm23", "data", "noisy"))
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=config.num_workers, worker_init_fn=lambda _: seed(torch.initial_seed()))
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=config.num_workers, worker_init_fn=lambda _: seed(torch.initial_seed()))
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=config.num_workers, worker_init_fn=lambda _: seed(torch.initial_seed()))


def save_checkpoint(base_path: Union[str, Path], name: str, data: Dict[str, Any]) -> None:
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    torch.save(data, base_path.joinpath(name + ".pth"))


def to_figure(*args: Tensor, **kwargs: Tensor) -> matplotlib.figure.Figure:
    figure = plt.figure()
    for data in args:
        plt.plot(data.detach().to("cpu").numpy())
    for name, data in kwargs.items():
        plt.plot(data.detach().to("cpu").numpy(), label=name)
    if len(kwargs) != 0:
        plt.legend()
    plt.tight_layout()
    return figure


network = Tiramisu(1, 1, cast(int, A.output_size), cast(int, A.input_size), dims=1).to(config.device)
optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=config.weight_decay, foreach=False)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_fn = lambda a, b: F.smooth_l1_loss(a, b)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1.0)
network.load_state_dict(torch.load(str(Path(__file__).parent.joinpath(f"runs/tiramisu/{config.ckpt}/final.pth").resolve()))["network"])
optimizer = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=0.0, foreach=False)
loss_fn = lambda a, b: ((a-b)**4).flatten(start_dim=1).mean(1).mean(0)


tb_logger = torch.utils.tensorboard.writer.SummaryWriter(str(OUTPUT_PATH.resolve()))
try:
    setattr(config, "slurm_id", os.environ["SLURM_ARRAY_JOB_ID"])
except:
    try:
        setattr(config, "slurm_id", os.environ["SLURM_JOB_ID"])
    except:
        pass
save_config(OUTPUT_PATH.joinpath("config.yaml"), config)

# ------------------------ VALIDATION ------------------------
loss_acc = 0.0
mse_acc = 0.0
adv_loss_acc = 0.0
adv_mse_acc = 0.0
for batch_no, (noisy_measurement, groundtruth) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):
    noisy_measurement, groundtruth = cast(Tensor, noisy_measurement), cast(Tensor, groundtruth)
    noisy_measurement, groundtruth = noisy_measurement.to(config.device), groundtruth.to(config.device)
    reconstruction = network(noisy_measurement.unsqueeze(1))[:, 0]
    loss_acc += loss_fn(reconstruction, groundtruth).item()
    mse_acc += F.mse_loss(reconstruction, groundtruth).item()
    # Adversarial attack
    """adversarial_measurement = noisy_measurement.clone()
    adversarial_measurement.requires_grad = True
    for _ in range(ADV_ITERATIONS):
        with torch.enable_grad():
            adv_reconstruction = network(adversarial_measurement.unsqueeze(1))[:,0]
            loss = loss_fn(adv_reconstruction, groundtruth)
            loss.backward()
        adversarial_measurement = adversarial_measurement + ADV_LR*torch.sign(adversarial_measurement.grad) # FGSM step
        adversarial_measurement = noisy_measurement + torch.clamp(adversarial_measurement-noisy_measurement, -ADV_EPSILON, ADV_EPSILON) # Projection
        adversarial_measurement.requires_grad = True
        torch.cuda.empty_cache()
    adv_loss_acc += loss_fn(adv_reconstruction, groundtruth).item()
    adv_mse_acc += F.mse_loss(adv_reconstruction, groundtruth).item()
    """
    if batch_no == 0:
        for i in range(min(groundtruth.shape[0], LOG_IMG_COUNT)):
            tb_logger.add_figure(f"val/output-{i}", to_figure(groundtruth=groundtruth[i], reconstruction=reconstruction[i]), 0)
            tb_logger.add_figure(f"val/noisy_measurement-{i}", to_figure(noisy_measurement=noisy_measurement[i]), 0)
tb_logger.add_scalar("val/loss", loss_acc / len(val_dataloader), 0)
tb_logger.add_scalar("val/mse", mse_acc / len(val_dataloader), 0)
best_loss = float("inf")
save_checkpoint(OUTPUT_PATH, "initial", {"network": network.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()})
for epoch_no in trange(config.epochs):
    # ------------------------ TRAINING ------------------------
    network.train()
    for batch_no, (noisy_measurement, groundtruth) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
        noisy_measurement, groundtruth = cast(Tensor, noisy_measurement), cast(Tensor, groundtruth)
        noisy_measurement, groundtruth = noisy_measurement.to(config.device), groundtruth.to(config.device)
        with torch.enable_grad():
            reconstruction = network(noisy_measurement.unsqueeze(1))[:, 0]
            loss = loss_fn(reconstruction, groundtruth)/config.batch_acc
        loss.backward()
        if ((batch_no + 1) % config.batch_acc == 0) or (batch_no + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()
        tb_logger.add_scalar("train/loss", loss_fn(reconstruction, groundtruth).item(), epoch_no * len(train_dataloader) + batch_no)
        tb_logger.add_scalar("train/mse", F.mse_loss(reconstruction, groundtruth).item(), epoch_no * len(train_dataloader) + batch_no)
        if loss.item() < best_loss:
            save_checkpoint(OUTPUT_PATH, "best", {"network": network.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()})
            best_loss = loss.item()
        torch.cuda.empty_cache()
    save_checkpoint(OUTPUT_PATH, f"epoch-{epoch_no}", {"network": network.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()})
    # ------------------------ VALIDATION ------------------------
    network.eval()
    loss_acc = 0.0
    mse_acc = 0.0
    adv_loss_acc = 0.0
    adv_mse_acc = 0.0
    for batch_no, (noisy_measurement, groundtruth) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):
        noisy_measurement, groundtruth = cast(Tensor, noisy_measurement), cast(Tensor, groundtruth)
        noisy_measurement, groundtruth = noisy_measurement.to(config.device), groundtruth.to(config.device)
        reconstruction = network(noisy_measurement.unsqueeze(1))[:, 0]
        loss_acc += loss_fn(reconstruction, groundtruth).item()
        mse_acc += F.mse_loss(reconstruction, groundtruth).item()
        # Adversarial attack
        """adversarial_measurement = noisy_measurement.clone()
        adversarial_measurement.requires_grad = True
        for _ in range(ADV_ITERATIONS):
            with torch.enable_grad():
                adv_reconstruction = network(adversarial_measurement.unsqueeze(1))[:,0]
                loss = loss_fn(adv_reconstruction, groundtruth)
                loss.backward()
            adversarial_measurement = adversarial_measurement + ADV_LR*torch.sign(adversarial_measurement.grad) # FGSM step
            adversarial_measurement = noisy_measurement + torch.clamp(adversarial_measurement-noisy_measurement, -ADV_EPSILON, ADV_EPSILON) # Projection
            adversarial_measurement.requires_grad = True
            torch.cuda.empty_cache()
        adv_loss_acc += loss_fn(adv_reconstruction, groundtruth).item()
        adv_mse_acc += F.mse_loss(adv_reconstruction, groundtruth).item()
        """
        if batch_no == 0:
            for i in range(min(groundtruth.shape[0], LOG_IMG_COUNT)):
                tb_logger.add_figure(f"val/output-{i}", to_figure(groundtruth=groundtruth[i], reconstruction=reconstruction[i]), epoch_no + 1)
                tb_logger.add_figure(f"val/noisy_measurement-{i}", to_figure(noisy_measurement=noisy_measurement[i]), epoch_no + 1)
    tb_logger.add_scalar("val/loss", loss_acc / len(val_dataloader), epoch_no + 1)
    tb_logger.add_scalar("val/mse", mse_acc / len(val_dataloader), epoch_no + 1)
    lr_scheduler.step()
    for i, param_group in enumerate(optimizer.param_groups):
        tb_logger.add_scalar(f"lr-{i}", param_group["lr"], epoch_no + 1)
    # Preliminary termination
    if epoch_no == 49 and loss_acc / len(val_dataloader) >= 0.1:
        print("PRELIMINARY TERMINATION DUE TO BAD PERFORMANCE!")
        break
save_checkpoint(OUTPUT_PATH, "final", {"network": network.state_dict(), "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()})
# ------------------------ TESTING ------------------------
loss_acc = 0.0
mse_acc = 0.0
network.eval()
for batch_no, (noisy_measurement, groundtruth) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    noisy_measurement, groundtruth = cast(Tensor, noisy_measurement), cast(Tensor, groundtruth)
    noisy_measurement, groundtruth = noisy_measurement.to(config.device), groundtruth.to(config.device)
    reconstruction = network(noisy_measurement.unsqueeze(1))[:, 0]
    loss_acc += loss_fn(reconstruction, groundtruth).item()
    mse_acc += F.mse_loss(reconstruction, groundtruth).item()
    if batch_no == 0:
        for i in range(min(groundtruth.shape[0], LOG_IMG_COUNT)):
            tb_logger.add_figure(f"test/output-{i}", to_figure(groundtruth=groundtruth[i], reconstruction=reconstruction[i]), 0)
            tb_logger.add_figure(f"test/noisy_measurement-{i}", to_figure(noisy_measurement=noisy_measurement[i]), 0)
tb_logger.add_scalar("test/loss", loss_acc / len(test_dataloader), 0)
tb_logger.add_scalar("test/mse", mse_acc / len(test_dataloader), 0)
