import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # :16:8
from pathlib import Path
import random

import matplotlib.pyplot  as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm, trange

from unet import UNet
from tiramisu import Tiramisu
from itnet import ItNet

plt.rcParams.update({
    "text.usetex": True,
    "font.family" : "sans-serif",
    "font.weight" : "bold",
    "font.size"   : 22
})

torch.set_grad_enabled(False)

def seed(seed: int):
    if seed < 0:
        return
    seed = seed%2^32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


SUFFIX = "clean"


SEED = 314176
seed(SEED)
DEVICE = torch.device("cuda:0")
NUM_WORKERS = 0 #os.cpu_count()

N_TRAIN_SAMPLES = 1024 # 64% * 1600
N_VAL_SAMPLES = 256    # 16% * 1600
N_TEST_SAMPLES = 320   # 20% * 1600
NOISE_LEVEL_VALTEST = 0.03 # 0.0
NOISE_LEVEL_TRAIN = 0.0 # NOISE_LEVEL_VALTEST

GRAD_BATCH_SIZE = 128
NOGRAD_BATCH_SIZE = 4096

DATA_SIZE = 1024
MEASUREMENT_SIZE = 512


TIK_ALPHA = 0.1

UNET_LR = 1e-5
UNET_EPOCHS = 5000
UNET_ADV_LR = 1e0
UNET_ADV_ITERATIONS = 10
UNET_ADV_EPSILON = 1.0

TIRAMISU_LR = 1e-5
TIRAMISU_EPOCHS = 5000
TIRAMISU_ADV_LR = 1e0
TIRAMISU_ADV_ITERATIONS = 10
TIRAMISU_ADV_EPSILON = 1.0

ITNET_LR = 1e-5
ITNET_EPOCHS = 5000
ITNET_INTERN_LR = 1e-3
ITNET_INTERN_ITERATIONS = 5
ITNET_ADV_LR = 1e0
ITNET_ADV_ITERATIONS = 10
ITNET_ADV_EPSILON = 1.0



#__A = torch.eye(DATA_SIZE, device=DEVICE).unsqueeze(0)
__A = 0.05*torch.randn((MEASUREMENT_SIZE,DATA_SIZE), device=DEVICE).unsqueeze(0)
__D = torch.eye(DATA_SIZE, device=DEVICE)-torch.diag(torch.ones((DATA_SIZE-1), device=DEVICE), -1).unsqueeze(0)
__ATApaD2iAT = (__A.mT@__A+TIK_ALPHA*__D.mT@__D).inverse()@__A.mT
A = lambda x: (__A@x.unsqueeze(-1))[...,0]
A_T = lambda x: (__A.mT@x.unsqueeze(-1))[...,0]
A_pinv = lambda x: (__ATApaD2iAT@x.unsqueeze(-1))[...,0]



def generate_signals(
    N_samples: int,
    noise_level: float = 0.03,
    N_signal: int = 1000,
    min_jumps: int = 10, 
    max_jumps: int = 20, 
    min_distance: int = 40, 
    boundary_width: int = 20, 
    height_std: float = 1.0,
    min_height: float = 0.2
) -> Tensor:
    #Calculate positions by generating min_distance-spaced positions
    #and randomly rightshifting by the cumulatice sum
    derivative = torch.zeros((N_samples,N_signal), device=DEVICE)
    for i in range(N_samples):
        jump_count = random.randint(min_jumps-2, max_jumps-2)
        delta = torch.rand((jump_count-1), device=DEVICE) # One more than needed (last will be discarded, as it would lead to using the right boundary as last value, due to the normalization)
        delta = torch.cumsum((N_signal-2*boundary_width-(jump_count-2)*min_distance)*delta/delta.sum(), 0)[:-1].ceil().to(torch.long)
        jump_positions = boundary_width+min_distance*torch.arange(jump_count-2, device=DEVICE)+delta.sort()[0]
        values = height_std*torch.randn((jump_count,), device=DEVICE)
        values[torch.abs(values)<min_height] = values[torch.abs(values)<min_height].sign()*min_height
        derivative[i,torch.cat([torch.tensor([0], device=DEVICE),jump_positions,torch.tensor([0], device=DEVICE)])] = values
    groundtruths = derivative.cumsum(1)
    measurements = A(groundtruths)
    noisy_measurement = measurements+noise_level*torch.randn_like(measurements)
    return noisy_measurement, groundtruths
train_dataset = TensorDataset(*generate_signals(N_TRAIN_SAMPLES, noise_level=NOISE_LEVEL_TRAIN, N_signal=DATA_SIZE))
val_dataset = TensorDataset(*generate_signals(N_VAL_SAMPLES, noise_level=NOISE_LEVEL_VALTEST, N_signal=DATA_SIZE))
test_dataset = TensorDataset(*generate_signals(N_TEST_SAMPLES, noise_level=NOISE_LEVEL_VALTEST, N_signal=DATA_SIZE))
train_dataloader = DataLoader(train_dataset, batch_size=GRAD_BATCH_SIZE, shuffle=True, drop_last=False, num_workers=NUM_WORKERS, worker_init_fn=lambda _: seed(torch.initial_seed()))
val_dataloader = DataLoader(val_dataset, batch_size=NOGRAD_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, worker_init_fn=lambda _: seed(torch.initial_seed()))
test_dataloader = DataLoader(test_dataset, batch_size=NOGRAD_BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS, worker_init_fn=lambda _: seed(torch.initial_seed()))
noisy_measurement, groundtruth = train_dataset[0]


class TensorModule(nn.Module):
    def __init__(self, initial_value: Tensor) -> None:
        super().__init__()
        self.__tensor = torch.nn.parameter.Parameter(initial_value)
    @property
    def tensor(self) -> torch.Tensor:
        return self.__tensor
def total_variation(x: Tensor) -> Tensor:
    return F.conv1d(x.unsqueeze(1), torch.tensor([[[-1.0,1.0,0.0]]], device=DEVICE), padding=1).abs().mean()



unet = UNet(1, 1, A_pinv, dims=1).to(DEVICE)
optimizer = torch.optim.Adam(unet.parameters(), lr=UNET_LR, weight_decay=1e-3)
train_losses = []
val_losses = []
val_adv_losses = []
loss_acc = 0.0
loss_fn = lambda a,b: F.smooth_l1_loss(a, b)
for noisy_measurement, groundtruth in val_dataloader:
    noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
    # ---------------- Forward ----------------
    reconstruction = unet(noisy_measurement.unsqueeze(1))[:,0]
    # ---------------- Evaluation -------------
    loss_acc += loss_fn(reconstruction, groundtruth).item()
val_losses.append(loss_acc/len(test_dataloader))
for _ in trange(UNET_EPOCHS):
    unet.train()
    for noisy_measurement, groundtruth in train_dataloader:
        noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
        # ---------------- Training ---------------
        with torch.enable_grad():
            reconstruction = unet(noisy_measurement.unsqueeze(1))[:,0]
            loss = loss_fn(reconstruction, groundtruth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ---------------- Evaluation -------------
        train_losses.append(loss_fn(reconstruction, groundtruth).item())
        torch.cuda.empty_cache()
    unet.eval()
    loss_acc = 0.0
    adv_loss_acc = 0.0
    for noisy_measurement, groundtruth in val_dataloader:
        noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
        # ---------------- Forward ----------------
        reconstruction = unet(noisy_measurement.unsqueeze(1))[:,0]
        # ---------------- Evaluation -------------
        loss_acc += loss_fn(reconstruction, groundtruth).item()
        # ------------ Adversarial attack ---------
        adversarial_measurement = noisy_measurement.clone()
        adversarial_measurement.requires_grad = True
        for _ in range(UNET_ADV_ITERATIONS):
            with torch.enable_grad():
                reconstruction = unet(adversarial_measurement.unsqueeze(1))[:,0]
                loss = loss_fn(reconstruction, groundtruth)
                loss.backward()
            adversarial_measurement = adversarial_measurement + UNET_ADV_LR*torch.sign(adversarial_measurement.grad) # FGSM step
            adversarial_measurement = noisy_measurement + torch.clamp(adversarial_measurement-noisy_measurement, -UNET_ADV_EPSILON, UNET_ADV_EPSILON) # Projection
            adversarial_measurement.requires_grad = True
            torch.cuda.empty_cache()
        # ----- Adversarial attack evaluation -----
        adv_loss_acc += loss_fn(reconstruction, groundtruth).item()
    val_losses.append(loss_acc/len(test_dataloader))
    val_adv_losses.append(adv_loss_acc/len(test_dataloader))
save_dir = f"train/unet_{SUFFIX}"
Path(save_dir).mkdir(parents=True, exist_ok=True)
torch.save(torch.tensor(train_losses), str(Path(save_dir, "train_losses.pth").resolve()))
torch.save(torch.tensor(val_losses), str(Path(save_dir, "val_losses.pth").resolve()))
torch.save(torch.tensor(val_adv_losses), str(Path(save_dir, "val_adv_losses.pth").resolve()))
torch.save(unet.state_dict(), str(Path(save_dir, "weights.pth").resolve()))
plt.subplots(1,3,figsize=(21,5))
plt.subplot(1,3,1)
plt.title("Training loss")
plt.plot(train_losses)
plt.yscale("log")
plt.subplot(1,3,2)
plt.title("Validation loss")
plt.plot([x*len(train_dataloader) for x in range(len(val_losses))], val_losses)
plt.yscale("log")
plt.subplot(1,3,3)
plt.title("Adversarial validation loss")
plt.plot([x*len(train_dataloader) for x in range(len(val_adv_losses))], val_adv_losses)
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"train/unet_{SUFFIX}/losses.png")



tiramisu = Tiramisu(1, 1, MEASUREMENT_SIZE, DATA_SIZE, dims=1).to(DEVICE)
optimizer = torch.optim.Adam(tiramisu.parameters(), lr=TIRAMISU_LR, weight_decay=1e-3)
train_losses = []
val_losses = []
val_adv_losses = []
loss_acc = 0.0
loss_fn = lambda a,b: F.smooth_l1_loss(a, b)
for noisy_measurement, groundtruth in val_dataloader:
    noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
    # ---------------- Forward ----------------
    reconstruction = tiramisu(noisy_measurement.unsqueeze(1))[:,0]
    # ---------------- Evaluation -------------
    loss_acc += loss_fn(reconstruction, groundtruth).item()
val_losses.append(loss_acc/len(test_dataloader))
for _ in trange(TIRAMISU_EPOCHS):
    tiramisu.train()
    for noisy_measurement, groundtruth in train_dataloader:
        noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
        # ---------------- Training ---------------
        with torch.enable_grad():
            reconstruction = tiramisu(noisy_measurement.unsqueeze(1))[:,0]
            loss = loss_fn(reconstruction, groundtruth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ---------------- Evaluation -------------
        train_losses.append(loss_fn(reconstruction, groundtruth).item())
        torch.cuda.empty_cache()
    tiramisu.eval()
    loss_acc = 0.0
    adv_loss_acc = 0.0
    for noisy_measurement, groundtruth in val_dataloader:
        noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
        # ---------------- Forward ----------------
        reconstruction = tiramisu(noisy_measurement.unsqueeze(1))[:,0]
        # ---------------- Evaluation -------------
        loss_acc += loss_fn(reconstruction, groundtruth).item()
        # ------------ Adversarial attack ---------
        adversarial_measurement = noisy_measurement.clone()
        adversarial_measurement.requires_grad = True
        for _ in range(TIRAMISU_ADV_ITERATIONS):
            with torch.enable_grad():
                reconstruction = tiramisu(adversarial_measurement.unsqueeze(1))[:,0]
                loss = loss_fn(reconstruction, groundtruth)
                loss.backward()
            adversarial_measurement = adversarial_measurement + TIRAMISU_ADV_LR*torch.sign(adversarial_measurement.grad) # FGSM step
            adversarial_measurement = noisy_measurement + torch.clamp(adversarial_measurement-noisy_measurement, -TIRAMISU_ADV_EPSILON, TIRAMISU_ADV_EPSILON) # Projection
            adversarial_measurement.requires_grad = True
            torch.cuda.empty_cache()
        # ----- Adversarial attack evaluation -----
        adv_loss_acc += loss_fn(reconstruction, groundtruth).item()
    val_losses.append(loss_acc/len(test_dataloader))
    val_adv_losses.append(adv_loss_acc/len(test_dataloader))
save_dir = f"train/tiramisu_{SUFFIX}"
Path(save_dir).mkdir(parents=True, exist_ok=True)
torch.save(torch.tensor(train_losses), str(Path(save_dir, "train_losses.pth").resolve()))
torch.save(torch.tensor(val_losses), str(Path(save_dir, "val_losses.pth").resolve()))
torch.save(torch.tensor(val_adv_losses), str(Path(save_dir, "val_adv_losses.pth").resolve()))
torch.save(tiramisu.state_dict(), str(Path(save_dir, "weights.pth").resolve()))
plt.subplots(1,3,figsize=(21,5))
plt.subplot(1,3,1)
plt.title("Training loss")
plt.plot(train_losses)
plt.yscale("log")
plt.subplot(1,3,2)
plt.title("Validation loss")
plt.plot([x*len(train_dataloader) for x in range(len(val_losses))], val_losses)
plt.yscale("log")
plt.subplot(1,3,3)
plt.title("Adversarial validation loss")
plt.plot([x*len(train_dataloader) for x in range(len(val_adv_losses))], val_adv_losses)
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"train/tiramisu_{SUFFIX}/losses.png")



itnet = ItNet(1, 1, A, A_T, A_pinv, ITNET_INTERN_ITERATIONS, ITNET_INTERN_LR, dims=1).to(DEVICE)
optimizer = torch.optim.Adam(itnet.parameters(), lr=ITNET_LR, weight_decay=1e-3)
train_losses = []
val_losses = []
val_adv_losses = []
loss_acc = 0.0
loss_fn = lambda a,b: F.smooth_l1_loss(a, b)
for noisy_measurement, groundtruth in val_dataloader:
    noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
    # ---------------- Forward ----------------
    reconstruction = itnet(noisy_measurement.unsqueeze(1))[:,0]
    # ---------------- Evaluation -------------
    loss_acc += loss_fn(reconstruction, groundtruth).item()
val_losses.append(loss_acc/len(test_dataloader))
for _ in trange(ITNET_EPOCHS):
    itnet.train()
    for noisy_measurement, groundtruth in train_dataloader:
        noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
        # ---------------- Training ---------------
        with torch.enable_grad():
            reconstruction = itnet(noisy_measurement.unsqueeze(1))[:,0]
            loss = loss_fn(reconstruction, groundtruth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ---------------- Evaluation -------------
        train_losses.append(loss_fn(reconstruction, groundtruth).item())
        torch.cuda.empty_cache()
    itnet.eval()
    loss_acc = 0.0
    adv_loss_acc = 0.0
    for noisy_measurement, groundtruth in val_dataloader:
        noisy_measurement, groundtruth = noisy_measurement.to(DEVICE), groundtruth.to(DEVICE)
        # ---------------- Forward ----------------
        reconstruction = itnet(noisy_measurement.unsqueeze(1))[:,0]
        # ---------------- Evaluation -------------
        loss_acc += loss_fn(reconstruction, groundtruth).item()
        # ------------ Adversarial attack ---------
        adversarial_measurement = noisy_measurement.clone()
        adversarial_measurement.requires_grad = True
        for _ in range(ITNET_ADV_ITERATIONS):
            with torch.enable_grad():
                reconstruction = itnet(adversarial_measurement.unsqueeze(1))[:,0]
                loss = loss_fn(reconstruction, groundtruth)
                loss.backward()
            adversarial_measurement = adversarial_measurement + ITNET_ADV_LR*torch.sign(adversarial_measurement.grad) # FGSM step
            adversarial_measurement = noisy_measurement + torch.clamp(adversarial_measurement-noisy_measurement, -ITNET_ADV_EPSILON, ITNET_ADV_EPSILON) # Projection
            adversarial_measurement.requires_grad = True
            torch.cuda.empty_cache()
        # ----- Adversarial attack evaluation -----
        adv_loss_acc += loss_fn(reconstruction, groundtruth).item()
    val_losses.append(loss_acc/len(test_dataloader))
    val_adv_losses.append(adv_loss_acc/len(test_dataloader))
save_dir = f"train/itnet_{SUFFIX}"
Path(save_dir).mkdir(parents=True, exist_ok=True)
torch.save(torch.tensor(train_losses), str(Path(save_dir, "train_losses.pth").resolve()))
torch.save(torch.tensor(val_losses), str(Path(save_dir, "val_losses.pth").resolve()))
torch.save(torch.tensor(val_adv_losses), str(Path(save_dir, "val_adv_losses.pth").resolve()))
torch.save(itnet.state_dict(), str(Path(save_dir, "weights.pth").resolve()))
plt.subplots(1,3,figsize=(21,5))
plt.subplot(1,3,1)
plt.title("Training loss")
plt.plot(train_losses)
plt.yscale("log")
plt.subplot(1,3,2)
plt.title("Validation loss")
plt.plot([x*len(train_dataloader) for x in range(len(val_losses))], val_losses)
plt.yscale("log")
plt.subplot(1,3,3)
plt.title("Adversarial validation loss")
plt.plot([x*len(train_dataloader) for x in range(len(val_adv_losses))], val_adv_losses)
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"train/itnet_{SUFFIX}/losses.png")
print(f"Loss: {loss_acc/len(test_dataloader):7.5f}")
print(f"MSE: {mse_acc/len(test_dataloader):7.5f}")