import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from unet import TimeConditionalUNet

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


######## utils
def inspect(label, im):
    """ Print some basic image stats."""
    if im is None:
      return
    print()
    print(label + ':')
    print('shape:', im.shape)
    print('dtype:', im.dtype)
    print('max:', torch.max(im))
    print('min:', torch.min(im))
    if im.dtype == torch.float32:
      print('mean:', torch.mean(im))
      print('std:', torch.std(im))
    print()

def ctime_as_fname():
    """ Return time.ctime() formatted well for a file name."""
    return  time.ctime().replace(' ', '_').replace(':', '.')

def get_video_frames(traj, nrows, ncols):
    """ Args:
            traj: (N, T, C, H, W)
        Returns:
            frames: list of(H*nrows, W*ncols, C) tensors
    """
    N, T, C, H, W = traj.shape
    assert nrows * ncols == N, 'N must be divisible by nrows'
    traj = (torch.clip(traj, 0, 1) * 255).byte().cpu().numpy()
    traj = traj.reshape(nrows, ncols, T, H, W, C).transpose(2, 0, 3, 1, 4, 5).reshape(T, nrows * H, ncols * W, C)
    frames = [frame for frame in traj]
    return frames
##########


class DDPM(nn.Module):
    def __init__(
        self,
        unet: TimeConditionalUNet,
        betas: tuple[float, float] = (1e-4, 0.02),
        num_ts: int = 300,
    ):
        super().__init__()
        
        self.C, self.H, self.W = unet.C, unet.H, unet.W
        self.unet = unet
        self.num_ts = num_ts
        self.schedule = DDPM.get_schedule(betas[0], betas[1], num_ts)

        for k, v in self.schedule.items():
            self.register_buffer(k, v, persistent=False)

    
    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """ Algorithm 1 of the DDPM paper.
        Args:
            x: (N, C, H, W) input tensor.

        Returns:
            (,) diffusion loss.
        """
        self.unet.train()
        
        N, C, H, W = x_0.shape
        t = torch.randint(low=0, high=self.num_ts, size=(N,), device=device)

        noise = torch.randn_like(x_0, device=device)
        ab = self.schedule['alpha_bars']
        x_t = (
            torch.sqrt(ab[t]) * x_0
            + torch.sqrt(1 - ab[t]) * noise
        )
        t = t.to(torch.float32) / self.num_ts  ## note t.unsqueeze(-1) for the FCBlocks
        noise_pred = self.unet(x=x_t, t=t)
        loss = F.mse_loss(noise, noise_pred)
        return loss
    

    @torch.inference_mode()
    def sample(
        self,
        seed: int = 0,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Algorithm 2 of the DDPM paper with classifier-free guidance.
    
        Args:
            seed: int, random seed.
    
        Returns:
            (N, C, H, W) final sample.
        """
        self.unet.eval()
        torch.manual_seed(seed)
        
        N, C, H, W = num_samples, self.C, self.H, self.W
        
        x_t = torch.randn(N, C, H, W, device=device)
        traj = [x_t]
        for t_scalar in torch.arange(self.num_ts - 1, 0, -1, device=device):
            torch.manual_seed(seed)
            if t_scalar > 1:
                z = torch.randn_like(x_t, device=device)
            else:
                z = torch.zeros_like(x_t, device=device)
            t = torch.ones(N, dtype=int, device=device) * t_scalar
    
            noise_pred = self.unet(x=x_t, t=t.to(torch.float32)/self.num_ts)
    
            a = self.schedule['alphas'].to(device)
            ab = self.schedule['alpha_bars'].to(device)
            b = self.schedule['betas'].to(device)
    
            clean_est = (x_t - torch.sqrt(1 - ab[t]) * noise_pred) / torch.sqrt(ab[t])
    
            a0 = torch.sqrt(ab[t-1]) * b[t]
            a1 = torch.sqrt(a[t]) * (1 - ab[t-1])
            x_t = (a0 * clean_est + a1 * x_t) / (1 - ab[t])
            x_t = x_t + torch.sqrt(b[t]) * z
            traj.append(x_t)
    
            seed += 1
        return torch.stack(traj, dim=1)
    

    @classmethod
    def get_schedule(cls, beta1: float, beta2: float, num_ts: int) -> dict:
        """Constants for DDPM training and sampling.
    
        Arguments:
            beta1: float, starting beta value.
            beta2: float, ending beta value.
            num_ts: int, number of timesteps.
    
        Returns:
            dict with keys:
                betas: linear schedule of betas from beta1 to beta2.
                alphas: 1 - betas.
                alpha_bars: cumulative product of alphas.
        """
        assert beta1 < beta2 < 1.0, "Expect beta1 < beta2 < 1.0."
        betas = torch.linspace(beta1, beta2, num_ts, device=device).reshape(num_ts, 1, 1, 1).to(torch.float32)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return {
            "betas": betas.to(device),
            "alphas": alphas.to(device),
            "alpha_bars": alpha_bars.to(device),
        }




def save_checkpoint(
    ddpm,
    optimizer,
    lr_scheduler,
    train_losses,
    path,
    use_ctime=False,
):
    checkpoint = {
        'ddpm_state_dict': ddpm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'train_losses': train_losses,
    }
    torch.save(checkpoint, path)



def train_ddpm(
    ddpm,
    dataset,
    num_epochs=1,
    batch_size=128,
    load_checkpoint_fname=None,
    save_checkpoint_fname=None,
):

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)
    gamma = 0.1 ** (1.0 / num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    train_losses = []
    if load_checkpoint_fname is not None:
        checkpoint = torch.load(checkpoint_fname, weights_only=True)
        train_losses = checkpoint['train_losses']
        ddpm.load_state_dict(checkpoint['ddpm_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x, _ in tqdm(dataloader):
            optimizer.zero_grad()
            loss = ddpm(x.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            epoch_loss += loss.item()
        average_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

        if lr_scheduler is not None:
            lr_scheduler.step()

    if save_checkpoint_fname is not None:
        save_checkpoint(
            ddpm=ddpm,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_losses=train_losses,
            path=save_checkpoint_fname,
        )

    plt.plot(train_losses, label='Training Loss')
    plt.yscale('log')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid('true', which='both')
    plt.title('Training Loss Over Time')
    plt.legend()


def main():
    num_hiddens = 16
    num_epochs = 1
    batch_size = 128

    train_set = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    print(train_set)
    x, _ = train_set[0]
    inspect('x', x)
    C, H, W = x.shape

    subset = torch.utils.data.Subset(train_set, list(range(10_000)))

    model = TimeConditionalUNet(
        in_channels=C,
        H=H,
        W=W,
        num_hiddens=num_hiddens,
    ).to(device)

    ddpm = DDPM(
        unet=model,
        betas=(1e-4, 0.09),
        num_ts=10,
    ).to(device)

    train_ddpm(
        ddpm,
        dataset=subset,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    nrows = 3
    ncols = 3
    num_samples = nrows * ncols

    traj = ddpm.sample(seed=11, num_samples=num_samples)
    traj = torch.clip(traj, 0, 1).detach().cpu()

    frames = get_video_frames(traj, nrows=nrows, ncols=ncols)

    cmap = 'grey' if (C == 1) else None

    fig, axs = plt.subplots(2)
    axs[0].imshow(frames[-1], cmap=cmap)
    im = axs[1].imshow(frames[0], cmap=cmap)
    def update(frame):
        im.set_data(frame)
        return im

    ani = FuncAnimation(
        fig,
        update,
        frames,
        save_count=1,
        interval=65
    )

    plt.show()

if __name__ == '__main__':
    main()
