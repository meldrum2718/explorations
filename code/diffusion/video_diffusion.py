import os
import time
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from unet import TimeConditionalUNet

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device', device)


verbose = False
def log(*args):
    if verbose:
        print(*args)


######## utils ########


def inspect(label, im):
    """ Print some basic image stats."""
    if im is None:
      return
    log()
    log(label + ':')
    log('shape:', im.shape)
    log('dtype:', im.dtype)
    log('max:', torch.max(im))
    log('min:', torch.min(im))
    if im.dtype == torch.float32:
      log('mean:', torch.mean(im))
      log('std:', torch.std(im))
    log()

def ctime_as_fname():
    """ Return time.ctime() formatted well for a file name."""
    return  time.ctime().replace(' ', '_').replace(':', '.')

def animate_frames(frames: torch.Tensor, ax=None, cmap=None) -> FuncAnimation:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(frames[0], cmap=cmap)

    def draw_func(t):
        im.set_data(frames[t % len(frames)])
        fig.canvas.draw()
        ax.set_title(str(t % len(frames)))
        return [im]
    ani = FuncAnimation(fig=fig, func=draw_func, cache_frame_data=False)
    return ani

def load_video_frames_from_file(filename: str, image_shape: Optional[tuple] = None, start_pts=0, end_pts=None) -> torch.Tensor:
    """
    Returns:
        frames: torch.Tensor of video frames
    """
    frames = torchvision.io.read_video(filename, pts_unit='sec', output_format='TCHW', start_pts=start_pts, end_pts=end_pts)[0] / 255.0
    if image_shape is not None:
        frames = F.interpolate(
            frames,
            size=image_shape,
            mode='bilinear',
            antialias=True,
        )
    return frames


class AutoregressiveFrames(Dataset):
    def __init__(self, frames: torch.Tensor, context_length: int):
        self.frames = frames
        self.context_length = context_length
        self.sequence_length = context_length + 1
        self.T, self.C, self.H, self.W = frames.shape

    def __len__(self):
        return max(0, self.T - self.context_length) # lets forget about any sort of padding for now. always assume theres some context we've seen..

    def __getitem__(self, idx):
        if idx + self.sequence_length > self.T:
            raise IndexError(f"Index {idx} + sequence length {self.sequence_length} exceeds number of frames {self.T}")
        sub_tensor = self.frames[idx:idx+self.context_length+1]
        sub_tensor = sub_tensor.reshape(self.sequence_length, self.C, self.H, self.W)
        c, x = sub_tensor[:-1], sub_tensor[-1]
        return c, x



########## ##########

class DDPM(nn.Module):
    def __init__(
        self,
        unet: TimeConditionalUNet,
        betas: tuple[float, float] = (1e-4, 0.02),
        num_ts: int = 300,
    ):
        super().__init__()
        
        self.T, self.C, self.H, self.W = unet.T, unet.C, unet.H, unet.W

        self.unet = unet
        self.num_ts = num_ts
        self.schedule = DDPM.get_schedule(betas[0], betas[1], num_ts)

        for k, v in self.schedule.items():
            self.register_buffer(k, v, persistent=False)

    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ Adapted from Algorithm 1 of the DDPM paper.

        Args:
            x: (B, C, H, W) input tensor
            c: (B, T, C, H, W) conditioning tensor

        Returns:
            (,) diffusion loss.
        """
        self.unet.train()

        B, T, C, H, W = x.size(0), self.T, self.C, self.H, self.W
        
        assert x.shape == (B, C, H, W)
        assert c.shape == (B, T-1, C, H, W)

        t = torch.randint(low=0, high=self.num_ts, size=(B,), device=device)

        noise = torch.randn_like(x, device=device)

        ab = self.schedule['alpha_bars']
        x_t = (
            torch.sqrt(ab[t]) * x
            + torch.sqrt(1 - ab[t]) * noise
        )
        # x_t = torch.cat((x_t, c), dim=1)

        t = t.to(torch.float32) / self.num_ts  ## note t.unsqueeze(-1) for the FCBlocks
        noise_pred = self.unet(x=x_t, c=c, t=t)
        loss = F.mse_loss(noise, noise_pred)

        return loss
    

    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        seed: int = 0,
        return_traj=False
    ) -> torch.Tensor:
        """ Adapted from Algorithm 2 of the DDPM paper
    
        Args:
            c: (B, T-1, C, H, W) conditioning tensor
            seed: int, random seed.
    
        Returns:
            (B, C, H, W) final sample.
        """
        self.unet.eval()
        torch.manual_seed(seed)
        
        B, T, C, H, W = c.shape[0], self.T, self.C, self.H, self.W

        assert c.shape == (B, T-1, C, H, W)
        
        x_t = torch.randn(B, C, H, W, device=device)

        traj = [x_t]
        for t_scalar in torch.arange(self.num_ts - 1, 0, -1, device=device):
            torch.manual_seed(seed)
            if t_scalar > 1:
                z = torch.randn_like(x_t, device=device)
            else:
                z = torch.zeros_like(x_t, device=device)
            t = torch.ones(B, dtype=torch.int, device=device) * t_scalar
    
            noise_pred = self.unet(x=x_t, c=c, t=t.to(torch.float32)/self.num_ts)
    
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
        if return_traj:
            return torch.stack(traj, dim=1)
        return traj[-1]
    

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

def train(model,
          dataset,
          channels,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gamma = 0.1 ** (1.0 / num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    train_losses = []
    if load_checkpoint_fname is not None:
        checkpoint = torch.load(load_checkpoint_fname, weights_only=True)
        train_losses = checkpoint['train_losses']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    log('len dataloader', len(dataloader))

    for epoch in range(num_epochs):
        epoch_loss = 0
        for (c, x) in tqdm(dataloader):
            log('---------- in traininng:')
            inspect('c', c)
            inspect('x', x)


            optimizer.zero_grad()
            loss = model(x=x.to(device), c=c.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            epoch_loss += loss.item()
        average_loss = epoch_loss / len(dataloader)
        log(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

        if lr_scheduler is not None:
            lr_scheduler.step()

    if save_checkpoint_fname is not None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'train_losses': train_losses,
        }
        torch.save(checkpoint, save_checkpoint_fname)

    plt.plot(train_losses, label='Training Loss')
    plt.yscale('log')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid('true', which='both')
    plt.title('Training Loss Over Time')
    plt.legend()

def flatten_video_across_batch_dim(traj, nrows, ncols):
    """ Args:
            traj: (B, T, C, H, W) -- diffusion trajectory
            nrows, ncols: int, such that nrows*ncols=B
        Returns:
            frames: (T, C, nrows*H, ncols*W)
    """
    B, T, C, H, W = traj.shape
    assert nrows * ncols == B
    traj = (torch.clip(traj, 0, 1) * 255).byte().cpu()
    traj = traj.reshape(nrows, ncols, T, H, W, C).permute(2, 0, 3, 1, 4, 5).reshape(T, nrows * H, ncols * W, C)
    return traj

def display_ddpm_traj(ddpm, c, nrows, ncols, seed=11):
    traj = ddpm.sample(c=c, seed=seed, return_traj=True)
    traj = torch.clip(traj, 0, 1).detach().cpu()
    frames = flatten_video_across_batch_dim(traj, nrows=nrows, ncols=ncols)
    cmap = 'grey' if (ddpm.C == 1) else None
    ani = animate_frames(frames, cmap=cmap)
    return ani


def generate_video(ddpm, context, n_steps, seed=11):
    """
    Args:
        context: (context_length, C, H, W)
    Returns:
        ani: FuncAnimation
        frames: (T, C, H, W) (where T = context_length + n_steps)
    """
    context_length, C, H, W = context.shape
    frames = [frame.unsqueeze(0) for frame in context]
    with torch.no_grad():
        for i in range(n_steps):

            inspect('frames[0]', frames[0])
            context = torch.stack(frames[-context_length:], dim=1)
            frame = ddpm.sample(context, seed=seed, return_traj=False)
            frames.append(frame.clip(0, 1))
    frames = torch.stack(frames, dim=1).cpu().squeeze(0) # (T, C, H, W)
    inspect('frames', frames)
    frames = frames.permute(0, 2, 3, 1) # (T, H, W, C)
    frames = torch.clip(frames, 0, 1)
    ani = animate_frames(frames)
    return ani, frames








    

def main():
    num_epochs = 1
    batch_size = 64
    filename = 'data/mandelbrot_zoom.mp4'

    H = 16
    W = 16
    C = 3
    context_length = 7
    T = context_length + 1
    D = 64 # hidden dim like in unet

    num_ts = 300 # ddpm traj length
    betas = (1e-4, 0.09) # ddpm schedule params

    frames = load_video_frames_from_file(filename, image_shape=(H, W)).to(device)
    train_set = AutoregressiveFrames(frames, context_length=context_length)
    print('here0')

    # log()
    # log('------- looking at dataset ------------')
    # inspect('frames', frames)
    # log('len frames', len(frames))
    # log('len train set', len(train_set))
    # log('train_set[0][0]', train_set[0][0])
    # log('train_set[0][1]', train_set[0][1])
    # log('---------------------------------------')
    # log()

    # dataloader = DataLoader(
    #     dataset=train_set,
    #     batch_size=1,
    #     shuffle=True
    #     sampler=
    # )

    # subset = torch.utils.data.Subset(train_set, list(range(100)))

    print('here1')
    model = TimeConditionalUNet(
        C=C,
        H=H,
        W=W,
        T=T,
        D=D,
    ).to(device)

    model = DDPM(
        unet=model,
        betas=betas,
        num_ts=num_ts,
    ).to(device)

    train(
        model,
        dataset=train_set,
        channels=C,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    print('here3')


    nrows = 2
    ncols = 2
    num_samples = nrows * ncols
    c = torch.randn(num_samples, context_length, C, H, W).to(device)
    ani = display_ddpm_traj(model, c, nrows, ncols)
    context = train_set[0][0]
    inspect('context:::::::', context)
    ani2 = generate_video(model, train_set[0][0], n_steps=10)

    plt.show()

if __name__ == '__main__':
    main()
