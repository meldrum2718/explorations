import os
from typing import List
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from utils import inspect, ctime_as_fname


device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def animate_frames(frames: List[np.ndarray], ax):
    """ each frame should have shape (H, W, C) """

    fig = ax.get_figure()

    ax.axis('off')
    fig.subplots_adjust(bottom=0.25)
    # cmap = 'grey'
    im = ax.imshow(frames[0])

    def draw_func(frame):
        im.set_data(frame)
        # fig.suptitle(str(t))
        return [im]

    ani = FuncAnimation(
        fig,
        func=draw_func,
        frames=frames,
        interval=100,
        save_count=1,
    )

    return ani


def pos_enc(coords: torch.Tensor, L: int = 10):
    """ coords.shape == (B, 2). """

    B = coords.shape[0]
    x, y = coords[:, 0], coords[:, 1]

    frequencies = 2 ** torch.arange(L, dtype=torch.float32, device=device) * torch.pi
    
    x_frequencies = torch.einsum('b,f -> bf', x, frequencies)
    y_frequencies = torch.einsum('b,f -> bf', y, frequencies)

    x_sin = torch.sin(x_frequencies)
    x_cos = torch.cos(x_frequencies)
    y_sin = torch.sin(y_frequencies)
    y_cos = torch.cos(y_frequencies)

    pe = torch.cat([coords, x_sin, x_cos, y_sin, y_cos], dim=-1)
    
    return pe


class MLP(nn.Module):
    def __init__(self, d_in, d_h, n_layers, L):
        super(MLP, self).__init__()
        layers = [nn.Linear(d_in, d_h), nn.ReLU()]
        for _ in range(n_layers):
            layers.append(nn.Linear(d_h, d_h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(normalized_shape=(d_h,)))

        layers.append(nn.Linear(d_h, 3))

        self.layers = nn.Sequential(*layers)
        self.L = L

    def forward(self, x):
        x = pos_enc(x, L=self.L)
        x = self.layers(x)
        x = F.sigmoid(x)
        return x


def generate_image(nf, resolution, batch_size=1024):
    H, W = resolution

    i = torch.linspace(0, 1, steps=H, device=device)
    j = torch.linspace(0, 1, steps=W, device=device)
    grid_i, grid_j = torch.meshgrid(i, j, indexing="ij") # (H, W)
    coords = torch.stack([grid_i, grid_j], dim=-1).reshape(-1, 2).to(device) # (H*W, 2)

    rgb_values = torch.zeros((H * W, 3), device=device)

    nf = nf.to(device)
    nf.eval()
    with torch.no_grad():
        for i in range(0, coords.shape[0], batch_size):
            batch_coords = coords[i:i + batch_size]
            rgb_values[i:i + batch_coords.shape[0]] = nf(batch_coords)

    image = rgb_values.reshape(H, W, 3).clip(0, 1)
    image = image.detach().cpu().numpy()
    return image



class ImageDataset(Dataset):
    def __init__(self, image, num_samples):
        super(ImageDataset, self).__init__()
        
        self.image = image

        self.height, self.width, _ = self.image.shape
        self.num_samples = num_samples

    def set_num_samples(self, ns):
        self.num_samples = ns

    def __len__(self):
        return 10000  # arbitrary large value

    def __getitem__(self, idx):
        i = torch.randint(0, self.height, (self.num_samples,))
        j = torch.randint(0, self.width, (self.num_samples,))
        
        coords = torch.stack([i, j], dim=1)#.float() # (num_samples, 2)
        coords = normalize_coords(coords, self.image.shape[0], self.image.shape[1])

        pixels = self.image[i, j, :] # (num_samples, 3)

        return coords, pixels


def get_psnr(mse):
    return 10 * torch.log10(1.0 / mse)


def load_checkpoint(model, optimizer, lr_scheduler, fname):
    checkpoint = torch.load(fname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

def save_checkpoint(model, optimizer, lr_scheduler, fname):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }
    torch.save(checkpoint, fname)


def train(model,
          image,
          optimizer,
          n_steps,
          batch_size,
    ):

    dataset = ImageDataset(image=image, num_samples=batch_size)
    # dataloader = DataLoader( dataset=dataset, batch_size=1, shuffle=True, collate_fn=lambda x: torch.stack(x, dim=0))

    train_losses = []

    for i in tqdm(range(n_steps)):
        coords, pixels = dataset[0]
        coords = coords.to(device)
        pixels = pixels.to(device)

        optimizer.zero_grad()
        preds = model(coords)
        loss = F.mse_loss(preds, pixels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return train_losses



def load_image(image_path):
    image = torch.Tensor((np.array(Image.open(image_path).convert('RGB')))).float() / 255.0
    return image


def normalize_coords(coords, H, W):
    """ expect coords: (B, 2) """
    coords = coords.float()
    coords[:, 0] = coords[:, 0] / H
    coords[:, 1] = coords[:, 1] / W
    return coords


def main():
    L = 10
    image_path = '../images/fox.jpg'

    n_training_steps = 300
    batch_size = 500

    d_h = 256
    n_layers = 3

    n_training_frames = 100

    pred_resolution = (256, 512)

    image = load_image(image_path).to(device) # (H, W, C)
    H, W, C = image.shape[0], image.shape[1], image.shape[2]

    model = MLP(
        d_in=2*(2*L + 1),
        d_h=d_h,
        n_layers=n_layers,
        L=L
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # training loop
    ns = n_training_steps // (n_training_frames - 1)

    frames = []
    train_losses = []
    for i in range(n_training_frames):
        frames.append(generate_image(model, resolution=pred_resolution))

        losses = train(
            model=model,
            image=image,
            optimizer=optimizer,
            batch_size=batch_size,
            n_steps=ns,
        )
        train_losses.extend(losses)

    fig, ax = plt.subplots()
    ani = animate_frames(frames, ax)

    if not os.path.exists('data'):
        os.mkdir('data')

    fname = f'L{L}_dh_{d_h}_nl{n_layers}_ns{n_training_steps}_bs{batch_size}.png'

    plt.savefig('data/' + 'preds_' + fname)


    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(train_losses, label='Training Loss')
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Loss')
    axs[0].grid('true', which='both')
    axs[0].set_title('Training Loss Over Time')

    psnrs = [get_psnr(l) for l in torch.Tensor(train_losses)]
    axs[1].plot(psnrs, label='Training PSNR')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('PSNR')
    axs[1].grid('true', which='both')
    axs[1].set_title('Training PSNR Over Time')

    plt.legend()

    plt.savefig('data/' + 'loss_' + fname)


    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(generate_image(model, resolution=pred_resolution))

    plt.show()

    # coords, pixels = dataset[0]
    # i, j = coords[:, 0], coords[:, 1]

    # inspect('coords', coords)

    # coords = normalize_coords(coords, H, W)
    # inspect('coords', coords)

    # inspect('pixels', pixels)

    # print('same:', torch.all(image[i, j] == pixels).item())

    # preds = mlp(coords)

    # inspect('preds', preds)

    pass

if __name__ == '__main__':
    main()
