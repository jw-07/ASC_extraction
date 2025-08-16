import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from scipy.signal.windows import taylor



def sind_torch(angle_degrees):
    angle_radians = angle_degrees * math.pi / 180
    return torch.sin(angle_radians)

def cosd_torch(angle_degrees):
    angle_radians = angle_degrees * math.pi / 180
    return torch.cos(angle_radians)




def generate_echo_torch(x, y, aerfa, fai, L, A, lens, fc, B, device):
    c = 3e8
    f = torch.linspace(fc - B / 2, fc + B / 2, lens, device=device)
    angle = torch.linspace(-1.7513, 1.7513, lens, device=device).view(-1, 1)  # MSTAR
    # angle = torch.linspace(-8.5, 8.5, lens, device=device).view(-1, 1)
    term1 = (1j * torch.ones(lens, 1, dtype=torch.cfloat, device=device) * f / fc) ** aerfa
    term2 = torch.exp((x * cosd_torch(angle) + y * sind_torch(angle)) * (-1j * 4 * math.pi / c * f))
    term3 = torch.sinc(sind_torch(angle - fai) * (2 * math.pi * f / c * L) / math.pi)

    echo = A * term1 * term2 * term3

    return echo

def ifft_and_image(echo, fc, B):

    win_2d = generate_2d_taylor(102, 102, nbar=4, sll=35)
    win_2d = torch.from_numpy(win_2d).to(echo.device)

    echo = echo * win_2d
    echo = echo
    time_domain_signal = torch.fft.ifft2(echo)
    shifted_signal = torch.fft.fftshift(time_domain_signal)


    abs_signal = torch.abs(shifted_signal).detach().cpu()

    plt.figure()
    # plt.imshow(abs_signal.numpy(), vmin=0, vmax=2.0)
    # plt.imshow(abs_signal.numpy(), cmap='hot',vmin=0, vmax=0.8)
    plt.imshow(abs_signal.numpy(), cmap='hot')
    # plt.imshow(abs_signal.numpy(), )
    # plt.colorbar(label='Amplitude')
    # plt.colorbar()
    plt.axis("off")
    # plt.title('Time Domain Imaging')
    # plt.show()




def generate_2d_taylor(Nx, Ny, nbar=4, sll=35):


    win_x = taylor(Nx, nbar=nbar, sll=sll)
    win_y = taylor(Ny, nbar=nbar, sll=sll)
    win_2d = np.outer(win_x, win_y)

    return win_2d






def find_x_y_topk(echo, fc, angel, B, lens, K):
    echo_ifft = torch.fft.ifft2(echo)
    echo_ifft = torch.fft.fftshift(echo_ifft)

    c = 3e8
    dx = c / (2 * B)  # range resolution
    angel_rand = angel / 180 * math.pi
    lamed = c / fc
    dy = lamed / (2 * angel_rand)  # angle resolution

    A = torch.abs(echo_ifft)

    A_topk, indices = torch.topk(A.view(-1), K)


    A_max = A_topk[K - 1]
    max_index = indices[K - 1]


    max_coords = np.unravel_index(max_index.cpu().numpy(), A.shape)
    x = (max_coords[1] - (lens / 2)) * dx
    y = (max_coords[0] - (lens / 2)) * dy

    return x, y, 0.01, 0.01, A_max.item()



