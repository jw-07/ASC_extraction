import numpy as np
import matplotlib.pyplot as plt
import asc_model as asc
from loss import calculateLoss,  calculateLoss_new, calculateLoss_new_window
import torch


def normalize(data):
    """ normalize loss to 0-1 """
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lens = 101
fc = 10e9
B = 3000e6


s_actual = asc.generate_echo_torch(-0.6530, 0.4184, 1.0, 2.3, 1.8, 16.1852, lens, fc, B, device)


# x1_vals = np.arange(-1, -0.2, 0.002) #x
x1_vals = np.arange(-1.8, 2.2, 0.002) #y
# x1_vals = np.arange(1.85, 2.7, 0.002) #fai
# x1_vals = np.arange(1, 2.8, 0.002) #L

losses = []
losses1 = []
losses2 = []
losses3 = []

for x1 in x1_vals:
    # s_predicted = asc.generate_echo_torch(x1, 0.4184, 1.0, 2.3, 1.8, 16.1852, lens, fc, B,device) #x
    s_predicted = asc.generate_echo_torch(-0.6530, x1, 1.0, 2.3, 1.8, 16.1852, lens, fc, B, device) #y
    # s_predicted = asc.generate_echo_torch(-0.6530, 0.4184, 1.0, x1, 1.8, 16.1852, lens, fc, B, device) #fai
    # s_predicted = asc.generate_echo_torch(-0.6530, 0.4184, 1.0, 2.3, x1, 16.1852, lens, fc, B, device) #L


    curr_loss1 = calculateLoss_new(s_predicted, s_actual)  #  nc
    curr_loss = calculateLoss(s_predicted, s_actual)  # norm
    curr_loss3 = calculateLoss_new_window(s_predicted, s_actual, device)  # win

    losses.append(curr_loss.item())
    losses1.append(curr_loss1.item())
    losses3.append(curr_loss3.item())


plt.plot(x1_vals, losses, label='Losses (L2 Norm)')
plt.plot(x1_vals, losses1, label='Losses1 (NC)')
plt.plot(x1_vals, losses3, label='Losses3 (Win)')


plt.xlabel('x')
plt.ylabel('Loss')
plt.title('Loss vs x')
plt.legend()

plt.show()






normalized_losses = normalize(losses)
normalized_losses1 = normalize(losses1)
normalized_losses3 = normalize(losses3)


plt.plot(x1_vals, normalized_losses, label='Losses (L2 Norm)')
plt.plot(x1_vals, normalized_losses1, label='Losses1 (NC)')
plt.plot(x1_vals, normalized_losses3, label='Losses3 (Win)')




plt.xlabel('x')
plt.ylabel('Normalized Loss')
plt.title('Loss function')
plt.legend()

plt.show()







