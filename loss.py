import torch
from asc_model import generate_2d_taylor


def calculateLoss(s_predicted, s_actual):
    # L_LS
    diff = s_actual.reshape(-1,1) - s_predicted.reshape(-1,1)
    diff_squared = torch.norm(diff, p=2)
    loss = diff_squared
    return loss

def calculateLoss_new(s_predicted, s_actual):
    # L_NC
    s_predicted = s_predicted.reshape(-1, 1)
    s_actual = s_actual.reshape(-1, 1)
    s_predicted = s_predicted.to(torch.cfloat)
    s_actual = s_actual.to(torch.cfloat)
    loss = (s_predicted.conj().t()) @ s_predicted - 2 * torch.abs(s_predicted.conj().t() @ s_actual) + (s_actual.conj().t()) @ s_actual
    loss = torch.abs((loss))

    return loss

def calculateLoss_new_window(s_predicted, s_actual, device):
    # L_win
    win_2d = generate_2d_taylor(102, 102, nbar=2, sll=35)
    window = torch.from_numpy(win_2d).to(s_predicted.device)

    s_predicted = s_predicted * window
    s_actual = s_actual * window

    s_predicted = s_predicted.view(-1, 1)
    s_actual = s_actual.reshape(-1, 1)

    s_predicted = s_predicted.to(torch.cfloat)
    s_actual = s_actual.to(torch.cfloat)

    loss = (s_predicted.conj().t()) @ s_predicted - 2 * torch.abs(s_predicted.conj().t() @ s_actual) + (s_actual.conj().t()) @ s_actual

    loss = torch.abs((loss))

    return loss




def calculateLosstest(s_predicted, s_actual):
    diff = torch.fft.ifft2(s_predicted - s_actual)
    diff_squared = torch.abs(diff)**2
    loss = torch.sum(diff_squared)
    return loss





