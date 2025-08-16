"""
    针对MSTAR/SAMPLE数据集
"""
import random
# from deap import base, creator, tools, algorithms
import torch
import matplotlib.pyplot as plt
import asc_model as asc
from loss import calculateLoss_new_window, calculateLosstest, calculateLoss, calculateLoss_new
import time
import scipy
import scipy.io as sio
import numpy as np
import hdf5storage
import os
import glob


def estimate_scattering_centers1(init_params, s_actual, s_actual_1, lens, fc, B, device, num_epoch, aerfa_values,
                                lr=0.001):
    "基于梯度优化的ASC提取"

    # 初始化参数
    x1 = torch.nn.Parameter(torch.tensor(init_params[0], device=device))
    y1 = torch.nn.Parameter(torch.tensor(init_params[1], device=device))
    fai1 = torch.nn.Parameter(torch.tensor(float(init_params[2]), device=device))
    L1 = torch.nn.Parameter(torch.tensor(init_params[3], device=device))
    A1 = torch.nn.Parameter(torch.tensor(init_params[4], device=device))
    aerfa1 = torch.nn.Parameter(aerfa_values[torch.randint(0, 5, (1,))])



    optimizer_params_xy = torch.optim.AdamW([x1, y1, ], lr=0.01, weight_decay=0.01)
    optimizer_params_l = torch.optim.AdamW([fai1, L1, ], lr=0.01, weight_decay=0.01)
    optimizer_all = torch.optim.AdamW([x1, y1, fai1, L1, ], lr=0.1, weight_decay=0.01)


    a = 10
    b = 20

    
    losses = []

    for epoch in range(num_epoch):
        # 生成预测回波
        s_predicted = asc.generate_echo_torch(x1, y1, aerfa1, fai1, L1, A1, lens, fc, B, device)

        if epoch < a:
            # 估计x,y（频域初始化）
            loss = calculateLoss_new_window(s_predicted, s_actual, device)
            # loss = calculateLoss_new(s_predicted, s_actual)
            optimizer = optimizer_params_xy
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        elif a <= epoch <= b:
            # 估计fai,L（频域初始化）
            # loss = calculateLoss_new_window(s_predicted, s_actual, device)
            loss = calculateLoss_new(s_predicted, s_actual)
            optimizer = optimizer_params_l
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        else :
            # 全部优化
            # loss = calculateLoss(s_predicted, s_actual)
            # loss = calculateLoss_new_window(s_predicted, s_actual, device)
            loss = calculateLoss_new(s_predicted, s_actual)
            optimizer = optimizer_all
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            dicttmp = asc.generate_echo_torch(x1, y1, aerfa1, fai1, L1, 1.0,
                                              lens, fc, B, device).reshape(-1, 1)  # 预测回波
            self_corr = torch.mm(dicttmp.conj().t(), dicttmp)  # 计算自相关矩阵
            self_corr_inv = 1 / self_corr  # 计算自相关矩阵的逆，由于只有一个元素，可以直接除以该元素
            projection = torch.mm(dicttmp.conj().t().to(s_actual_1.dtype), s_actual.reshape(-1, 1))  # 计算投影
            Coef = torch.mm(self_corr_inv.to(s_actual_1.dtype), projection)  # 计算系数 Coef.to(torch.complex128)
            A1 = Coef



            # 限制参数的范围
            # with torch.no_grad():
            #     x1.clamp_(-8.5, 8)
            #     y1.clamp_(-8, 8)
            #     fai1.clamp_(-3, 3)
            #     L1.clamp_(0.0, 2.0)
            #     A1.clamp_(0.0, 50)

            losses.append(loss.item())



    best_aerfa_loss = float('inf')  # 枚举法估计aerfa
    for aerfa in aerfa_values:
        # 根据最小二乘法重新估计A A此时为复数
        dicttmp = asc.generate_echo_torch(x1, y1, aerfa, fai1, L1, 1.0,
                                          lens, fc, B, device).reshape(-1, 1)  # 预测回波
        self_corr = torch.mm(dicttmp.conj().t(), dicttmp)  # 计算自相关矩阵
        self_corr_inv = 1 / self_corr  # 计算自相关矩阵的逆，由于只有一个元素，可以直接除以该元素
        projection = torch.mm(dicttmp.conj().t().to(s_actual_1.dtype), s_actual.reshape(-1, 1))  # 计算投影
        Coef = torch.mm(self_corr_inv.to(s_actual_1.dtype), projection)  # 计算系数 Coef.to(torch.complex128)
        A_complex = Coef


        s_predicted = asc.generate_echo_torch(x1, y1, aerfa, fai1, L1, A_complex, lens, fc, B, device)
        current_loss = calculateLosstest(s_predicted, s_actual_1)
        # current_loss = calculateLoss_new(s_predicted, s_actual_1)


        if current_loss < best_aerfa_loss:
            best_aerfa_loss = current_loss
            aerfa1 = aerfa
            A_best = A_complex


    saved_params = {
        'x1': x1.item(),
        'y1': y1.item(),
        'aerfa1': aerfa1.item(),
        'fai1': fai1.item(),
        'L1': torch.abs(L1).item(),

        'A1': A_best.item(),
    }

    return saved_params, losses




def compute_residual_echo(K, params_list, s_actual_1, lens, fc, B, device):
    "减掉前K个估计出来的回波和"
    # 如果参数列表为空或K为0，则直接返回原始回波
    if not params_list or K == 0:
        return s_actual_1.clone()

    # 确保参数列表中至少有K个参数字典
    assert len(params_list) >= K, "参数列表中的目标数量少于K"

    # 从原始回波中减去前K个目标的回波
    s_residual = s_actual_1.clone()
    for i in range(K):
        target_params = params_list[i]
        s_target = asc.generate_echo_torch(
            torch.tensor(target_params['x1'], device=device),
            torch.tensor(target_params['y1'], device=device),
            torch.tensor(target_params['aerfa1'], device=device),
            torch.tensor(target_params['fai1'], device=device),
            torch.tensor(target_params['L1'], device=device),
            torch.tensor(target_params['A1'], device=device),
            lens, fc, B, device
        )
        s_residual -= s_target

    return s_residual


def compute_residual_echo_except_k(K, params_list, s_actual_1, lens, fc, B, device):
    """
    生成除了第K组参数以外的所有其他目标的回波差。
    :param K: 要排除的目标的索引（基于0开始，即第一个目标是0）
    :param params_list: 包含所有目标参数的列表，每个目标的参数是一个字典
    :param s_actual_1: 原始完整的回波
    :param lens: 回波长度
    :param fc: 中心频率
    :param B: 带宽
    :param device: 设备类型，例如 'cuda' 或 'cpu'
    :return: 除了第K组参数指定的目标之外的所有其他目标的回波差
    """
    assert K < len(params_list), "K is out of the range of params_list"

    # 从原始回波中减去除第K个目标之外的所有其他目标的回波
    s_echoes_except_k = s_actual_1.clone()
    for i, target_params in enumerate(params_list):
        if i != K:  # 排除第K个目标
            s_target = asc.generate_echo_torch(
                torch.tensor(target_params['x1'], device=device),
                torch.tensor(target_params['y1'], device=device),
                torch.tensor(target_params['aerfa1'], device=device),
                torch.tensor(target_params['fai1'], device=device),
                torch.tensor(target_params['L1'], device=device),
                torch.tensor(target_params['A1'], device=device),
                lens, fc, B, device
            )
            s_echoes_except_k -= s_target

    return s_echoes_except_k

def compute_predict_echo(K, params_list, lens, fc, B, device):
    """
    生产K组预测参数所有的回波
    """
    # 确保参数列表中至少有K个参数字典
    assert len(params_list) >= K, "参数列表中的目标数量少于K"

    # 从原始回波中减去除第K个目标之外的所有其他目标的回波
    s_all_predicted = torch.zeros((lens, lens), dtype=torch.complex64, device=device)
    for i, target_params in enumerate(params_list):
        s_target = asc.generate_echo_torch(
            torch.tensor(target_params['x1'], device=device),
            torch.tensor(target_params['y1'], device=device),
            torch.tensor(target_params['aerfa1'], device=device),
            torch.tensor(target_params['fai1'], device=device),
            torch.tensor(target_params['L1'], device=device),
            torch.tensor(target_params['A1'], device=device),
            lens, fc, B, device
        )
        s_all_predicted += s_target

    return s_all_predicted



def process_single_file(file_path, save_result_path):
    # 单个mat文件的散射中心提取
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = 90
    num_targets = 70  # ASC 数量
    lens = 102
    fc = 9.6e9
    B = 591e6

    aerfa_values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device)
    data = hdf5storage.loadmat(file_path)
    s_actual = data['specRmZeroAndWin']  # 去窗去0后的回波矩阵
    s_actual = torch.from_numpy(s_actual).to(device)

    asc.ifft_and_image(s_actual, fc, B)

    directory = save_result_path

    if not os.path.exists(directory):
        os.makedirs(directory)
    # plt.savefig(directory + '/' + file_path[4:-4] + '.png', bbox_inches='tight', pad_inches=0)  # 存储原始sar图像
    plt.savefig(directory+'/'+file_path[4:-4]+'.png')  # 存储原始sar图像
    plt.close()
    plt.clf()
    s_actual_1 = s_actual.clone()

    # 存储所有目标的参数和损失
    all_params = []
    all_losses = []
    loss_val = [0]*num_targets
    start_time = time.time()
    for k in range(1, num_targets + 1):
        # 第k次迭代，减去k-1个回波
        s_actual = compute_residual_echo(k - 1, all_params, s_actual_1, lens, fc, B, device)  # 当前残差回波
        topk = 1
        init_params = asc.find_x_y_topk(s_actual, fc, 3.5026, B, lens, topk)  # 图像域初始化，获取当前最大值

        # 第k次迭代，估计第k个目标的参数
        new_params, new_loss = estimate_scattering_centers1(init_params, s_actual, s_actual_1, lens, fc, B, device,
                                                           num_epoch, aerfa_values, lr=0.01)
        # 当前epoch的loss
        loss_val[k-1] = new_loss[-1]



        all_params.append(new_params)
        all_losses.append(new_loss)


        # 每次更新都存参数到mat矩阵
        # 将字典列表转化为数组
        params_arr = [[d['x1'], d['y1'], d['aerfa1'], d['fai1'], d['L1'], d['A1']] for d in all_params]

        # 保存为MAT文件
        savepath = directory + '/' + file_path[4:-4] + '.mat'
        sio.savemat(savepath, {'params': np.array(params_arr)})

        # print(f'第{k}次结果保存完成!')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"散射中心提取时间为：{execution_time} 秒")

    s_all_predict = compute_predict_echo(num_targets, all_params, lens, fc, B, device)
    # 结果存储
    asc.ifft_and_image(s_all_predict, fc, B)
    plt.savefig(directory + '/' + file_path[4:-4] + '_ASC.png')
    plt.close()
    plt.clf()
    asc.ifft_and_image(s_actual_1 - s_all_predict, fc, B)
    plt.savefig(directory + '/' + file_path[4:-4] + '_error.png')
    plt.close()
    plt.clf()


    # 绘制当前估计流程的loss变化（批量处理请注释）
    plt.plot(loss_val)
    plt.show()
    print(loss_val[-1])
    dicttmp0 = s_actual_1.reshape(1, -1)  # 预测回波
    E_target = torch.sum(torch.abs(dicttmp0)**2)  # 能量
    print('回波能量：', E_target)
    dicttmp1 = (s_actual_1-s_all_predict).reshape(-1, 1)  # 预测回波
    E_red = torch.sum(torch.abs(dicttmp1)**2)  # 能量
    print('残差能量：', E_red)
    print(f'能量重构比为：{torch.abs(E_red) / torch.abs(E_target)}')


    pass


if __name__ == "__main__":

    directory = 'data'  # ASC提取数据文件夹路径，mat文件
    saveresult_path = 'result' # ASC提取结果存储路径


    total_files = len(glob.glob(f"{directory}/*.mat"))

    for index, mat_file in enumerate(glob.glob(f"{directory}/*.mat"), start=1):

        progress = f"{index}/{total_files}"
        print(f"正在提取第 {progress} 个目标的散射中心")

        start_time = time.time()
        process_single_file(mat_file, saveresult_path)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"第 {progress} 个文件处理完成，耗时: {execution_time:.4f} 秒")














