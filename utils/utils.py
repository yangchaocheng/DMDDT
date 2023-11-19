import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
import torch
import random
from sklearn.metrics import mean_squared_error


# def adjust_mask_based_on_confidence(y_true, y_pred, threshold=0):
#     # 计算均方误差
#     criterion = nn.MSELoss(reduction='none')
#     mse = criterion(y_true, y_pred)
#
#     # 根据均方误差计算置信度分数
#     confidence_score = 1.0 / (1.0 + mse)
#
#     # threshold需要动态，可根据计算出来的confidence，然后取前90%设为True
#     # 获取每个batch和dim中的最高90%置信度分数的阈值
#     batch_size, dim, seq_length = confidence_score.shape
#
#     # 将置信度分数重新形状为一维数组
#     flat_scores = confidence_score.reshape(-1, seq_length)
#
#     # 对每个batch和dim中的分数排序
#     sorted_scores, _ = torch.sort(flat_scores, descending=True, dim=1)
#
#     # 计算每个batch和dim的阈值，取最高90%的分数
#     threshold_index = int(0.95 * seq_length)
#     threshold_values = sorted_scores[:, threshold_index].view(batch_size, dim, 1)
#
#     # 使用阈值生成掩码
#     mask = confidence_score > threshold_values
#     # 在第三个维度上设置邻居，但不超出原始维度大小
#     expanded_mask = mask.clone()
#     mask_scale = 10
#     for i in range(1, mask_scale + 1):
#         expanded_mask[:, :, i:] &= mask[:, :, :-i]
#         expanded_mask[:, :, :-i] &= mask[:, :, i:]
#
#     return expanded_mask


def adjust_mask_based_on_confidence(y_true, y_pred, threshold=0, gradient_threshold=0.1):
    criterion = nn.MSELoss(reduction='none')
    mse = criterion(y_true, y_pred)

    # 根据均方误差计算置信度分数
    confidence_score = 1.0 / (1.0 + mse)

    # threshold需要动态，可根据计算出来的confidence，然后取前90%设为True
    batch_size, dim, seq_length = confidence_score.shape

    # 将置信度分数重新形状为一维数组
    flat_scores = confidence_score.reshape(-1, seq_length)

    # 对每个batch和dim中的分数排序
    sorted_scores, _ = torch.sort(flat_scores, descending=True, dim=1)

    # 计算每个batch和dim的阈值，取最高90%的分数
    threshold_index = int(0.95 * seq_length)
    threshold_values = sorted_scores[:, threshold_index].view(batch_size, dim, 1)

    # 使用阈值生成掩码
    mask = confidence_score > threshold_values

    # 计算梯度
    gradients = torch.autograd.grad(confidence_score.sum(), y_pred, create_graph=True)[0]

    # 根据每个点的梯度幅值来动态确定 mask_scale
    dynamic_mask_scale = torch.abs(gradients) / gradient_threshold
    dynamic_mask_scale = dynamic_mask_scale.clamp_min(1).int()  # 最小值为1，确保至少为1

    # 计算需要更新的位置索引
    update_indices = torch.nonzero(~mask, as_tuple=False)

    # 获取需要更新的值
    v = dynamic_mask_scale[update_indices[:, 0], update_indices[:, 1], update_indices[:, 2]]

    # 将前 v 个节点设置为 False
    expanded_mask = mask.clone()

    # Broadcasting to update the mask without a for loop
    start_idx = torch.clamp(update_indices[:, 2] - v, min=0).long()
    end_idx = torch.min(torch.tensor(seq_length), update_indices[:, 2] + 1 + v).long()

    # Create index tensors using torch.arange
    index_range = torch.arange(seq_length, device=mask.device).view(1, 1, -1)

    # Update the mask using element-wise comparison
    mask_update = ~((index_range >= start_idx.view(-1, 1, 1)) & (index_range < end_idx.view(-1, 1, 1)))
    unique_indices, inverse_indices = torch.unique(update_indices[:, :2], return_inverse=True, dim=0)

    # 初始化结果张量
    result_tensor = torch.zeros((batch_size, dim, seq_length), dtype=mask_update.dtype, device=mask.device)

    # 遍历每个唯一的索引组合
    for i in range(unique_indices.shape[0]):
        # 找到在 update_indices 中对应的索引
        indices = torch.nonzero((update_indices[:, :2] == unique_indices[i]).all(dim=1)).squeeze()

        # 使用这些索引在 a 中取值并相乘
        product = torch.prod(mask_update[indices], dim=0)

        # 将结果存储到对应位置
        result_tensor[unique_indices[i, 0], unique_indices[i, 1], :] = product

    expanded_mask = result_tensor

    return expanded_mask

# def adjust_mask_based_on_confidence(y_true, y_pred, threshold=0, gradient_threshold=0.1):
#     criterion = nn.MSELoss(reduction='none')
#     mse = criterion(y_true, y_pred)
#
#     # 根据均方误差计算置信度分数
#     confidence_score = 1.0 / (1.0 + mse)
#
#     # threshold需要动态，可根据计算出来的confidence，然后取前90%设为True
#     batch_size, dim, seq_length = confidence_score.shape
#
#     # 将置信度分数重新形状为一维数组
#     flat_scores = confidence_score.reshape(-1, seq_length)
#
#     # 对每个batch和dim中的分数排序
#     sorted_scores, _ = torch.sort(flat_scores, descending=True, dim=1)
#
#     # 计算每个batch和dim的阈值，取最高90%的分数
#     threshold_index = int(0.95 * seq_length)
#     threshold_values = sorted_scores[:, threshold_index].view(batch_size, dim, 1)
#
#     # 使用阈值生成掩码
#     mask = confidence_score > threshold_values
#
#     # 计算梯度
#     gradients = torch.autograd.grad(confidence_score.sum(), y_pred, create_graph=True)[0]
#
#     # 根据每个点的梯度幅值来动态确定mask_scale
#     dynamic_mask_scale = torch.abs(gradients) / gradient_threshold
#     dynamic_mask_scale = dynamic_mask_scale.clamp_min(1).int()  # 最小值为1，确保至少为1
#
#     need_mask = ~mask * dynamic_mask_scale
#
#     # 在第三个维度上设置邻居，但不超出原始维度大小
#     expanded_mask = mask.clone()
#
#     # 生成需要更新的位置索引
#     update_indices = torch.nonzero(~mask, as_tuple=False)
#
#     # 获取需要更新的节点的值
#     v_values = need_mask[update_indices[:, 0], update_indices[:, 1], update_indices[:, 2]]
#
#     # 计算前 v 个节点和后 v 个节点的范围
#     zero_tensor = torch.tensor(0, device=mask.device)
#     start_indices = torch.maximum(zero_tensor, update_indices[:, 2] - v_values)
#     max_length = torch.tensor(seq_length, device=mask.device)
#     end_indices = torch.minimum(max_length, update_indices[:, 2] + v_values)
#
#     # 生成对应的索引，并转换为 LongTensor
#     # row_indices = torch.arange(len(update_indices)).to(mask.device)
#
#     row_indices = torch.arange(len(update_indices)).to(torch.long).to(mask.device)
#     start_indices_indices = torch.stack([row_indices, start_indices], dim=1)
#     end_indices_indices = torch.stack([row_indices, update_indices[:, 2]], dim=1)
#
#     # 将相应范围的节点设置为 False
#     expanded_mask[start_indices_indices[:, 0], update_indices[:, 1],
#     start_indices_indices[:, 1]:end_indices_indices[:, 1]] = False
#     expanded_mask[end_indices_indices[:, 0], update_indices[:, 1], end_indices_indices[:, 1]:end_indices] = False
#
#     return expanded_mask

# def adjust_mask_based_on_confidence(y_true, y_pred, threshold=0, gradient_threshold=0.1):
#     criterion = nn.MSELoss(reduction='none')
#     mse = criterion(y_true, y_pred)
#
#     # 根据均方误差计算置信度分数
#     confidence_score = 1.0 / (1.0 + mse)
#
#     # threshold需要动态，可根据计算出来的confidence，然后取前90%设为True
#     batch_size, dim, seq_length = confidence_score.shape
#
#     # 将置信度分数重新形状为一维数组
#     flat_scores = confidence_score.reshape(-1, seq_length)
#
#     # 对每个batch和dim中的分数排序
#     sorted_scores, _ = torch.sort(flat_scores, descending=True, dim=1)
#
#     # 计算每个batch和dim的阈值，取最高90%的分数
#     threshold_index = int(0.95 * seq_length)
#     threshold_values = sorted_scores[:, threshold_index].view(batch_size, dim, 1)
#
#     # 使用阈值生成掩码
#     mask = confidence_score > threshold_values
#
#     # 计算梯度
#     gradients = torch.autograd.grad(confidence_score.sum(), y_pred, create_graph=True)[0]
#
#     # 根据每个点的梯度幅值来动态确定 mask_scale
#     dynamic_mask_scale = torch.abs(gradients) / gradient_threshold
#     dynamic_mask_scale = dynamic_mask_scale.clamp_min(1).int()  # 最小值为1，确保至少为1
#
#     # 计算需要更新的位置索引
#     update_indices = torch.nonzero(~mask, as_tuple=False)
#
#     # 获取需要更新的值
#     v = dynamic_mask_scale[update_indices[:, 0], update_indices[:, 1], update_indices[:, 2]]
#
#     # 将前 v 个节点设置为 False
#     expanded_mask = mask.clone()
#
#     # Broadcasting to update the mask without a for loop
#     start_idx = torch.clamp(update_indices[:, 2] - v, min=0).long()
#     end_idx = torch.min(torch.tensor(seq_length), update_indices[:, 2] + 1 + v).long()
#
#     # Create index tensors using torch.arange
#     index_range = torch.arange(seq_length, device=mask.device).view(1, 1, -1)
#
#     # Update the mask using element-wise comparison
#     mask_update = ~((index_range >= start_idx.view(-1, 1, 1)) & (index_range < end_idx.view(-1, 1, 1)))
#     unique_indices, inverse_indices = torch.unique(update_indices[:, :2], return_inverse=True, dim=0)
#
#     # 初始化结果张量
#     result_tensor = torch.zeros((batch_size, dim, seq_length), dtype=mask_update.dtype, device=mask.device)
#
#
#     result = torch.zeros((batch_size, dim, seq_length), dtype=mask_update.dtype, device=mask.device)
#
#     a = mask_update
#     result_shape = torch.Size([batch_size, dim, seq_length])
#     result = torch.ones(result_shape, device='cuda:0', dtype=a.dtype)  # 确保数据类型相同
#
#     # 提取前两列作为索引
#     indices = update_indices[:, :2]
#
#     # 计算每一组索引在结果张量中的一维索引
#     flat_indices = indices[:, 0] * result_shape[1] + indices[:, 1]
#
#     # 将对应位置的值相乘累积到结果张量中
#     result.view(-1, 100).scatter_(0, flat_indices.unsqueeze(1).expand(-1, 100), a.view(-1, 100))
#
#     return expanded_mask

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path

    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0
    device = diffusion_steps.device
    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(device)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            mean = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = mean + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask.float() + z * (1 - mask).float()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (transformed_X, cond, mask, diffusion_steps.view(B, 1),))  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z)


def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))  # length of series indexes
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask


def get_mask_mnr(sample, k):
    """Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    return mask
