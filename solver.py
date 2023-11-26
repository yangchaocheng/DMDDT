import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import lr_scheduler
import time
from datetime import datetime
from utils.utils import *
from model.DDTM import DDTM, Autoencoder
from data_factory.data_loader import get_loader_segment
from utils.utils import get_mask_bm, get_mask_rm, get_mask_mnr
from utils.utils import sampling, print_size, training_loss, calc_diffusion_hyperparams, std_normal


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=0, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.train_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, train_loss, val_loss, model, path):
        score = train_loss
        score2 = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(train_loss, val_loss, model, path)
        elif score2 > self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(train_loss, val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, train_loss, val_loss, model, path):
        if self.verbose:
            print(f'val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.train_loss_min = train_loss
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = DDTM(enc_in=self.input_c, c_out=self.output_c, mask_scale=self.mask_scale)
        self.Autoencoder = Autoencoder(input_dim=self.input_c, hidden_dim=self.input_c*2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer2 = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.to(self.device)
            self.Autoencoder.to(self.device)

    def vali(self, vali_loader):
        self.model.eval()
        loss2_list = []
        diffusion_hyperparams = calc_diffusion_hyperparams(self.T, self.beta_0, self.beta_T)
        for i, (input_data, _) in enumerate(vali_loader):
            if self.masking == 'rm':
                transposed_mask = get_mask_rm(input_data[0], self.masking_k)
            elif self.masking == 'mnr':
                transposed_mask = get_mask_mnr(input_data[0], self.masking_k)
            elif self.masking == 'bm':
                transposed_mask = get_mask_bm(input_data[0], self.masking_k)
            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(input_data.size()[0], 1, 1).float().to(self.device)
            loss_mask = ~mask.bool()
            input_data = input_data.permute(0, 2, 1)

            assert input_data.size() == mask.size() == loss_mask.size()

            input = input_data.float().to(self.device)

            T, Alpha_bar = diffusion_hyperparams["T"], diffusion_hyperparams["Alpha_bar"].to(self.device)
            series = input
            cond = input
            B, C, L = series.shape
            diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(self.device)
            z = std_normal(series.shape).to(self.device)
            if self.only_generate_missing == 1:
                z = series * mask.float() + z * (1 - mask).float()
            x_t = torch.sqrt(Alpha_bar[diffusion_steps]) * series + torch.sqrt(
                1 - Alpha_bar[diffusion_steps]) * z

            epsilon_theta = self.model((x_t, cond, mask, diffusion_steps.view(B, 1),))

            loss = self.criterion(epsilon_theta, z)

            loss2_list.append(loss.item())
        return np.average(loss2_list)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        diffusion_hyperparams = calc_diffusion_hyperparams(self.T, self.beta_0, self.beta_T)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = datetime.now()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                if self.masking == 'rm':
                    transposed_mask = get_mask_rm(input_data[0], self.masking_k)
                elif self.masking == 'mnr':
                    transposed_mask = get_mask_mnr(input_data[0], self.masking_k)
                elif self.masking == 'bm':
                    transposed_mask = get_mask_bm(input_data[0], self.masking_k)
                mask = transposed_mask.permute(1, 0)
                mask = mask.repeat(input_data.size()[0], 1, 1).float().to(self.device)
                loss_mask = ~mask.bool()
                input_data = input_data.permute(0, 2, 1)

                assert input_data.size() == mask.size() == loss_mask.size()

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                ##
                output = self.Autoencoder(input)
                loss2 = self.criterion(input, output)
                loss_mask = adjust_mask_based_on_confidence(input, output)

                loss2.backward()
                self.optimizer2.step()
                ##

                T, Alpha_bar = diffusion_hyperparams["T"], diffusion_hyperparams["Alpha_bar"].to(self.device)
                series = input*loss_mask
                cond = input*loss_mask
                B, C, L = series.shape
                diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(self.device)
                z = std_normal(series.shape).to(self.device)
                if self.only_generate_missing == 1:
                    z = series * mask.float() + z * (1 - mask).float()
                x_t = torch.sqrt(Alpha_bar[diffusion_steps]) * series + torch.sqrt(
                    1 - Alpha_bar[diffusion_steps]) * z

                epsilon_theta = self.model((x_t, cond, mask, diffusion_steps.view(B, 1),))

                loss = self.criterion(epsilon_theta, z)
                loss1_list.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, datetime.now() - epoch_time))
            train_loss = np.average(loss1_list)
            val_loss = self.vali(self.test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, val_loss))
            early_stopping(train_loss, val_loss, self.model, path)
            print('Updating learning rate to {}'.format(self.scheduler.get_last_lr()))
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')

        # (1) find the threshold
        start_time1 = datetime.now()
        attens_energy = []
        diffusion_hyperparams = calc_diffusion_hyperparams(self.T, self.beta_0, self.beta_T)
        for i, (input_data, labels) in enumerate(self.thre_loader):

            if self.masking == 'rm':
                transposed_mask = get_mask_rm(input_data[0], self.masking_k)
            elif self.masking == 'mnr':
                transposed_mask = get_mask_mnr(input_data[0], self.masking_k)
            elif self.masking == 'bm':
                transposed_mask = get_mask_bm(input_data[0], self.masking_k)
            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(input_data.size()[0], 1, 1).float().to(self.device)

            input_data = input_data.permute(0, 2, 1)
            input = input_data.float().to(self.device)

            T = diffusion_hyperparams["T"]
            Alpha = diffusion_hyperparams["Alpha"]
            Alpha_bar = diffusion_hyperparams["Alpha_bar"]
            Sigma = diffusion_hyperparams["Sigma"].to(self.device)

            size = (input.size(0), input.size(1), input.size(2))
            x = std_normal(size).to(self.device)
            with torch.no_grad():
                for t in range(T - 1, -1, -1):
                    if self.only_generate_missing == 1:
                        x = x * (1 - mask).float() + input * mask.float()
                    diffusion_steps = (t * torch.ones((size[0], 1))).to(self.device)
                    epsilon_theta = self.model((x, input, mask, diffusion_steps,))
                    mean = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                    if t > 0:
                        x = mean + Sigma[t] * std_normal(size).to(self.device)

            output1 = x
            loss = torch.mean(criterion(input, output1), dim=1)
            # output2 = self.Autoencoder(input)
            # loss2 = torch.mean(criterion(input, output2), dim=1)
            cri = loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        thre_energy = np.array(attens_energy)

        thresh = np.percentile(thre_energy, 100 - self.anomaly_ratio)
        end_time1 = datetime.now()
        time1 = end_time1 - start_time1
        print("spend time1:", time1)

        # (2) evaluation on the test set
        sum_f_score = []
        sum_accuracy = []
        sum_precision = []
        sum_recall = []
        for k in range(0, 100):
            test_labels = []
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.thre_loader):

                if self.masking == 'rm':
                    transposed_mask = get_mask_rm(input_data[0], self.masking_k)
                elif self.masking == 'mnr':
                    transposed_mask = get_mask_mnr(input_data[0], self.masking_k)
                elif self.masking == 'bm':
                    transposed_mask = get_mask_bm(input_data[0], self.masking_k)

                mask = transposed_mask.permute(1, 0)
                mask = mask.repeat(input_data.size()[0], 1, 1).float().to(self.device)

                input_data = input_data.permute(0, 2, 1)
                input = input_data.float().to(self.device)

                T = diffusion_hyperparams["T"]
                Alpha = diffusion_hyperparams["Alpha"]
                Alpha_bar = diffusion_hyperparams["Alpha_bar"]
                Sigma = diffusion_hyperparams["Sigma"].to(self.device)

                size = (input.size(0), input.size(1), input.size(2))
                x = std_normal(size).to(self.device)
                with torch.no_grad():
                    for t in range(T - 1, -1, -1):
                        if self.only_generate_missing == 1:
                            x = x * (1 - mask).float() + input * mask.float()
                        diffusion_steps = (t * torch.ones((size[0], 1))).to(self.device)
                        epsilon_theta = self.model((x, input, mask, diffusion_steps,))
                        mean = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(
                            Alpha[t])
                        if t > 0:
                            x = mean + Sigma[t] * std_normal(size).to(self.device)
                output1 = x
                loss = torch.mean(criterion(input, output1), dim=1)
                # output2 = self.Autoencoder(input)
                # loss2 = torch.mean(criterion(input, output2), dim=1)
                cri = loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)

            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            pred = (test_energy > thresh).astype(int)
            gt = test_labels.astype(int)

            # detection adjustment
            anomaly_state = False
            for i in range(len(gt)):
                if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    pred[i] = 1

            pred = np.array(pred)
            gt = np.array(gt)
            if k == 0:
                print("pred: ", pred.shape)
                print("gt:   ", gt.shape)

            from sklearn.metrics import precision_recall_fscore_support
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                                  average='binary')

            sum_f_score.append(f_score)
            sum_recall.append(recall)
            sum_precision.append(precision)
            sum_accuracy.append(accuracy)
        max_index = sum_f_score.index(np.max(sum_f_score))
        max_f_score = sum_f_score[max_index]
        max_recall = sum_recall[max_index]
        max_precision = sum_precision[max_index]
        max_accuracy = sum_accuracy[max_index]
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                max_accuracy, max_precision,
                max_recall, max_f_score))
