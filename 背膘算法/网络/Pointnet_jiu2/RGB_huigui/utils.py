import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt





def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    optimizer.zero_grad()


    data_loader = tqdm(data_loader, file=sys.stdout)

    # all_labels_numpy = []
    # all_outputs_numpy = []

    for step, data in enumerate(data_loader):
        images, labels = data

        # images = torch.mean(images, dim=1, keepdim=True)   # 对图像进行处理，这里是取均值降维
        # label = label.reshape(-1, 1)  # 将标签重塑为列向量
        images = images.to(device)
        pred = model(images.to(torch.float32))    # 将图像数据转换为float32类型，并进行预测

        labels = labels.to(device)
        labels = labels.to(torch.float32)
        labels = labels.unsqueeze(1)      # 增加一个维度

        # label_numpy = labels.cpu().numpy()
        # pred_numpy = pred.cpu()
        # pred_numpy = pred_numpy.detach().numpy()
        # all_labels_numpy = np.append(all_labels_numpy, label_numpy)
        # all_outputs_numpy = np.append(all_outputs_numpy, pred_numpy)
        # R2 = r2_score(y_true=all_labels_numpy, y_pred=all_outputs_numpy)
        # MAE = mean_absolute_error(y_true=all_labels_numpy, y_pred=all_outputs_numpy)
        # MSE = mean_squared_error(y_true=all_labels_numpy, y_pred=all_outputs_numpy)

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()    # 累加损失，detach防止梯度被追踪

        # data_loader.desc = "[train epoch {}] loss: {:.3f}  R^2: {}  MAE: {} MSE:{}, ".format(epoch, accu_loss.item() / (step + 1), R2, MAE,MSE)
        data_loader.desc = "[train epoch {}] loss: {:.3f}  ".format(epoch, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss):    # 如果损失不是有限数
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)      # 退出程序

        optimizer.step()     # 更新模型参数
        optimizer.zero_grad()     # 清除梯度，为下一步迭代准备


    return accu_loss.item() / (step + 1)     # 返回平均损失


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.MSELoss()

    model.eval()


    accu_loss = torch.zeros(1).to(device)  # 累计损失

    # all_labels_numpy = []
    # all_outputs_numpy = []

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data

        # images = torch.mean(images, dim=1, keepdim=True)

        images = images.to(device)
        pred = model(images.to(torch.float32))

        labels = labels.to(device)
        labels = labels.to(torch.float32)
        labels = labels.unsqueeze(1)

        # label_numpy = labels.cpu().numpy()
        # pred_numpy = pred.cpu()
        # pred_numpy = pred_numpy.detach().numpy()
        # all_labels_numpy = np.append(all_labels_numpy, label_numpy)
        # all_outputs_numpy = np.append(all_outputs_numpy, pred_numpy)
        # R2 = r2_score(y_true=all_labels_numpy, y_pred=all_outputs_numpy)
        # MAE = mean_absolute_error(y_true=all_labels_numpy, y_pred=all_outputs_numpy)
        # MSE = mean_squared_error(y_true=all_labels_numpy, y_pred=all_outputs_numpy)


        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f} ".format(epoch+1, accu_loss.item() / (step + 1),)

    return accu_loss.item() / (step + 1)

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
