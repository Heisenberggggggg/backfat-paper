import torch
import numpy as np
from torch.utils.data import Dataset
import sys
import os
import pandas as pd
import argparse
from data_utils.dataloader import MyDataset
import importlib
import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))



def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--num_category', default=1, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')   #根据自己训练设置的
    parser.add_argument('--model', default='pointnet2_jiu_msg', help='model name [default: pointnet_cls]')
    # parser.add_argument('--log_dir', type=str, required=None, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    # parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def main(args):



    test_path = r'D:\data_point\48point\48_1.1\train'


    test_dataset = MyDataset(root=test_path, args=args, )

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = importlib.import_module(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = model.get_model(num_class=1, normal_channel=args.use_normals).to(device)
    classifier.apply(inplace_relu)  #1

    #加载模型
    #classifier.load_state_dict(torch.load(r"F:\Pointnet_jiu\weight\best_model2024_07_11_09_18.pth"),False)  #这里需要再考虑
    # classifier.load_state_dict(torch.load(r"D:\Pointnet_jiu2\log\regression_jiu\2025-04-09_21-55\checkpoints\best_model_jiu.pth"))

    checkpoint = torch.load(r"D:\Pointnet_jiu2\log\regression_jiu\2025-04-09_21-55\checkpoints\best_model_jiu.pth")
    classifier.load_state_dict(checkpoint['model_state_dict'])  # 只加载模型状态字典
    #获取预测结果


    test_classifier = classifier.eval()

    # with torch.no_grad():



    all_labels_numpy = []
    all_outputs_numpy = []
    # for j, (test_points, test_target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
    for j, (test_points, test_target) in enumerate(testDataLoader):

        test_points, test_target = test_points.to(device), test_target.to(device)  # 1
        test_points = test_points.transpose(2, 1)
        test_pred, test_trans_feat = test_classifier(test_points)

        test_target = test_target.unsqueeze(1)

        test_target = test_target.to(torch.float32)
        label_numpy = test_target.cpu().numpy()
        all_labels_numpy = np.append(all_labels_numpy, label_numpy)

        out_numpy = test_pred.cpu()
        out_numpy = out_numpy.detach().numpy()
        all_outputs_numpy = np.append(all_outputs_numpy, out_numpy)





# ------------------------------打印标签值------------------------------------------------#
    rows, leftover = divmod(all_labels_numpy.size, 90)
    reshaped_data = all_labels_numpy[:rows * 90].reshape(-1, 90)
    if leftover:
        # 把剩余的数值加入到一个新列中
        reshaped_data = np.column_stack((reshaped_data, all_labels_numpy[rows * 90:]))
    # 对每一行进行排序
    reshaped_data = np.sort(reshaped_data, axis=1)
    # 转换为DataFrame
    df = pd.DataFrame(reshaped_data)
    df = df.T

    # 写入CSV文件
    df.to_csv('labels_90.csv', index=False, header=False)


#------------------------------打印预测值------------------------------------------------#

    rows, leftover = divmod(all_outputs_numpy.size, 90)
    reshaped_data = all_outputs_numpy[:rows * 90].reshape(-1, 90)
    if leftover:
        # 把剩余的数值加入到一个新列中
        reshaped_data = np.column_stack((reshaped_data, all_outputs_numpy[rows * 90:]))
    # 对每一行进行排序
    reshaped_data = np.sort(reshaped_data, axis=1)
    # 转换为DataFrame
    df = pd.DataFrame(reshaped_data)
    df = df.T

    # 写入CSV文件
    df.to_csv('pred_90.csv', index=False, header=False)








if __name__ == '__main__':
    args = parse_args()
    main(args)
