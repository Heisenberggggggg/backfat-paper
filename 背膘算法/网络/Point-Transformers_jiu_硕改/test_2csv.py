import torch
import numpy as np
from torch.utils.data import Dataset
import sys
import os
import pandas as pd
import argparse
from utils import MyDataset
import importlib
import tqdm
import hydra
import omegaconf
import time
os.HYDRA_FULL_ERROR=1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

@hydra.main(config_path='config', config_name='backfat', version_base='1.2')
def main(args):
    start_time = time.time()
    omegaconf.OmegaConf.set_struct(args, False)
    print(omegaconf.OmegaConf.to_yaml(args))  # 1
    args.input_dim = 6 if args.use_normals else 3

    test_path = r''


    test_dataset = MyDataset(root=test_path, args=args, )

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,num_workers=8)
    total_files = len(testDataLoader.dataset)
    # model = importlib.import_module(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # classifier = model.get_model(num_class=1, normal_channel=args.use_normals).to(device)
    classifier = getattr(importlib.import_module('models.{}.model_gai'.format(args.model.name)), 'PointTransformer_jiu')(
        args).to(device)
    # classifier = getattr(importlib.import_module('models.Hengshuang.model'), 'PointTransformerCls')(args).to(device)
    classifier.apply(inplace_relu)  #1

    # 加载模型
    classifier.load_state_dict(torch.load(r".pth"))
    # 获取预测结果


    test_classifier = classifier.eval()


    all_labels_numpy = []
    all_outputs_numpy = []
    for j, (test_points, test_target) in enumerate(testDataLoader):


        test_points, test_target = test_points.to(device), test_target.to(device)  # 1
        # test_points = test_points.transpose(2, 1)
        test_pred = test_classifier(test_points)

        test_target = test_target.unsqueeze(1)

        test_target = test_target.to(torch.float32)
        label_numpy = test_target.cpu().numpy()
        all_labels_numpy = np.append(all_labels_numpy, label_numpy)

        out_numpy = test_pred.cpu()
        out_numpy = out_numpy.detach().numpy()
        all_outputs_numpy = np.append(all_outputs_numpy, out_numpy)

    total_time = time.time() - start_time
    files_per_second = total_files / total_time
    time_per_file = total_time / total_files

    print(f"总共测试了 {total_files} 个文件")
    print(f"总共耗时 {total_time:.2f} 秒")
    print(f"每秒可以测试 {files_per_second:.2f} 个文件")
    print(f"测试一个文件需要 {time_per_file:.4f} 秒")



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
    df.to_csv('labels_90.csv', index=False, header=False, float_format='%.2f')


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
    df.to_csv('pred_90.csv', index=False, header=False, float_format='%.2f')




if __name__ == '__main__':

    main()
