import torch.nn as nn
import matplotlib.pyplot as plt
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from models.Hengshuang.transformer import TransformerBlock
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import MyDataset
import importlib
import tqdm
import hydra
import omegaconf
import time
class MixedPooling(nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedPooling, self).__init__()
        self.alpha = alpha
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        mixed_pooled = self.alpha * avg_pooled + (1 - self.alpha) * max_pooled
        return mixed_pooled.view(x.size(0), -1)

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)

class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats

class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.mixed_pooling = MixedPooling(alpha=0.5)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_c)
        )
        self.nblocks = nblocks

        # 用于存储中间层输出的变量
        self.backbone_output = None
        self.mixed_pooling_output = None

        # 注册钩子
        self.backbone.register_forward_hook(self._hook_backbone)
        self.mixed_pooling.register_forward_hook(self._hook_mixed_pooling)

    def _hook_backbone(self, module, input, output):
        self.backbone_output = output

    def _hook_mixed_pooling(self, module, input, output):
        self.mixed_pooling_output = output

    def forward(self, x):
        points, _ = self.backbone(x)
        pooled_features = self.mixed_pooling(points)
        res = self.fc2(pooled_features)
        return res

    def get_visualization_data(self):
        return self.backbone_output, self.mixed_pooling_output

def plot_heatmap(data, title="Heatmap"):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

@hydra.main(config_path='config', config_name='cls', version_base='1.2')
def main(args):
    start_time = time.time()
    omegaconf.OmegaConf.set_struct(args, False)
    print(omegaconf.OmegaConf.to_yaml(args))  # 1
    args.input_dim = 6 if args.use_normals else 3

    test_path = r'D:\point_single\1'


    test_dataset = MyDataset(root=test_path, args=args, )

    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,num_workers=8)
    total_files = len(testDataLoader.dataset)
    # model = importlib.import_module(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # classifier = model.get_model(num_class=1, normal_channel=args.use_normals).to(device)
    classifier = getattr(importlib.import_module('models.{}.model_guocheng'.format(args.model.name)), 'PointTransformerCls')(
        args).to(device)
    # classifier = getattr(importlib.import_module('models.Hengshuang.model'), 'PointTransformerCls')(args).to(device)
    classifier.apply(inplace_relu)  #1

    #加载模型
    classifier.load_state_dict(torch.load(r"D:\Point-Transformers_jiu\log\regression_jiu\model_1024\checkpoints\epoch70_model2024_10_31_18_20.pth"))
    #获取预测结果


    test_classifier = classifier.eval()


    all_labels_numpy = []
    all_outputs_numpy = []
    # for j, (test_points, test_target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
    for j, (test_points, test_target) in enumerate(testDataLoader):
        # print(test_points.shape)

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



# 假设 model 是 PointTransformerCls 的实例，并且已经经过前向传播
# 例如：model = PointTransformerCls(cfg)
#       output = model(input_data)

    backbone_output, mixed_pooling_output = classifier.get_visualization_data()

    # 从backbone_output中提取points
    points, _ = backbone_output

    # 将输出转换为 numpy 数组
    points_np = points.detach().cpu().numpy()
    mixed_pooling_output_np = mixed_pooling_output.detach().cpu().numpy()

    # 选择一个样本进行可视化
    sample_index = 0
    points_sample = points_np[sample_index]
    mixed_pooling_sample = mixed_pooling_output_np[sample_index]

    # 绘制热力图
    plot_heatmap(points_sample, title="Backbone Points Output Heatmap")
    plot_heatmap(mixed_pooling_sample, title="Mixed Pooling Output Heatmap")

if __name__ == '__main__':

    main()