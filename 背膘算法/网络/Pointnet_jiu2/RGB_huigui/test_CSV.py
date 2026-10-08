'''
计算测试集的预测值、标签，90个值（每个样本预测90次）为一列
将预测值，标签，保存在.csv文件中    test的时候mydataset部分不影响
'''

import torch

from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor,ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import numpy as np
import os
import pandas as pd
# from EfficientViT import EfficientViT_M0 as create_model
#from model.EfficientViT_CA import EfficientViT_M0 as create_model
from sklearn.metrics import r2_score,mean_absolute_error
# from official_my_9_1 import cmt_ti as CMT
from model.EfficientViT import EfficientViT_M0 as create_model
# from EfficientViT import EfficientViT_M0 as create_model
# from EfficientViT import EfficientViT_M0 as create_model
# from official_my_9_CA import cmt_ti as create_model
# from official_my_9_1 import cmt_ti as create_model
from skimage import io



transforms = transforms.Compose([
   # transforms.Resize(256),  # 将图片短边缩放至256，长宽比保持不变：
    #transforms.CenterCrop(224),  # 将图片从中心切剪成3*224*224大小的图片


    # transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 把图片进行归一化，并把数据转换成Tensor类型
    transforms.Normalize(mean=(0.44939196,), std=(0.26499626,))
    # transforms.Normalize(mean=(0.44920433,), std=(0.26527813,))    #这是第二组数据集的
# transforms.Normalize(mean=(0.4544133,), std=(0.26383656,))
])


class MyDataset(Dataset):
    def __init__(self, img_path, transform=None):
        super(MyDataset, self).__init__()
        self.root = img_path

      #  self.txt_root = self.root + '\\' + 'train.txt'
        self.txt_root = self.root + '.txt'
        f = open(self.txt_root, 'r')
        data = f.readlines()

        imgs = []
        labels = []
        for line in data:
            line = line.rstrip()
            word = line.split()
            # print(word[0], word[1]
            #       #, word[2]
            #       )
            # word[0]是图片名字.jpg  word[1]是label  word[2]是文件夹名，如sunflower
            #imgs.append(os.path.join(self.root, word[2], word[0]))
            imgs.append(os.path.join(self.root, word[0]))

            labels.append(word[1])
        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        img = Image.open(img)

        # 此时img是PIL.Image类型   label是str类型

        if self.transform is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int32)
        label = torch.from_numpy(label)

        return img, label
# class MyDataset(Dataset):
#     def __init__(self, root_dir, names_file, transform=None):
#         self.root_dir = root_dir
#         self.names_file = names_file
#         self.transform = transform
#         self.size = 0
#         self.names_list = []
#
#         if not os.path.isfile(self.names_file):
#             print(self.names_file + 'does not exist!')
#         file = open(self.names_file)
#         for f in file:
#             self.names_list.append(f)
#             self.size += 1
#
#     def __len__(self):
#         #获取数据集长度
#         return self.size
#
#     def __getitem__(self, idx):
#         #获取数据集中每一项
#         # image_path = self.root_dir + self.names_list[idx].split(' ')[0]
#         image_path = os.path.join(self.root_dir, self.names_list[idx].split(' ')[0])   #更改师姐的路拼接
#         if not os.path.isfile(image_path):
#             print(image_path + 'does not exist!')
#             return None
#         image = io.imread(image_path)   # use skitimage
#         # image = Image.open(image_path)             #上一句是师姐源代码，更改成这一句打开
#         label = int(self.names_list[idx].split(' ')[1])
#
#
#
#         if self.transform is not None:
#             image = self.transform(image)
#
#
#         # sample = {'image': image, 'label': label}   #原来返回sample，现在返回image，label
#
#
#
#         return image ,label           #  image dtype torch.float32,Size([1, 224, 224])



def main():






    # time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')

    test_path = r'D:\106dong\2\test'
    test_dataset = MyDataset(test_path, transform=transforms)
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # data_transform = transforms([transforms.Resize([224, 224]),
    #                                transforms.ToTensor(),
    #                                transforms.Normalize(mean=(0.4544133,), std=(0.26383656,))])

    # test_dataset = MyDataset(root_dir=r'D:\106dong\1\test', names_file=r'D:\106dong\1\test.txt',
    #
    #                         transform=transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             # collate_fn=val_dataset.collate_fn
                                             )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model().to(device)

    #加载模型
    model.load_state_dict(torch.load(r"D:\Backfat_jiu\Regression\weights\best_model2024_06_07_19_50.pth"))
    #获取预测结果


    model.eval()







    all_labels_numpy = []
    all_outputs_numpy = []


    for i, data in enumerate(test_dataloader):
            images, label = data

            # images = torch.mean(images, dim=1, keepdim=True)

            images = images.to(device)

            out = model(images.to(torch.float32))

            # out = out.reshape(-1)

            label = label.to(device)
            label = label.to(torch.float32)

            # label_numpy = label.reshape(-1)
            label_numpy = label.cpu().numpy()

            out_numpy = out.cpu()
            # print(out_numpy)

            # out_numpy = out_numpy.reshape(-1)
            # print(out_numpy)

            out_numpy = out_numpy.detach().numpy()
            # print(out_numpy)

            all_labels_numpy = np.append(all_labels_numpy, label_numpy)
            all_outputs_numpy = np.append(all_outputs_numpy, out_numpy)









            #
            # print('dim:', all_labels_numpy.ndim)  # 查看数组的维度
            # print('shape:', all_labels_numpy.shape)  # 查看数组的形状，即几行几列
            # print('size:', all_labels_numpy.size)  # 查看数组共有多少个元素
            # print('type:', all_labels_numpy.dtype)  # 查看数组中的元素是什么类型
            #
            # print('*************************************')
            #
            # print(all_outputs_numpy)
            # print('dim:', all_outputs_numpy.ndim)  # 查看数组的维度
            # print('shape:', all_outputs_numpy.shape)  # 查看数组的形状，即几行几列
            # print('size:', all_outputs_numpy.size)  # 查看数组共有多少个元素
            # print('type:', all_outputs_numpy.dtype)  # 查看数组中的元素是什么类型


            # print('loss is {}'.format(loss))


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
    main()


