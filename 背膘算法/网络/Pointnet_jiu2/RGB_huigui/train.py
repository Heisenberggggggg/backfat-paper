
'''
用这个训练代码，需要把模型文件放在同一级文件夹内，根据霹雳更改
MyDataset部分是自己写的，师姐，曹师兄也有dataset部分
'''


import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from official_my_9_1 import cmt_ti as create_model
# from EfficientViT import EfficientViT_M0 as create_model
from model.EfficientViT import EfficientViT_M0 as create_model
# from model.EfficientViT_CA import EfficientViT_M0 as create_model
from utils import train_one_epoch, evaluate,seed_everything,worker_init_fn
from PIL import Image
import datetime
from skimage import io
import math
from functools import partial
import random
import torch.optim.lr_scheduler as lr_scheduler




#-------------------------------------这个dataset是之前自己写的----------------------------------------------------#

class MyDataset(Dataset):
    def __init__(self, images_path, transform=None):
        super(MyDataset, self).__init__()
        self.root = images_path
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
            # imgs.append(os.path.join(self.root, word[2], word[0]))
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

        # img = Image.open(img).convert('RGB')
        img = Image.open(img)
        # 此时img是PIL.Image类型   label是str类型

        if self.transform is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.int32)
        label = torch.from_numpy(label)
        # print('img size',img.size)
        # print('img dtype', img.dtype)
        # print(img)
        # print('label size',label.size)
        # print('label dtype', label.dtype)
        # print(label)

        return img, label           #img dtype torch.float32  label dtype torch.int32




def main(args):

    seed = 589
    rank = 0
    seed_everything(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()





    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),

                                     transforms.RandomChoice([transforms.RandomVerticalFlip(),
                                                             transforms.RandomHorizontalFlip(),
                                                              transforms.RandomAffine(degrees=0, translate=(0.05, 0.5)),
                                                              transforms.RandomRotation(degrees=(45, 90))]),
                                     transforms.ToTensor(),
                                     # transforms.Normalize(mean=(0.44939196,), std=(0.26499626,))# 这是第1组数据集的
                                     # transforms.Normalize(mean=(0.4544133,), std=(0.26383656,))  #48头数据的
                                     # transforms.Normalize(mean=(0.44920433,), std=(0.26527813,))  # 这是第二组数据集的
                                     # transforms.Normalize(mean=(0.44866872,), std=(0.26525354,))  # 这是第4组数据集的
                                     transforms.Normalize(mean=(0.44790643,), std=(0.26498762,))  # 这是第3组数据集的
                                     ]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   # transforms.Normalize(mean=(0.44939196,), std=(0.26499626,))# 这是第1组数据集的
                                   # transforms.Normalize(mean=(0.4544133,), std=(0.26383656,))  #48头数据的
                                   # transforms.Normalize(mean=(0.44920433,), std=(0.26527813,))  # 这是第二组数据集的
                                   # transforms.Normalize(mean=(0.44866872,), std=(0.26525354,))  # 这是第4组数据集的
                                   transforms.Normalize(mean=(0.44790643,), std=(0.26498762,))# 这是第3组数据集的
                                   ])}




#----------------------下面的实例化部分是自己的dataset对应的-------------------------------------------------------------#
    train_path = r'D:\106dong\3\train'
    val_path = r'D:\106dong\3\test'
    # 实例化训练数据集
    train_dataset = MyDataset(images_path=train_path,

                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataset(images_path=val_path,

                            transform=data_transform["val"])


# ----------------------下面的实例化部分是改之后师姐的dataset对应的-------------------------------------------------------------#
#     train_dataset = MyDataset(root_dir=r'D:\106dong\1\train',names_file=r'D:\106dong\1\train.txt',
#
#                               transform=data_transform["train"])
#     val_dataset = MyDataset(root_dir=r'D:\106dong\1\valid',names_file=r'D:\106dong\1\valid.txt',
#
#                             transform=data_transform["val"])




    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))



    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
                                               # collate_fn=train_dataset.collate_fn
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
                                             # collate_fn=val_dataset.collate_fn
                                             )

    # model = create_model(num_classes=args.num_classes).to(device)
    model = create_model().to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print('载入预训练权重',model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]


    #optimizer = optim.Adam(pg, lr=args.lr, weight_decay=5E-2)      #transformer用的SGD

    # optimizer = optim.Adam(pg, lr=0.01, weight_decay=5E-2)  #这里是A金的优化器及学习率调整
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)

    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    optimizer = optim.Adam(pg, lr=0.001, weight_decay=0.08)   #ajing 设置的是0.01，原代码是5e-2
    def lr_lambda(epoch):
        if epoch < args.epochs*0.5:
            return 1.0  # 在前15轮保持原学习率
        elif epoch < args.epochs*0.75:
            return 0.1  # 在接下来的7轮将学习率缩小为原来的0.1倍
        else:
            return 0.01  # 在接下来的8轮将学习率缩小为原来的0.01倍
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    best_loss = 10000
    # best_MSE = 10000
    # best_R2 = 0
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')

    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()

        # validate
        val_loss = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        print(f"Epoch {epoch+1}: Learning rate: {scheduler.get_last_lr()}")

        # print('batch_size is {}'.format(batch_size))

        tags = ["train_loss", "val_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], val_loss, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        #torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #
        #     torch.save(model.state_dict(), "./weights/best_model{}.pth".format(str(time_str)))
        #
        #
        # torch.save(model.state_dict(), "./weights/latest_model{}.pth".format(str(time_str)))

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(model.state_dict(), "./weights/best_model{}.pth".format(str(time_str)))


        torch.save(model.state_dict(), "./weights/latest_model{}.pth".format(str(time_str)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)     #回归网络，这里为1
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=16)
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--lrf', type=float, default=0.01)



    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')  #./pre_weight/swin_tiny_patch4_window7_224.pth
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
