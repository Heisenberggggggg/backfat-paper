"""
Author: jiu
Date: Nov 2019

代码为跑回归任务的，换模型需要再改损失函数

"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from data_utils.dataloader import MyDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_jiu_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=1, type=int,  help='regrsssion is 1')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    #这个采样方式注意，关键是数据随机还是不随机

    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=1):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)


        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('regression_jiu')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')



    train_path = r'D:\data_point\48point\48_1.1\train'
    test_path = r'D:\data_point\48point\48_1.1\test'

    train_dataset = MyDataset(root=train_path, args=args,)
    test_dataset = MyDataset(root=test_path, args=args,)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_regression.py', str(exp_dir))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 1

    classifier = model.get_model(num_class, normal_channel=args.use_normals).to(device)
    criterion = model.get_loss().to(device)
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)
    global_epoch = 0
    global_step = 0
    # best_instance_acc = 0.0
    # best_class_acc = 0.0

    best_loss = 1000.0



    '''TRANING'''

    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        # mean_correct = []
        classifier = classifier.train()
        classifier = classifier  #1

        # optimizer.zero_grad()
        accu_loss = torch.zeros(1).to(device)   #1
        # scheduler.step()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):


            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])  #前三列进行随机缩放操作
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])   #前三列进行随机平移操作
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.to(device), target.to(device)  #1
            # if not args.use_cpu:
            #     points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            target = target.unsqueeze(1)
            # print('pred',pred)
            # print('target', target)
            loss = criterion(pred.float(), target.float(), trans_feat)
            # print('训练的',loss)
            # pred_choice = pred.data.max(1)[1]

            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()

            accu_loss += loss.detach() #1

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            testDataLoader.desc = "[train epoch {}] loss: {:.3f}  ".format(epoch,
                                                                           accu_loss.item() / (batch_id + 1))  # 1
        scheduler.step()    #1
        # train_instance_acc = np.mean(mean_correct)
        train_epoch_loss = accu_loss.item() / (batch_id + 1)
        print('epoch_loss:{}'.format(train_epoch_loss))
        log_string('Train Instance loss(jiu) : %f' % train_epoch_loss)

        test_classifier = classifier.eval()
        with torch.no_grad():   #验证（测试部分）


            accu_loss = torch.zeros(1).to(device)  # 累计损失 1
            for j, (test_points, test_target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

                # if not args.use_cpu:
                    # test_points, test_target = test_points.cuda(), test_target.cuda()
                test_points, test_target = test_points.to(device), test_target.to(device)#1
                test_points = test_points.transpose(2, 1)
                test_pred, test_trans_feat = test_classifier(test_points)
                test_target = test_target.unsqueeze(1)
                loss = criterion(test_pred.float(), test_target.long(), test_trans_feat)
                # print('测试的',loss)
                accu_loss += loss  #1

                testDataLoader.desc = "[valid(test) epoch {}] loss: {:.3f} ".format(epoch , accu_loss.item() / (j + 1), )

            test_epoch_loss = accu_loss.item() / (j + 1)
            print('test_epoch_loss:{}'.format(test_epoch_loss))
            log_string('Test Instance loss(jiu) : %f' % test_epoch_loss)

            # instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)


            # if (instance_acc >= best_instance_acc):
            #     best_instance_acc = instance_acc
            #     best_epoch = epoch + 1
            #
            # if (class_acc >= best_class_acc):
            #     best_class_acc = class_acc
            # log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            # log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (test_epoch_loss <= best_loss):
                best_loss = test_epoch_loss

                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model_jiu.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    # 'epoch': best_epoch,
                    # 'instance_acc': instance_acc,
                    # 'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
