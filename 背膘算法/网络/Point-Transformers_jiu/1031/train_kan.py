"""
Author: Benny
Date: Nov 2019
"""
from utils import MyDataset
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf
import random
from sklearn.metrics import mean_absolute_error,r2_score
from models.Hengshuang.model_Geo import PointTransformerCls
@hydra.main(config_path='config', config_name='cls', version_base='1.2')
def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    omegaconf.OmegaConf.set_struct(args, False)

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

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    print(omegaconf.OmegaConf.to_yaml(args))  # 1
    '''DATA LOADING'''

    logger.info('Load dataset ...')

    train_path = r'D:\71data\B\1\train'
    test_path = r'D:\71data\B\1\test'
    train_dataset = MyDataset(root=train_path, args=args, )
    test_dataset = MyDataset(root=test_path, args=args, )


    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=8, pin_memory=True, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                 pin_memory=True, drop_last=True)



    '''MODEL LOADING'''
    # num_class = args.num_class
    args.num_class = 1
    args.input_dim = 6 if args.use_normals else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 1


    # classifier = getattr(importlib.import_module('models.{}.model_Geo'.format(args.model.name)), 'PointTransformerCls')(
    #     args).to(device)
    classifier = PointTransformerCls(args).to(device)
    criterion = torch.nn.MSELoss().to(device)  # 损失函数要注意

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)  # gamma原来是0.3,现在改为0.8,学习率衰减系数
    global_epoch = 0
    global_step = 0
    # best_instance_acc = 0.0
    # best_class_acc = 0.0
    # best_epoch = 0
    # mean_correct = []
    all_labels_numpy = []
    all_outputs_numpy = []
    best_loss = 1000.0
    best_R2 = 0  #定义一个初始R2
    '''TRANING'''
    time_str_last = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')
    savepath_last = str(checkpoints_dir) + '/last_model{}.pth'.format(str(time_str_last))

    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        classifier = classifier.train()
        accu_loss = torch.zeros(1).to(device)  # 1

        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)


            points, target = points.to(device), target.to(device)  # 1
            pred = classifier(points)  # 1
            target = target.unsqueeze(1)  # 1
            loss = criterion(pred.float(), target.float())  # 1
            loss.backward()  # 1
            accu_loss += loss.detach()  # 1
            optimizer.step()  # 1
            optimizer.zero_grad()
            global_step += 1




        scheduler.step()

        train_epoch_loss = accu_loss.item() / (batch_id + 1)  # 1
        print('epoch_loss:{}'.format(train_epoch_loss))  # 1
        logger.info('Train Instance loss(jiu) : %f' % train_epoch_loss)  # 1

        test_classifier = classifier.eval()
        with torch.no_grad():

            accu_loss = torch.zeros(1).to(device)  # 累计损失 1
            for j, (test_points, test_target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                # if not args.use_cpu:
                # test_points, test_target = test_points.cuda(), test_target.cuda()
                test_points, test_target = test_points.to(device), test_target.to(device)  # 1
                # test_points = test_points.transpose(2, 1)
                test_pred = test_classifier(test_points)
                test_target = test_target.unsqueeze(1)
                loss = criterion(test_pred.float(), test_target.long())
                # print('test_pred', test_pred)
                # print('test_target', test_target)
                # print('test_loss',loss)
                accu_loss += loss  # 1

                testDataLoader.desc = "[valid(test) epoch {}] loss: {:.3f} ".format(epoch, accu_loss.item() / (j + 1), )


                label = test_target.reshape(-1)
                out = test_pred.reshape(-1)

                label = label.to(device)
                label = label.to(torch.float32)

                label_numpy = label.reshape(-1)
                label_numpy = label_numpy.cpu().numpy()

                out_numpy = out.cpu()


                out_numpy = out_numpy.reshape(-1)


                out_numpy = out_numpy.detach().numpy()


                all_labels_numpy = np.append(all_labels_numpy, label_numpy)
                all_outputs_numpy = np.append(all_outputs_numpy, out_numpy)

            test_epoch_loss = accu_loss.item() / (j + 1)
            print('test_epoch_loss:{}'.format(test_epoch_loss))
            logger.info('Test Instance loss(jiu) : %f' % test_epoch_loss)

            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')
            R2 = r2_score(all_labels_numpy, all_outputs_numpy)
            # MAE = mean_absolute_error(all_labels_numpy, all_outputs_numpy)
            # logger.info('Test R2 score: %f' % R2)
            test_R2 = R2
            if test_R2 > best_R2:
                best_R2 = test_R2
                savepath = str(checkpoints_dir) + '/best_R2{}_model{}.pth'.format(str(best_R2),str(time_str))

                logger.info('Saving at %s' % savepath)
                torch.save(classifier.state_dict(), savepath)

            if (test_epoch_loss <= best_loss):
                best_loss = test_epoch_loss

                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_loss{}_model{}.pth'.format(str(best_loss),str(time_str))

                logger.info('Saving at %s' % savepath)

                torch.save(classifier.state_dict(), savepath)
            if (epoch + 1) % 10 == 0:
                savepath_epoch = str(checkpoints_dir) + '/epoch{}_model{}.pth'.format(epoch + 1, time_str)
                logger.info('Saving model at epoch %d to %s' % (epoch + 1, savepath_epoch))
                torch.save(classifier.state_dict(), savepath_epoch)
            if epoch >= args.epoch - 5:
                savepath_last_five = str(checkpoints_dir) + '/last5_epoch{}_model{}.pth'.format(epoch + 1, time_str)
                logger.info('Saving last 5 epochs model at epoch %d to %s' % (epoch + 1, savepath_last_five))
                torch.save(classifier.state_dict(), savepath_last_five)

            global_epoch += 1
            torch.save(classifier.state_dict(), savepath_last)

    logger.info('End of training...')


if __name__ == '__main__':
    main()