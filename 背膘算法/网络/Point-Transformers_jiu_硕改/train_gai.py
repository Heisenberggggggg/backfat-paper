"""
Author: Benny
Date: Nov 2019
"""
from utils import MyDataset
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import provider
import shutil
import hydra
import omegaconf
import random
from models.Hengshuang.model_gai import PointTransformer_jiu  #这里是导入模型
@hydra.main(config_path='config', config_name='backfat', version_base='1.2')

def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    omegaconf.OmegaConf.set_struct(args, False)

    # '''CREATE DIR'''
    # timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # exp_dir = Path('./log/')
    # exp_dir.mkdir(exist_ok=True)
    # exp_dir = exp_dir.joinpath('regression_146')
    # exp_dir.mkdir(exist_ok=True)
    # if args.log_dir is None:
    #     exp_dir = exp_dir.joinpath(timestr)
    # else:
    #     exp_dir = exp_dir.joinpath(args.log_dir)
    # exp_dir.mkdir(exist_ok=True)
    # checkpoints_dir = exp_dir.joinpath('checkpoints/')
    # checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = exp_dir.joinpath('logs/')
    # log_dir.mkdir(exist_ok=True)
    #
    # '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # logger = logging.getLogger(__name__)
    #
    # print(omegaconf.OmegaConf.to_yaml(args))  # 1
    # '''DATA LOADING'''

    # ... (在 main 函数内) ...

    omegaconf.OmegaConf.set_struct(args, False)

    # '''CREATE DIR'''  <-- Hydra 会自动完成这部分工作
    # 当使用Hydra时, 它会自动创建一个输出目录并进入该目录。
    # 我们只需要在其中创建子目录即可。

    # hydra.utils.get_original_cwd() 可以获取项目根目录
    # os.getcwd() 在这里会获取到hydra创建的输出目录，比如 log/regression_146/Hengshuang/2025-07-06/19-46-26

    output_dir = Path(os.getcwd())
    checkpoints_dir = output_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = output_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # hydra会自动处理日志，但如果你的logger有特定配置，可以保留
    logger = logging.getLogger(__name__)
    logger.info(f"All outputs will be saved to: {output_dir}")

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 打印所有配置，非常有用
    print(omegaconf.OmegaConf.to_yaml(args))

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    # 使用 hydra 获取项目根目录，这是最可靠的方法
    # hydra.utils.get_original_cwd() 会返回你执行 `python train_gai.py` 命令时所在的目录
    try:
        project_root = hydra.utils.get_original_cwd()
    except:
        # 如果不是通过hydra运行（例如直接在IDE中debug），则使用os.getcwd()
        project_root = os.getcwd()

    # 从项目根目录和配置文件中的名称，构建到数据文件夹的绝对路径
    train_path = os.path.join(project_root, args.data.root, args.data.train_dir)
    test_path = os.path.join(project_root, args.data.root, args.data.test_dir)
    # 如果要用验证集，也可以在这里加载
    # valid_path = os.path.join(project_root, args.data.root, args.data.valid_dir)

    logger.info(f"Loading training data from: {train_path}")
    logger.info(f"Loading testing data from: {test_path}")

    train_dataset = MyDataset(root=train_path, args=args, )
    test_dataset = MyDataset(root=test_path, args=args, )


    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=8, pin_memory=True, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8,
                                                 pin_memory=True)



    '''MODEL LOADING'''
    args.num_class = 1
    args.input_dim = 6 if args.use_normals else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 1
    classifier = PointTransformer_jiu(args).to(device)
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)  # gamma原来是0.3,现在改为0.8,学习率衰减系数
    global_epoch = 0
    global_step = 0

    all_labels_numpy = []
    all_outputs_numpy = []
    best_loss = 1000.0
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
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])  #随机缩放
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])    #随机平移
            # points = provider.random_flip_point_cloud(points)# 随机翻转点云
            # points = provider.jitter_point_cloud(points)# 随机抖动点云
            # points = provider.rotate_perturbation_point_cloud(points)# 随机旋转点云
            # 随机选择一种数据增强方法
            augmentations = [
                lambda x: provider.random_scale_point_cloud(x[:, :, 0:3]),
                lambda x: provider.shift_point_cloud(x[:, :, 0:3]),
                lambda x: provider.random_flip_point_cloud(x),
                lambda x: provider.jitter_point_cloud(x),
                lambda x: provider.rotate_perturbation_point_cloud(x)
            ]

            # 随机选择一种增强方法并应用
            chosen_augmentation = random.choice(augmentations)
            points[:, :, 0:3] = chosen_augmentation(points)

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

                test_pred = test_classifier(test_points)
                test_target = test_target.unsqueeze(1)
                loss = criterion(test_pred.float(), test_target.long())
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
            # print('test_epoch_loss:{}'.format(test_epoch_loss))
            logger.info('Test Instance loss(jiu) : %f' % test_epoch_loss)

            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')
            if epoch > 30 and (test_epoch_loss <= best_loss):
                best_loss = test_epoch_loss

                logger.info('Save model...')
                # savepath = str(checkpoints_dir) + '/best_loss{}_model{}.pth'.format(str(best_loss),str(time_str))
                savepath = str(checkpoints_dir) + '/best_loss{:.3f}_model{}.pth'.format(best_loss, str(time_str))
                logger.info('Saving at %s' % savepath)

                torch.save(classifier.state_dict(), savepath)

            if epoch >= args.epoch - 10:
                savepath_last_five = str(checkpoints_dir) + '/last5_epoch{}_model{}.pth'.format(epoch + 1, time_str)
                logger.info('Saving last 10 epochs model at epoch %d to %s' % (epoch + 1, savepath_last_five))
                torch.save(classifier.state_dict(), savepath_last_five)

            global_epoch += 1
            torch.save(classifier.state_dict(), savepath_last)

    logger.info('End of training...')


if __name__ == '__main__':
    main()