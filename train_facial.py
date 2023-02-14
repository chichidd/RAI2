import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import random
from tqdm import trange
import pickle
import utils
from conf import settings

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-inter_propor', type=float, default=0.6)
    parser.add_argument('-rand_seed', type=int, default=None, help='random_seed')
    parser.add_argument('-copy_id', type=int, default=None, help='different copy id')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-gpu_id', type=int, default=0, help='device id')

    parser.add_argument('-same_data_dist', type=bool, default=True, help='whether the adversary has the same data distribution as the victim. If True, it is used for preparing the surrogate models and the look-up table, otherwise it is used to train real adversary models.')
    parser.add_argument('-adaptive_trans', type=str, default=None, help='the adversary adaptive attack for dataset similarity estimation, used for data_dist is not same')
    
    parser.add_argument('-dst_ratio', type=int, default=1, help='shrink the dataset size by dst_ratio times')

    # set to 0.6 for training with lower epochs
    parser.add_argument('-epoch_range', type=float, default=1.0, help='epoch ratio comparing to default setting')
    args = parser.parse_args()
    print("Args:", args)
    torch.cuda.set_device(args.gpu_id)
    
    if args.rand_seed is not None:
        np.random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        torch.cuda.manual_seed(args.rand_seed)
        random.seed(args.rand_seed)
        torch.backends.cudnn.deterministic=True

    training_loader, set_mean, set_std, num_classes = utils.get_facial_dataloader(
        inter_propor=args.inter_propor, batch_size=args.b, num_workers=4 * torch.cuda.device_count(),
        shuffle=True, same_data_dist=args.same_data_dist, adaptive_trans=args.adaptive_trans, dst_ratio=args.dst_ratio,seed=args.rand_seed)
    
    test_loader = utils.get_test_dataloader_facial(
        set_mean, set_std, num_workers=4 * torch.cuda.device_count(), batch_size=args.b, shuffle=False)

    net = getattr(models, args.net)(num_classes=num_classes).cuda()

    # process training hyperparameters
    epochs = int(settings.CASE_STUDY_EPOCH * args.epoch_range)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-5)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.CASE_STUDY_MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = utils.WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # process saving directory
    chkfolder_subset = f"facial_{'same_dist' if args.same_data_dist else 'diff_dist'}_{f'dst_ratio{args.dst_ratio}_' if args.dst_ratio != 1 else ''}{args.inter_propor}"
    if args.rand_seed is not None:
        chkfolder_subset += "_rs{}".format(args.rand_seed)
    checkpoint_path = settings.CASE_STUDY_CHECKPOINT_PATH
    if args.adaptive_trans is not None:
        checkpoint_path = os.path.join(checkpoint_path, 'adaptive_trans', f'{args.net}_adaptive_{args.adaptive_trans}_{chkfolder_subset}')
    elif args.dst_ratio != 1:
        if args.dst_ratio == 10:
            epochs *= 2 # for higher validation accuracy
        checkpoint_path = os.path.join(checkpoint_path, 'change_dst_size', f'{args.net}_{chkfolder_subset}')
    elif args.epoch_range != 1:
        checkpoint_path = os.path.join(checkpoint_path, 'low_epoch', f'epoch{epochs}', '_'.join([args.net, chkfolder_subset]))
    else:
        checkpoint_path = os.path.join(checkpoint_path, '_'.join([args.net, chkfolder_subset]))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print("Save at:", checkpoint_path)

    best_acc = 0.0
    tqdm_range = trange(1, epochs + 1, desc='Loss', leave=True)
    for epoch in tqdm_range:
        if epoch > args.warm:
            train_scheduler.step(epoch)

        loss, acc_train = utils.train(epoch, net, training_loader, loss_function, optimizer, warmup_epoch=args.warm, warmup_scheduler=warmup_scheduler, adv_training=False, verbose=False)
        acc_val = utils.eval_training(epoch, net, test_loader, loss_function, verbose=False)

        if best_acc < acc_val:
            if args.adaptive_trans is not None:
                weights_path = os.path.join(checkpoint_path, 'model_with_acc_{}.pth'.format(args.copy_id))
                torch.save({'net_sd':net.state_dict(), 'mean_std':(set_mean, set_std), 'val_acc':acc_val}, weights_path)
            else:
                weights_path = os.path.join(checkpoint_path, 'model_{}.pth'.format(args.copy_id))
                torch.save(net.state_dict(), weights_path)
            best_acc = acc_val
            
        tqdm_range.set_description(f"L:{loss:.2f}, lr:{optimizer.param_groups[0]['lr']:.2f}, {acc_val*100:.2f}/{best_acc*100:.2f}, {acc_train*100:.2f}")
        tqdm_range.refresh()
