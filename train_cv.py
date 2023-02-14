import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from conf import settings
import utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net architecture')
    parser.add_argument('-dataset', type=str, required=True, help='dataset, cifar10, cifar100 or tinyimagenet')
    parser.add_argument('-subset', type=int, default=1, help='subset index, 1 or 2, 1 denotes the victim part and 2 denotes the adversary or surrogate part. If None, training with all dataset.')
    parser.add_argument('-inter_propor', type=float, default=0.5, help="intersection proportion between two subsets, i.e., dataset similarity")
    parser.add_argument('-rand_seed', type=int, default=None, help='random_seed')
    parser.add_argument('-copy_id', type=int, default=None, help='different copy id')

    parser.add_argument('-gpu_id', type=int, default=0, help='device id')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-epoch_range', type=float, default=1.0, help='epoch ratio comparing to default setting')
    
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    if args.rand_seed is not None:
        np.random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        torch.cuda.manual_seed(args.rand_seed)
        random.seed(args.rand_seed)
        torch.backends.cudnn.deterministic=True

    # process network
    net = utils.get_network(args.dataset, args.net, True)

    # process dataset
    if args.subset is None:
        set_mean, set_std = utils.get_dataset_mean_std(args.dataset)
        training_loader = utils.get_training_dataloader(args.dataset, set_mean, set_std, num_workers=8, batch_size=args.b, shuffle=True)
        chkfolder_subset = f"_{args.subset}"
    else:
        training_loader, set_mean, set_std = utils.get_intersect_dataloader(dataset=args.dataset, propor=args.inter_propor, num_workers=8, batch_size=args.b, shuffle=True, sub_idx=args.subset-1)
        chkfolder_subset = f"subset{args.subset}_{args.dataset}_intersect_{args.inter_propor}"

    test_loader = utils.get_test_dataloader(args.dataset, set_mean, set_std, num_workers=8, batch_size=args.b, shuffle=False)

    if args.rand_seed is not None:
        chkfolder_subset += "_rs{}".format(args.rand_seed)
    
    # process training hyperparameters
    epochs, milestones = utils.get_dataset_hyperparam(args.dataset)
    epochs = int(epochs * args.epoch_range)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2) #learning rate decay
    warmup_scheduler = utils.WarmUpLR(optimizer, len(training_loader) * args.warm)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net + chkfolder_subset)


    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)


    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        utils.train(epoch, net, training_loader, loss_function, optimizer, args.warm, warmup_scheduler, verbose=True)
        if epoch > milestones[1]:
            acc = utils.eval_training(epoch, net, test_loader, loss_function, verbose=True)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > milestones[1] and best_acc < acc:
            weights_path = os.path.join(checkpoint_path, 'model_{}.pth'.format(args.copy_id))
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        
