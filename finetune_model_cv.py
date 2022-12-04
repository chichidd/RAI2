import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from conf import settings
import utils


if __name__ == '__main__':
    # python finetune.py -net vgg16 -dataset cifar100 -chkp_folder hash/cifar100/ -chkp_file vgg16_0.pth -finetune_epoch 10 -subset 2 -copy_id 0 -gpu_id 1 -adv
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net architecture')
    parser.add_argument('-dataset', type=str, required=True, help='dataset, cifar10, cifar100 or tinyimagenet')
    parser.add_argument('-chkp_folder', type=str, default=None, help='checkpoint folder')
    parser.add_argument('-chkp_file', type=str, default=None, help='checkpoint file')
    parser.add_argument('-finetune_epoch', type=int, default=None, help='finetune epoch')
    parser.add_argument('-subset', type=int, default=None, help='subset 1 for victim part or 2 adversary part (i.e., unrelated data)')
    parser.add_argument('-copy_id', type=int, default=None, help='different copy id')
    parser.add_argument('-gpu_id', type=int, default=1, help='device id')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial finetuning learning rate')
    parser.add_argument('-adv_training' , action='store_true', default=False, helper="choice for adversarial training")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    net = utils.get_network(args.dataset, args.net, True)
    if args.subset is None:
        set_mean, set_std = utils.get_dataset_mean_std(args.dataset)
        training_loader = utils.get_training_dataloader(args.dataset, set_mean, set_std, num_workers=8, batch_size=args.b, shuffle=True)
    else:
        training_loader, set_mean, set_std = utils.get_intersect_dataloader(dataset=args.dataset, propor=0.0, num_workers=8, batch_size=args.b, shuffle=True, sub_idx=args.subset-1)

    test_loader = utils.get_test_dataloader(args.dataset, set_mean, set_std, num_workers=8, batch_size=args.b, shuffle=False)

    savename = ""
    if args.copy_id is not None:
        savename = f"_{args.copy_id}"

    loss_function = nn.CrossEntropyLoss()
    
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.chkp_folder)
    weights_path = os.path.join(checkpoint_path, args.chkp_file)
    print(weights_path)
    print('load training file to test acc...')
    net.load_state_dict(torch.load(weights_path))
    net.eval()
    best_acc = utils.eval_training(0, net, test_loader, loss_function)
    print(f'best acc is {best_acc:0.2f}')

    save_path = os.path.join(checkpoint_path, f'finetune{savename}/')
    if args.adv_training:
        save_path = os.path.join(checkpoint_path, f'advfinetune{savename}/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)
    for epoch in range(1, args.finetune_epoch + 1):
        
        utils.train(epoch, net, training_loader, loss_function, optimizer)
        acc = utils.eval_training(epoch, net, test_loader, loss_function)
        if epoch % 1 == 0:
            weights_path = os.path.join(save_path, 'finetune_{}.pth'.format(epoch))
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)



