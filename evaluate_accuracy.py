import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import pickle
import utils
from conf import settings
COPY_NUM = 10
# dataset_list = ['cifar10', 'cifar100', 'tinyimagenet']
# nets_list = [['resnet18', 'vgg13', 'resnet34', 'resnet50'],
#        ['resnet34', 'resne101', 'vgg16'],
#        ['resnet152', 'densenet121', 'vgg19']]

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):

    with torch.no_grad():
       
        maxk = max(topk)  
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)
        y_pred = y_pred.t()

        target_reshaped = target.view(1, -1).expand_as(y_pred)
        
        correct = (y_pred == target_reshaped) 

        list_topk_accs = [] 
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)
            topk_acc = tot_correct_topk / batch_size
            list_topk_accs.append(topk_acc.item())
        return list_topk_accs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, help='dataset: cifar10, cifar100, tinyimagenet')
    parser.add_argument('-gpu_id', type=int, default=0, help='gpu id used for inference')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    test_loader = utils.get_test_dataloader(args.dataset,
        mean=(0, 0, 0), std=(1, 1, 1), batch_size=512, num_workers=8, shuffle=False, pin_memory=False)

    file_path = os.path.join(settings.CHECKPOINT_PATH, 'similarity', args.dataset)
    # get different architectures trained on this dataset
    model_type_list = os.listdir(file_path)
    # get names of folders containing models trained on intersected datasets
    inter_names = os.listdir(os.path.join(file_path, model_type_list[0]))
    # dictionary storing normalization mean and std for each intersection case
    mean_std_dict = utils.get_intersection_mean_std_dict(args.dataset)
    # path for storing accuracy results
    result_path = os.path.join(settings.RESULT_PATH, "accuracy")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # Load models into model_dict and proceed inference. Code can be adjusted according to RAM size.
    all_acc = {}
    for model_type in model_type_list:
        print(model_type)
        model_acc = {}
        for intersection in inter_names:
            mean, std = mean_std_dict[intersection]
            models_list = []
            for i in range(COPY_NUM):
                
                net = utils.get_network(args.dataset, model_type, False)
                net.load_state_dict(
                    torch.load(os.path.join(file_path, model_type, intersection,'model_{}.pth'.format(i)), 
                    map_location='cpu'))
                net.to('cpu')
                net.eval()
                models_list.append(nn.Sequential(transforms.Normalize(mean, std), net))

            print("Initialized model {}/{}.".format(model_type, intersection))

            with torch.no_grad():
                model_chkp_acc = []
                for model_id, model in enumerate(models_list):
                    model.cuda()
                    outputs = []
                    targets = []
                    for x, y in test_loader:
                        x, y = x.cuda(), y.cuda()
                        outputs.append(model(x))
                        targets.append(y.reshape(-1, 1))
                    outputs = torch.cat(outputs, dim=0).cpu()
                    targets = torch.cat(targets, dim=0).cpu()
                    top1_acc, top5_acc = accuracy(outputs, targets, topk=(1, 5))
                    model_chkp_acc.append((top1_acc, top5_acc))
                    print(model_type, intersection, model_id, top1_acc, top5_acc)
                    del model
                    torch.cuda.empty_cache()
                model_acc[intersection] = model_chkp_acc
                
        all_acc[model_type] = model_acc
    pickle.dump(all_acc, open(
            os.path.join(result_path, f"{args.dataset}_acc.pkl"), "wb"))


