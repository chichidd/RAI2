

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle
import sys
sys.path.append("../")
import utils
from conf import settings

np.random.seed(0)
torch.manual_seed(0)

dataset_abrev_dict = {'cifar10': 'c10', 'cifar100': 'c100', 'tinyimagenet':'tiny'}
vic_dict = {'cifar10': 'resnet18', 'cifar100': 'resnet34', 'tinyimagenet':'resnet152'}
epochs_dict = {
    'cifar10': [70, 80, 90, 100, 110, 120, 130, 140, 150],
    'cifar100': [120, 140, 160, 180, 200, 220, 240, 260, 280, 300],
    'tinyimagenet': [120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
}


def sample_intersect1_loader(dataset_name, n_sample=100):
    # Load dataset and select random data points
    (X_vic, y_vic), _ = pickle.load(open(os.path.join(settings.DATA_PATH, 'similarity', f'{dataset_name.upper()}_intersect_1.0.pkl'), 'rb'))
    np.random.seed(0)
    idx = np.random.choice(len(X_vic), n_sample)
    dataset = utils.SubTrainDataset(X_vic[idx], list(y_vic[idx]), transform=transforms.ToTensor())
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, help='dataset: cifar10, cifar100, tinyimagenet')
    parser.add_argument('-mc_n_sample', type=int, default=100)
    parser.add_argument('-ratio_lr', type=float, required=True, help='0.01, 0.05, 0.1')
    parser.add_argument('-gpu_id', type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    loader = sample_intersect1_loader(args.dataset, args.mc_n_sample)

    # list of random data points' neighrbood
    x_neighborhood_list = []
    for sample, y in loader:
        x_neighborhood_list.append(sample)

    print("Initialized neighborhoods.")

    dataset_abrev = dataset_abrev_dict[args.dataset]
    model_type = vic_dict[args.dataset]
    ablation_path = os.path.join(settings.CHECKPOINT_PATH, 
    'ablation', '{}abla_{}_{}'.format(model_type, dataset_abrev, args.ratio_lr))

    epoch_list = epochs_dict[args.dataset]
    inter_names = os.listdir(ablation_path)
    # mean and std for each intersection
    mean_std_dict = utils.get_intersection_mean_std_dict(args.dataset)

    result_path = os.path.join(settings.RESULT_PATH, "model_distance")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # Load models into model_dict
    model_predict = {}
    for intersection_folder in inter_names:

        mean, std = mean_std_dict[intersection_folder.replace('_', '')]
        intersect_dict = {}
        
        for i in range(10):
            copy_folder = 'copy_{}'.format(i)
            copy_dict = {}
            models_dict = {}
            for epoch in epoch_list:
                net = utils.get_network(args.dataset, model_type, False)
                net.load_state_dict(
                    torch.load(os.path.join(ablation_path, 
                    intersection_folder, copy_folder, 'model_{}.pth'.format(epoch)), 
                    map_location='cpu'))
                net.to('cpu')
                net.eval()
                models_dict[epoch] = nn.Sequential(transforms.Normalize(mean, std), net)

            print("Initialized model {}/{}.".format(intersection_folder, copy_folder))

        
            with torch.no_grad():
                for epoch, model in models_dict.items():
                    tmp = []
                    model.cuda()
                    for neighborhood in x_neighborhood_list:
                        tmp.append(model(neighborhood.cuda()).softmax(dim=1))
                    copy_dict[epoch] = torch.cat(tmp, dim=0).detach().cpu()
                    del model
                    torch.cuda.empty_cache()

            
            intersect_dict[copy_folder] = copy_dict
        model_predict[intersection_folder] = intersect_dict


    pickle.dump(model_predict, open(
            os.path.join(result_path, f"{args.dataset}_predict_ablation_{args.ratio_lr}.pkl"), "wb"))


