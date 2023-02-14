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
COPY_NUM = 10

np.random.seed(0)
torch.manual_seed(0)



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
    parser.add_argument('-gpu_id', type=int, default=0, help="GPU device for inference")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    loader = sample_intersect1_loader(args.dataset, args.mc_n_sample)
    verify_samples = []
    for sample, y in loader:
        verify_samples.append(sample)
    verify_tensor = torch.cat(verify_samples)
    print("Initialized verificaiton set.")

    file_path = os.path.join(settings.CHECKPOINT_PATH, 'similarity', args.dataset)
    model_type_list = os.listdir(file_path)
    inter_names = os.listdir(os.path.join(file_path, model_type_list[0]))
    # mean and std for each intersection
    mean_std_dict = utils.get_intersection_mean_std_dict(args.dataset)

    result_path = os.path.join(settings.RESULT_PATH, "dataset_similarity")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    # Load models into model_dict
    all_model_dict = {}
    for model_type in model_type_list:
        print(model_type)

        model_predict_on_neighrboods = {}
        for intersection in inter_names:
            mean, std = mean_std_dict[intersection]
            models_list = []
            for i in range(COPY_NUM):
                net = utils.get_network(args.dataset, model_type, False)
                net.load_state_dict(
                    torch.load(
                    os.path.join(file_path, model_type, intersection, 'model_{}.pth'.format(i)), 
                    map_location='cpu'))
                net.to('cpu')
                net.eval()
                models_list.append(nn.Sequential(transforms.Normalize(mean, std), net))
            print("Initialized model {}/{}.".format(model_type, intersection))

            
            with torch.no_grad():
                model_predicts = []
                for model in models_list:
                    model.cuda()
                    model_predicts.append(model(verify_tensor.cuda()).softmax(dim=1))
                    del model
                    torch.cuda.empty_cache()
                model_predict_on_neighrboods[intersection] = torch.stack(model_predicts).detach().cpu()
                
        pickle.dump(model_predict_on_neighrboods, 
                open(os.path.join(result_path, f"{args.dataset}_predict_{model_type}.pkl"), "wb"))


