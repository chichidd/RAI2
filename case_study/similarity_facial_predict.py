import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import argparse
import sys
sys.path.append("../")
from tqdm import tqdm
import pickle
import utils 
from conf import settings

# Example commands:
#  python similarity_facial_predict.py -case same_dist
#  python similarity_facial_predict.py -case diff_dist
# python similarity_facial_predict.py -case large_adv_dst_same_dist -dst_ratio 2 
# python similarity_facial_predict.py -case large_adv_dst_same_dist -dst_ratio 10
# python similarity_facial_predict.py -case large_adv_dst_diff_dist -dst_ratio 10
target_s_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
np.random.seed(0)
torch.manual_seed(0)

def verification_set(n_sample=100, dst_ratio=1):
    X_tensor, y_tensor = pickle.load(open(os.path.join(settings.DATA_PATH, "facial_attribute", "fairface_set1_tensor.pkl"), "rb"))
    X_tensor, y_tensor = X_tensor[::dst_ratio], y_tensor[::dst_ratio]
    idx = []
    chunk_idx_list = list(torch.chunk(torch.arange(len(X_tensor)), len(target_s_list) - 1))
    np.random.seed(0)
    for chunk_idx in chunk_idx_list:
        idx.append(np.random.choice(chunk_idx.numpy(), n_sample // (len(target_s_list) - 1), replace=False))
    idx = np.concatenate(idx)
    return idx, X_tensor[idx], len(torch.unique(y_tensor)), y_tensor[idx]

def process(data_type, dst_ratio=1):
    '''
    Prepare the mean and std of dataset used for model standarization (or normalization).
    see https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
    '''
    mean_std_dict = {}
    X_set1, _ = pickle.load(open(os.path.join(settings.DATA_PATH, "facial_attribute", "fairface_set1_tensor.pkl"), "rb"))
    X_set1 = X_set1[::dst_ratio]
    set_num = len(X_set1)
    if data_type == 'diff_dist':
        X_set2, _ = pickle.load(open(
            os.path.join(settings.DATA_PATH, "facial_attribute", "utk_tensor.pkl"), "rb"))
        X_set2 = X_set2[::dst_ratio]

    else:
        # if the data are from same distribution, we neglect here the case where adversary's own data are sampled from unrelated data pool for simplicty. Note that this case is equivalent to fixed sampling of adversary's own data.
        X_set2, _ = pickle.load(open(
            os.path.join(settings.DATA_PATH, "facial_attribute", "fairface_set_rest_tensor.pkl"), "rb"))
        X_set2 = X_set2[::dst_ratio][:set_num]
    
    for s in target_s_list:
        shift = int(s * set_num)
        X_tensor = torch.cat([X_set1[set_num - shift:], X_set2[:set_num - shift]])
        mean_std_dict['int{}'.format(s)] = (X_tensor.mean(dim=[0, 2, 3]), X_tensor.std(dim=[0, 2, 3]))
    
    if data_type == 'same_dist':
        mean_std_dict['vic'] = mean_std_dict['int1.0']

    
    return mean_std_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-case', type=str, required=True, help='same_dist, diff_dist, low_epoch, gauss, color, gauss_color, large_adv_dst_same_dist, large_adv_dst_diff_dist, small_adv_dst')
    parser.add_argument('-dst_ratio', type=int, default=1, help='shrink the dataset size by dst_ratio times. Default values are 2 and 10.')
    parser.add_argument('-mc_n_sample', type=int, default=100)
    parser.add_argument('-gpu_id', type=int, default=0, help='device id')
    parser.add_argument('-epoch_range', type=float, default=1.0, help='epoch ratio comparing to default setting')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    
    if args.case in ['large_adv_dst_same_dist', 'large_adv_dst_diff_dist']:
        # in this case, we load compute the model output 
        # obtain verification set, and mean and std for model standarizaiton
        select_idx, verify_tensor, num_classes, verify_label = verification_set(args.mc_n_sample, args.dst_ratio)
    else:
        select_idx, verify_tensor, num_classes, _ = verification_set(args.mc_n_sample, 1)


    if args.case == 'same_dist':
        file_path = settings.CASE_STUDY_CHECKPOINT_PATH
        model_type = "resnet101"
        target_s_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        folder_names = [f"{model_type}_facial_same_dist_{s}" for s in target_s_list]
        folder_names.append("{}_facial_vic".format(model_type))
        target_s_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        mean_std_dict = process("same_dist", args.dst_ratio)

    elif args.case == 'large_adv_dst_same_dist':
        file_path = os.path.join(settings.CASE_STUDY_CHECKPOINT_PATH, 'change_dst_size')
        model_type = "resnet101"
        folder_names = [f"{model_type}_facial_same_dist_dst_ratio{args.dst_ratio}_{s}" for s in target_s_list]
        folder_names.append("{}_facial_vic".format(model_type))
        mean_std_dict = process("same_dist", args.dst_ratio)
        # if the adversary has more data (i.e., we shrink the victim dataset by dst_ratio), 
        # the prepared model on whole victim dataset (fairface_set1) can also be used as the model should also be confident on verification samples.
        # We neglect the change to standarization mean and std caused by dataset size reduction because of small difference
    elif args.case == 'large_adv_dst_diff_dist':
        file_path = settings.CASE_STUDY_CHECKPOINT_PATH
        model_type = "regnet_y_8gf"
        folder_names = [f"{model_type}_facial_diff_dist_{s}" for s in target_s_list]
        mean_std_dict = process("diff_dist", 1)
    elif args.case == 'diff_dist':
        file_path = settings.CASE_STUDY_CHECKPOINT_PATH
        model_type = "regnet_y_8gf"
        target_s_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        folder_names = [f"{model_type}_facial_diff_dist_{s}" for s in target_s_list]
        mean_std_dict = process("diff_dist", args.dst_ratio)
        target_s_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        mean_std_dict = process("diff_dist", args.dst_ratio)

    elif args.case == 'low_epoch':
        epochs = int(settings.CASE_STUDY_EPOCH * args.epoch_range)
        file_path = os.path.join(settings.CASE_STUDY_CHECKPOINT_PATH, 'low_epoch', f'epoch{epochs}')
        model_type = "regnet_y_8gf"
        folder_names = [f"{model_type}_facial_diff_dist_{s}" for s in target_s_list]
        mean_std_dict = process("diff_dist", args.dst_ratio)

    elif args.case in ['gauss', 'color', 'gauss_color']:
        file_path = os.path.join(settings.CASE_STUDY_CHECKPOINT_PATH, 'adaptive_trans')
        model_type = "regnet_y_8gf"
        folder_names = [f"{model_type}_adaptive_{args.case}_facial_diff_dist_{s}" for s in target_s_list]
        mean_std_dict = process("diff_dist", args.dst_ratio)

    elif args.case == 'small_adv_dst':
        # in this case, we reduce the adversary's size by dst_ratio, the victim's models (and prepared ones) remain same
        file_path = os.path.join(settings.CASE_STUDY_CHECKPOINT_PATH, 'change_dst_size')
        model_type = "regnet_y_8gf"
        if args.dst_ratio == 10:
            target_s_list = [0.0, 1.0]
        folder_names = [f"{model_type}_facial_diff_dist_dst_ratio{args.dst_ratio}_{s}" for s in target_s_list]
        mean_std_dict = process("diff_dist", args.dst_ratio)

    result_path = os.path.join(settings.CASE_STUDY_RESULT_PATH , "dataset_similarity")
    if not os.path.exists(result_path):
        os.mkdir(result_path)



    verify_tensor = verify_tensor.cuda()
    verify_label = verify_label.cuda()
    model_outputs_dict = {}
    for folder in folder_names:
        inter_propor = f'int{folder.split("_")[-1]}'
        if inter_propor.endswith('vic'):
            mean, std = mean_std_dict['int1.0']
            inter_propor = 'vic'
        else:
            mean, std = mean_std_dict[inter_propor]
        models_list = []
        model_path = os.path.join(file_path, folder)
        for model_file_name in os.listdir(model_path):

            net = getattr(models, model_type)(num_classes=num_classes)
            if args.case in ['gauss', 'color', 'gauss_color']:
                net_dicts = torch.load(os.path.join(model_path, model_file_name) , map_location='cpu')
                net.load_state_dict(net_dicts['net_sd'])
                net.to('cpu')
                net.eval()
                models_list.append(nn.Sequential(transforms.Normalize(*(net_dicts['mean_std'])), net))
            else:
                net.load_state_dict(torch.load(os.path.join(model_path, model_file_name), map_location='cpu'))
                net.to('cpu')
                net.eval()
                models_list.append(nn.Sequential(transforms.Normalize(mean, std), net))

        print(f"Initialized models in {folder}.")

        if args.case == 'small_adv_dst':
            folder_name_s = float(folder.split("_")[-1])
            folder_name_s /= args.dst_ratio
            inter_propor = f'int{folder_name_s}'
        
        with torch.no_grad():
            model_predicts = []
            for model in models_list:
                model.cuda()
                model_predicts.append(model(verify_tensor).softmax(dim=1))
                torch.cuda.empty_cache()
            model_outputs_dict[inter_propor] = torch.stack(model_predicts).detach().cpu()
    
    pickle.dump(model_outputs_dict, open(
        os.path.join(settings.CASE_STUDY_RESULT_PATH , "dataset_similarity", f"{args.case}{f'{args.dst_ratio}' if args.dst_ratio != 1 else ''}.pkl"), "wb"))
    
    if args.case == 'same_dist':
        # Evaluate the heuristic
        shard_dict = {}
        shard_dict['int1.0'] = model_outputs_dict['int1.0']
        shard_dict['int0.0'] = model_outputs_dict['int0.0']

        shard_size = len(verify_tensor) // (len(target_s_list) -1)
        
        for start_idx in range(1, 5):
            tmp_list = []
            for i in range(10):
                tmp = torch.zeros_like(model_outputs_dict['int0.0'][i])
                tmp[:int(start_idx * shard_size)] = model_outputs_dict['int0.0'][i][:int(start_idx * shard_size)]
                tmp[int(start_idx * shard_size):] = model_outputs_dict['int1.0'][i][int(start_idx * shard_size):]
                tmp_list.append(tmp)

            shard_dict['int{:.1f}'.format(1.0 - 0.2 * start_idx)] = torch.stack(tmp_list)
        
        pickle.dump(shard_dict, open(
            os.path.join(settings.CASE_STUDY_RESULT_PATH , "dataset_similarity", f"{args.case}_shards.pkl"), "wb"))