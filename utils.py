""" helper function
author baiyu
"""
import os
import sys
import re
import time
import numpy as np
from conf import settings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report




###########################
####### BEGIN SELF ########
###########################
def leave_topk(output, k=5, fillzero=False):
    '''
    output of shape batch_size * num_classes
    '''
    num_classes = output.shape[1]
    values, index = output.topk(dim=1, k=k, largest=True, sorted=True)
    onehot_index = F.one_hot(index, num_classes=num_classes).sum(dim=1).bool()
    index = torch.sort(index, dim=1)[0]
    values = torch.gather(output, dim=1, index=index)
    if fillzero:
        output1 = torch.zeros_like(output)
    else:
        fill_values = (1-values.sum(dim=1))/(num_classes - k)
        output1 = fill_values.reshape(-1, 1).expand(output.shape[0], output.shape[1]).clone()
    output1[onehot_index] = values.reshape(-1)
    return output1

def shannon_entropy(X):
    '''
    X of shape: n*m.
    '''
    return - torch.sum(torch.log(X)*X, dim=1)

def fast_load_model(name, folder, num_classes, device, model_file='model.pth', norm_mean=(0, 0, 0), norm_std=(1, 1, 1), return_path=False):
    '''
    auxillary function for fast model loading, and combine the model with training standarization mean and std.
    '''
    path = "./checkpoint/{}/{}/".format(name, folder)
    net = get_network(name, False, num_classes=num_classes).to(device) 
    net.load_state_dict(torch.load(path + model_file, map_location=device))
    net.eval()
    net = nn.Sequential(transforms.Normalize(norm_mean, norm_std), net)
    if return_path:
        return net, path
    return net


def evaluate_net(net, test_loader, device):
    '''
    similar to the next one.
    '''
    net.eval()
    correct_1 = 0.0
    correct_5 = 0.0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            # print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))
            image = image.to(device)
            label = label.to(device)
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            #compute top 5
            correct_5 += correct[:, :5].sum()
            #compute top1
            correct_1 += correct[:, :1].sum()

    # print()
    # print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    # print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    # print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    return correct_1.item(), correct_5.item()

# model evaluation
def compute_accuracy(net, testloader, device):
    '''
    as the name.
    '''
    net.eval()
    pred = []
    true = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            pred += net(images).argmax(dim=1).tolist()
            true += labels.tolist()
    return accuracy_score(true, pred), pred, true

### train functions
#train the model
def train_mlp(net, trainloader, testloader, n_epochs, optimizer, criterion, device):
    '''
    used for train meta MLP.
    '''
    for epoch in range(1, n_epochs+1):
        net.train()
        running_loss = 0.
        train_pred = []
        train_true = []
        for i, (inputs, labels) in enumerate(trainloader, 1):
            train_true += labels.tolist()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            train_pred += outputs.argmax(dim=1).tolist()

        # Print accuracy after every epoch
        accuracy, pred, true = compute_accuracy(net, testloader, device)
        print('Epoch {}. Train loss: {:.2f}. Train acc: {:.2f}. Test acc: {:.2f}.'.format(
            epoch, 
            running_loss / len(trainloader.dataset),
            100 * accuracy_score(train_true, train_pred),
            100 * accuracy ))

    print('Finished Training')
    return accuracy

###########################
######## END SELF #########
###########################


###########################
####### BEGIN NEW #########
###########################

class SubTrainDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # TODO: unify the following line, in case study, the below line does not exist.
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)

# methods for loading sub-datasets of CIFAR-10/100 and Tiny-ImageNet


def get_dataset_hyperparam(dataset):
    if dataset == 'cifar10':
        return settings.CIFAR10_EPOCH, settings.CIFAR10_MILESTONES
    if dataset == 'cifar100':
        return settings.CIFAR100_EPOCH, settings.CIFAR100_MILESTONES
    if dataset == 'tinyimagenet':
        return settings.TINYIMAGENET_EPOCH, settings.TINYIMAGENET_MILESTONES

def get_dataset_mean_std(dataset):
    if dataset == 'cifar10':
        return settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD
    if dataset == 'cifar100':
        return settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD
    if dataset == 'tinyimagenet':
        return settings.TINYIMAGENET_TRAIN_MEAN, settings.TINYIMAGENET_TRAIN_STD
        
def get_intersection_mean_std_dict(dataset_name):
    '''
    get normalization mean and std for each intersections 0.0, 0.1, ..., 0.9, 1.0. used for evaluation.
    '''
    mean_std_dict = {}
    for s in (np.arange(11) / 10):
        Set1, Set2 = pickle.load(open(os.path.join(settings.DATA_PATH, f'similarity/{dataset_name.upper()}_intersect_{s}.pkl'), 'rb'))
        mean = tuple((Set2[0] / 255).mean(axis=(0, 1, 2)))
        std = tuple((Set2[0] / 255).std(axis=(0, 1, 2)))
        mean_std_dict['int{}'.format(s)] = (mean, std)
    mean_std_dict['vic'] = mean_std_dict['int1.0']
    return mean_std_dict

def get_subtraining_dataloader_cifar10_intersect(propor=0.5, batch_size=16, num_workers=8, shuffle=True, sub_idx=1):

    
    X_set, y_set = pickle.load(open(os.path.join(settings.DATA_PATH, f'similarity/CIFAR10_intersect_{propor}.pkl'), 'rb'))[sub_idx]
    mean = tuple((X_set / 255).mean(axis=(0, 1, 2)))
    std = tuple((X_set / 255).std(axis=(0, 1, 2)))
    
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader, mean, std

def get_subtraining_dataloader_cifar100_intersect(propor=0.5, batch_size=16, num_workers=8, shuffle=True, sub_idx=1):

    
    X_set, y_set = pickle.load(open(os.path.join(settings.DATA_PATH, f'similarity/CIFAR100_intersect_{propor}.pkl'), 'rb'))[sub_idx]
    mean = tuple((X_set / 255).mean(axis=(0, 1, 2)))
    std = tuple((X_set / 255).std(axis=(0, 1, 2)))
    
    
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader, mean, std

def get_subtraining_dataloader_tinyimagenet_intersect(propor=0.5, batch_size=16, num_workers=8, shuffle=True, sub_idx=1):

    X_set, y_set = pickle.load(open(os.path.join(settings.DATA_PATH, 'similarity/TINYIMAGENET_intersect_{propor}.pkl'), 'rb'))[sub_idx]
    mean = tuple((X_set / 255).mean(axis=(0, 1, 2)))
    std = tuple((X_set / 255).std(axis=(0, 1, 2)))
    
    
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tinyimagenet_training = SubTrainDataset(X_set, list(y_set), transform=transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return tinyimagenet_training_loader, mean, std

def get_intersect_dataloader(dataset, propor, batch_size=16, num_workers=8, shuffle=True, sub_idx=1):
    if dataset == 'cifar10':
        return get_subtraining_dataloader_cifar10_intersect(propor=propor, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sub_idx=sub_idx)
    elif dataset == 'cifar100':
        return get_subtraining_dataloader_cifar100_intersect(propor=propor, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sub_idx=sub_idx)
    elif dataset == 'tinyimagenet':
        return get_subtraining_dataloader_tinyimagenet_intersect(propor=propor, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sub_idx=sub_idx)




def get_training_dataloader_cifar10(mean, std, batch_size=16, num_workers=8, shuffle=True):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_training = torchvision.datasets.CIFAR10(root=os.path.join(settings.DATA_PATH, 'CIFAR10'), train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_training_dataloader_cifar100(mean, std, batch_size=16, num_workers=8, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root=os.path.join(settings.DATA_PATH, 'CIFAR100'), train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_training_dataloader_tinyimagenet(mean, std, batch_size=16, num_workers=8, shuffle=True):

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tinyimagenet_training = datasets.ImageFolder(os.path.join(settings.DATA_PATH, 'tiny-imagenet-200/train/'), transform=transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return tinyimagenet_training_loader

def get_training_dataloader(dataset, mean, std, batch_size=16, num_workers=8, shuffle=True):
    if dataset == 'cifar10':
        return get_training_dataloader_cifar10(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    if dataset == 'cifar100':
        return get_training_dataloader_cifar100(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    if dataset == 'cifar10':
        return get_training_dataloader_tinyimagenet(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)




def get_test_dataloader_cifar10(mean, std, batch_size=16, num_workers=8, shuffle=False, pin_memory=True):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root=os.path.join(settings.DATA_PATH, 'CIFAR10'), train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)

    return cifar10_test_loader

def get_test_dataloader_cifar100(mean, std, batch_size=16, num_workers=8, shuffle=True, pin_memory=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root=os.path.join(settings.DATA_PATH, 'CIFAR100'), train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

    return cifar100_test_loader

def get_test_dataloader_tinyimagenet(mean, std, batch_size=16, num_workers=8, shuffle=True, pin_memory=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    X_set, y_set = pickle.load(open(os.path.join(settings.DATA_PATH, 'TinyImagenet_test.pkl'), 'rb'))
    tinyimagenet_test = SubTrainDataset(X_set, list(y_set), transform=transform_test)
    tinyimagenet_test_loader = DataLoader(
        tinyimagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    return tinyimagenet_test_loader

def get_test_dataloader(dataset, mean, std, batch_size=16, num_workers=8, shuffle=True, pin_memory=True):
    if dataset == 'cifar10':
        return get_test_dataloader_cifar10(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    elif dataset == 'cifar100':
        return get_test_dataloader_cifar100(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    if dataset == 'tinyimagenet':
        return get_test_dataloader_tinyimagenet(mean=mean, std=std, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)

######################################################################################
# methods for loading sub-datasets for case-study of facial attribute classification.#
######################################################################################
def get_subtraining_dataloader_facial_intersect(propor=0.6, batch_size=16, num_workers=8, shuffle=True):

    X_tensor, y_tensor = pickle.load(open(os.path.join(settings.DATA_PATH, 'facial_attribute', 'fairface_similarity', f'intersect_{propor}.pkl'), 'rb'))
    mean = X_tensor.mean(dim=[0, 2, 3])
    std = X_tensor.std(dim=[0, 2, 3])
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tinyimagenet_training = SubTrainDataset(X_tensor, list(y_tensor), transform=transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, 
        num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return tinyimagenet_training_loader, mean, std, len(torch.unique(y_tensor))

def get_subtraining_dataloader_facial_mix(propor=0.6, batch_size=16, num_workers=8, shuffle=True):

    X_tensor, y_tensor = pickle.load(open(os.path.join(settings.DATA_PATH, 'facial_attribute', 'fairface_utk_mix', f'/intersect_{propor}.pkl'), 'rb'))
    mean = X_tensor.mean(dim=[0, 2, 3])
    std = X_tensor.std(dim=[0, 2, 3])
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        # transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tinyimagenet_training = SubTrainDataset(X_tensor, list(y_tensor), transform=transform_train)
    tinyimagenet_training_loader = DataLoader(
        tinyimagenet_training, shuffle=shuffle, 
        num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return tinyimagenet_training_loader, mean, std, len(torch.unique(y_tensor))


def get_facial_dataloader(inter_propor=0.6, batch_size=16, num_workers=8, shuffle=True, same_data_dist=True, adaptive_trans="gauss_color", dst_ratio=1, seed=None):
    ''' 
    same_data_dist means adversary data is of same distribution as the victim data, diff means the adversary data is of the different distribution.
    adaptve is a list, test for adaptive attacks
    '''
    X_tensor_list, y_tensor_list = [], []
    fair_train_img_set_tensor, fair_train_label_set = pickle.load(open(os.path.join(settings.DATA_PATH, "facial_attribute", "fairface_set1_tensor.pkl"), "rb"))
    # we first shrink the dataset (reduce the dataset size if dst_ratio is other than 1)
    fair_train_img_set_tensor = fair_train_img_set_tensor[::dst_ratio]
    fair_train_label_set = fair_train_label_set[::dst_ratio]
    set_num = len(fair_train_img_set_tensor)
    shift = int(inter_propor * set_num)

    print(f"Sample id {set_num - shift} to id {set_num} from set1.")
    X_tensor_list.append(fair_train_img_set_tensor[set_num - shift:])
    y_tensor_list.append(fair_train_label_set[set_num - shift:])
    

    if same_data_dist:
        print("The adversary has data of same distribution: sampling from FairFace data as the adversary data.")
        fair_train_img_set_tensor_unrelated, fair_train_label_set_unrelated = pickle.load(open(os.path.join(settings.DATA_PATH, "facial_attribute", "fairface_set_rest_tensor.pkl"), "rb"))
        if seed is not None:
            # random sampling from unrelated data of same distribution
            np.random.seed(seed)
            idx = np.random.permutation(len(fair_train_img_set_tensor_unrelated))[:set_num - shift]
            # Here, the set_num is based on (shrinked, if dst_ratio != 1) dataset size, so we only randomly select the equal number of samples from unrelated data pool.
            X_tensor_list.append(fair_train_img_set_tensor_unrelated[idx])
            y_tensor_list.append(fair_train_label_set_unrelated[idx])
        else:
            X_tensor_list.append(fair_train_img_set_tensor_unrelated[::dst_ratio][:set_num - shift])
            y_tensor_list.append(fair_train_label_set_unrelated[::dst_ratio][:set_num - shift])
    else:
        print("The adversary has data of different distribution: sampling from UTK data as the adversary data.")
        utk_train_img_set_tensor, utk_train_label_set = pickle.load(open(os.path.join(settings.DATA_PATH, "facial_attribute", "utk_tensor.pkl"), "rb"))
        X_tensor_list.append(utk_train_img_set_tensor[::dst_ratio][:set_num - shift])
        y_tensor_list.append(utk_train_label_set[::dst_ratio][:set_num - shift])
    
    print(f"0 to {set_num - shift} from set2")
    X_tensor, y_tensor = torch.cat(X_tensor_list), torch.cat(y_tensor_list)

    if not same_data_dist: # in this case, the adversary trains the model and can shift data's visual features by transformations.
        trans_list = []
        if 'gauss' in adaptive_trans:
            trans_list.append(transforms.GaussianBlur(kernel_size=15))
        if 'color' in adaptive_trans:
            trans_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
        if len(trans_list) > 0:
            trans = transforms.Compose(trans_list)
            torch.manual_seed(seed)
            for i in tqdm(range(len(X_tensor))):
                X_tensor[i] = trans(X_tensor[i])

    mean, std = X_tensor.mean(dim=[0, 2, 3]), X_tensor.std(dim=[0, 2, 3])
    transform_train = transforms.Compose([
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean, std)])
    tinyimagenet_training = SubTrainDataset(X_tensor, list(y_tensor), transform=transform_train)
    tinyimagenet_training_loader = torch.utils.data.DataLoader(
        tinyimagenet_training, shuffle=shuffle, 
        num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return tinyimagenet_training_loader, mean, std, len(torch.unique(y_tensor))



def get_test_dataloader_facial(mean, std, batch_size=16, num_workers=8, shuffle=True, pin_memory=True):
    transform_test = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    X_tensor, y_tensor = pickle.load(open(os.path.join(settings.DATA_PATH, 'facial_attribute', 'fairface_val_tensor.pkl'), 'rb'))
    tinyimagenet_test = SubTrainDataset(X_tensor, y_tensor, transform=transform_test)
    tinyimagenet_test_loader = DataLoader(
        tinyimagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)
    return tinyimagenet_test_loader





###########################
##### Load Networks #######
###########################

def get_num_classes(dataset):
    if dataset == 'cifar10':
        return 10
    if dataset == 'cifar100':
        return 100
    if dataset == 'tinyimagenet':
        return 200

def get_network_cifar(netname, gpu, num_classes):
    """ return given network, architectures are based on kangliu's training code.
    check https://github.com/kuangliu/pytorch-cifar.
    """
    if netname == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_classes=num_classes)
    elif netname == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_classes=num_classes)
    elif netname == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_classes=num_classes)
    elif netname == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_classes=num_classes)
    elif netname == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(num_classes=num_classes)
    elif netname == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(num_classes=num_classes)
    elif netname == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(num_classes=num_classes)
    elif netname == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(num_classes=num_classes)
    elif netname == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=num_classes)
    elif netname == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=num_classes)
    elif netname == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=num_classes)
    elif netname == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if gpu: #use_gpu
        net = net.cuda()

    return net

def get_network_torchvision(netname, gpu, num_classes):
    if netname == 'wideresnet101':
        from torchvision.models import wide_resnet101_2
        net = wide_resnet101_2(num_classes=num_classes)
    elif netname == 'densenet121':
        from torchvision.models import densenet121
        net = densenet121(num_classes=num_classes)
    elif netname == 'resnet152':
        from torchvision.models import resnet152
        net = resnet152(num_classes=num_classes)
    elif netname == 'resnet101':
        from torchvision.models import resnet101
        net = resnet101(num_classes=num_classes)
    elif netname == 'vgg19':
        from torchvision.models import vgg19_bn
        net = vgg19_bn(num_classes=num_classes)
    elif netname == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2
        net = mobilenet_v2(num_classes=num_classes)
    elif netname == 'wide_resnet101_2':
        from torchvision.models import wide_resnet101_2
        net = wide_resnet101_2(num_classes=num_classes)
        
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if gpu: #use_gpu
        net = net.cuda()

    return net

def get_network(dataset, netname, gpu):
    num_classes = get_num_classes(dataset)
    if dataset.startswith('cifar'):
        return get_network_cifar(netname, gpu, num_classes)
    elif dataset == 'tinyimagenet':
        return get_network_torchvision(netname, gpu, num_classes)


###########################
####### Training ##########
###########################





class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]



##For adversarial training####

def PGD(x, loss_fn, y=None, model=None, eps=None, steps=3, gamma=None):

    # convert to cuda...
    x_adv = x.clone().cuda()
    # create an adv. example w. random init
    x_rand = torch.rand(x_adv.shape).cuda()
    x_adv += (2.0 * x_rand - 1.0) * eps
    x_adv.requires_grad_(True)
    # run steps
    for t in range(steps):
        out_adv_branch = model(x_adv)   # use the main branch
        loss_adv = loss_fn(out_adv_branch, y)
        grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]

        x_adv.data.add_(gamma * torch.sign(grad.data))
        _linfball_projection(x, eps, x_adv, in_place=True)

        x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def _linfball_projection(center, radius, t, in_place=True):
    min_range = center - radius
    max_range = center + radius
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min_range
    res.data[idx] = min_range[idx]
    idx = res.data > max_range
    res.data[idx] = max_range[idx]
    return res



def train(epoch, net, training_loader, loss_function, optimizer, warmup_epoch=0, warmup_scheduler=None, adv_training=False, verbose=False):
    start = time.time()
    net.train()
    loss_epoch = 0
    correct = 0.0
    for batch_index, (images, labels) in enumerate(training_loader):
        labels, images = labels.cuda(), images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        if adv_training and (np.random.rand() > 0.9):
            b_advx = PGD(images, loss_function, y=labels, model=net, eps=128/255., steps=1, gamma=1/255.).data.cuda()
            loss += loss_function(net(b_advx), labels)
        loss_epoch += loss.item() * len(labels)
        correct += outputs.argmax(dim=1).eq(labels).sum().item()
        loss.backward()
        optimizer.step()

        if epoch <= warmup_epoch and warmup_scheduler is not None:
            warmup_scheduler.step()
        if verbose:
            print(f"Training Epoch: {epoch} [{batch_index * training_loader.batch_size + len(images)}/{len(training_loader.dataset)}]\tLoss: \
                {loss.item():0.4f}\tLR: {optimizer.param_groups[0]['lr']:0.6f}")
    finish = time.time()
    if verbose:
        print(f'Epoch {epoch} training time consumed: {finish - start:.2f}s')
    return loss_epoch / len(training_loader.dataset), correct / len(training_loader.dataset)





@torch.no_grad()
def eval_training(epoch, net, test_loader, loss_function, verbose=False):
    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if verbose:
        print(f'Test set: Epoch: {epoch}, Average loss: {test_loss / len(test_loader.dataset):.4f}, Accuracy: {correct.float() / len(test_loader.dataset):.4f}, Time consumed:{finish - start:.2f}s')
    return correct.float() / len(test_loader.dataset)


