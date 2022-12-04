import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss-based MIA

def return_losses(net, trainloader, testloader, device):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(reduce=False).to(device)
        train_losses = []
        for x, y in trainloader:
            pred = net(x.to(device))
            train_losses.extend(criterion(pred, y.to(device)).tolist())
        test_losses = []
        for x, y in testloader:
            pred = net(x.to(device))
            test_losses.extend(criterion(pred, y.to(device)).tolist())
        return np.array(train_losses), np.array(test_losses)

def mi_success(train_memberships, test_memberships, print_details=True):
    tp = np.sum(train_memberships)
    fp = np.sum(test_memberships)
    fn = len(train_memberships) - tp
    tn = len(test_memberships) - fp

    # yeom's membership inference advantage
    acc = 100 * (tp + tn) / (tp + fp + tn + fn)
    advantage = 2*(acc - 50)

    if print_details:
        precision = 100*(tp/(tp+fp)) if (tp+fp) > 0 else 0
        recall = 100*(tp/(tp+fn)) if (tp+fn) > 0 else 0
        print('Adversary Advantage: {0:.3f}%, Accuracy: {1:.3f}%, Precision : {2:.3f}%, Recall: {3:.3f}%'.format(advantage,  acc, precision, recall))
        print('In training: {}/{}, In testing: {}/{}'.format(tp, len(train_memberships), tn, len(test_memberships)))
    return advantage

# YEOM et all's membership inference attack using pred loss
def yeom_mi_attack(losses, avg_loss):
    memberships = (losses < avg_loss).astype(int)
    return memberships


def yeom_w_get_best_threshold(train_losses, test_losses):    
    advantages = []

    mean_loss = np.mean(train_losses)
    std_dev = np.std(train_losses)

    coeffs = np.linspace(-5, 5, num=1001, endpoint=True)

    for coeff in coeffs:
        cur_threshold = mean_loss + std_dev*coeff
        cur_yeom_mi_advantage = mi_success(yeom_mi_attack(train_losses, cur_threshold),  yeom_mi_attack(test_losses, cur_threshold), print_details=False)
        advantages.append(cur_yeom_mi_advantage)
    best_threshold = mean_loss + std_dev*coeffs[np.argmax(advantages)]
    return best_threshold

def apply_best_attacks(train_losses, test_losses, select_num=1000):
    np.random.seed(0)
    train_in_atk = np.random.choice(len(train_losses), select_num * 2, replace=False)
    test_in_atk = np.random.choice(len(test_losses), select_num * 2, replace=False)
    train_in_atk_train_idx, train_in_atk_test_idx = train_in_atk[:select_num], train_in_atk[select_num:]
    test_in_atk_train_idx, test_in_atk_test_idx = test_in_atk[:select_num], test_in_atk[select_num:]

    best_threshold = yeom_w_get_best_threshold(train_losses[train_in_atk_train_idx], test_losses[test_in_atk_train_idx])
    best_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], best_threshold)
    best_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], best_threshold)
    best_yeom_mi_advantage = mi_success(best_train_memberships, best_test_memberships, print_details=False)
    best_results = (best_threshold, best_train_memberships, best_test_memberships, best_yeom_mi_advantage)

    return best_results