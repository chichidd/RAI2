import numpy as np
from tqdm import tqdm

from sklearn import datasets
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# import seaborn as sns


def return_cumsum_mean_var(Y_array):
    Y_cumsum = np.cumsum(Y)
    idx_cunsum = np.arange(1, len(Y)+1)

    mean_cumsum = Y_cumsum / idx_cunsum

    var_cumsum = np.cumsum((Y - mean_cumsum) **2)
    var_cumsum = var_cumsum[1:] / idx_cunsum[:-1]
    return idx_cunsum, mean_cumsum, var_cumsum


if __name__ == '__main__':
    # X, y = fetch_california_housing(return_X_y=True)
    X, y = datasets.load_diabetes(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    clf_list = []
    for i in range(50):
        clf = SGDRegressor(learning_rate='constant', eta0=0.02, random_state=2022+i)
        clf.fit(X, y)
        clf_list.append(clf)

    clf = SGDRegressor(learning_rate='constant', eta0=0.01, random_state=42)
    clf.fit(X, y)
    clf_finetune = [clf]
    clf.set_params(**{'eta0':0.0005})
    for i in range(50):
        clf_f = deepcopy(clf)
        np.random.seed(i)
        select_idx = np.random.permutation(len(X))[:int(0.1 * len(X))]
        clf_f.partial_fit(X[select_idx], y[select_idx])
        print(clf.coef_)
        clf_finetune.append(clf_f)

        # clf.partial_fit(X, y)
        # clf_finetune.append(deepcopy(clf))

    point_clouds_indep = []
    np.random.seed(2022)
    for clf in clf_list:
        tmp = np.random.randn(1000, X.shape[1])
        Y = clf.predict(tmp)
        print(f"{np.mean(Y):.2f}, {np.var(Y):.2f}, {float(clf.intercept_):.2f}, {np.sum(clf.coef_**2):.2f}")
        print(f"{clf.score(X, y):.2f}")
        point_clouds_indep.append([clf.intercept_, np.sum(clf.coef_**2)])
    point_clouds_indep = np.array(point_clouds_indep)

    print('*'*50)
    point_clouds_finetune = []
    for clf in clf_finetune:
        tmp = np.random.randn(1000, X.shape[1])
        Y = clf.predict(tmp)
        print(f"{np.mean(Y):.2f}, {np.var(Y):.2f}, {float(clf.intercept_):.2f}, {np.sum(clf.coef_**2):.2f}")
        print(f"{clf.score(X, y):.2f}")
        point_clouds_finetune.append([clf.intercept_, np.sum(clf.coef_**2)])
    point_clouds_finetune = np.array(point_clouds_finetune)

    Y = clf.predict(np.random.randn(1000, X.shape[1]))
    idx_range, cumsum_mean, cumsum_var = return_cumsum_mean_var(Y)





    sns.set_theme()
    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(6, 2.7)
    # linestyles = ['-', '--', ':', '-.']
    cumsum_std = np.sqrt(np.append(cumsum_var, cumsum_var[-1]))

    ax1.plot(idx_range, cumsum_mean - clf.intercept_, c='b', label="Mean", linestyle='--')
    ax1.plot(idx_range, cumsum_std - np.sqrt(np.sum(clf.coef_**2)), c='r', label="Std")
    ax1.set_title("(a) Error of MC methods.")
    ax1.legend(prop={'size': 10})
    ax1.set_ylabel("Estimation error")


    ax3.scatter(point_clouds_indep[:, 0], point_clouds_indep[:, 1], 
                marker='o', c='r', s=5, label="Independent")
    ax3.scatter(point_clouds_finetune[:, 0], point_clouds_finetune[:, 1], 
                marker='x', c='b', s=10, label='Finetuning')
    ax3.set_ylabel("$||\mathbf{W}||_2^2$")
    ax3.set_xlabel("$b$")
    ax3.set_title("(b) Distribution of tuples ($b$, $||\mathbf{W}||_2^2$).")
    ax3.set_xticks([]) 
    ax3.set_yticks([]) 

    ax3.legend(loc='upper right', bbox_to_anchor=(0.65, 1.02),
          ncol=1, prop={'size': 10})#, fancybox=True, shadow=True)
    plt.tight_layout()

    plt.savefig('cumsum.pdf', bbox_inches='tight')

    # plt.savefig('cumsum.png', bbox_inches='tight')










    # sns.set_theme()
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # fig.set_size_inches(8.2, 2)
    # # linestyles = ['-', '--', ':', '-.']
    
    # ax1.plot(idx_range, cumsum_mean, c='b')
    # ax1.axhline(y=clf.intercept_, color='r', linestyle='--', label="True value")
    # ax1.set_title("(a) Mean by MC")
    # ax1.legend(prop={'size': 7})
    # # ax1.grid(True)

    # ax2.plot(idx_range[1:], cumsum_var, c='b')
    # ax2.axhline(y=np.sum(clf.coef_**2), color='r', linestyle='--', label="True value")
    # ax2.set_title("(b) Variance by MC")
    # ax2.legend(prop={'size': 7})
    # # ax2.grid(True)

    # ax3.scatter(point_clouds_indep[:, 0], point_clouds_indep[:, 1], 
    #             marker='o', c='r', s=2, label="Independent")
    # ax3.scatter(point_clouds_finetune[:, 0], point_clouds_finetune[:, 1], 
    #             marker='x', c='b', s=2, label='Finetune')
    # ax3.set_ylabel("Variance")
    # ax3.set_xlabel("Mean")
    # ax3.set_title("(c) Pairs of mean and variance")
    # ax3.set_xticks([]) 
    # ax3.set_yticks([]) 
    # ax3.legend(loc='upper right', bbox_to_anchor=(1.65, 1.02),
    #       ncol=1, prop={'size': 7})#, fancybox=True, shadow=True)
    # plt.tight_layout()

    # plt.savefig('cumsum.pdf', bbox_inches='tight')
    # # plt.savefig('cumsum.png', bbox_inches='tight')















# 'solid', 'solid'),      # Same as (0, ()) or '-'
#      ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
#      ('dashed', 'dashed'),    # Same as '--'
#      ('dashdot', 'dashdot')]  # Same as '-.'

