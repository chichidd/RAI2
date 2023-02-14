# $\text{RAI}^2$ : Responsible Identity Audit Governing the Artificial Intelligence

This repository contains code for reproducing the results of our NDSS'23 paper "$\text{RAI}^2$ : Responsible Identity Audit Governing the Artificial Intelligence"
(based on the popular Pytorch [implementation](https://github.com/kuangliu/pytorch-cifar)). The code contains scripts to train the models and Jupyter notebooks to generate tables and figures.

Remember to cite if you find our repo useful :)

*Note:* The repo may not be bug-free as I tried to integrate multiple components into one repository, so there can be errors like "missing varaible". If you are only interested in the algorithm implementation, it will be easier to check out the notebooks.

```latex
@inproceedings{dong2023rai2,
    author={Tian Dong and Shaofeng Li and Guoxing Chen and Minhui Xue and Haojin Zhu and Zhen Liu},
    title={{RAI2:} Responsible Identity Audit Governing the Artificial Intelligence},
    booktitle = {30th Annual Network and Distributed System Security Symposium, {NDSS} 2023},
    publisher = {The Internet Society},
    year={2023}
}
```


## Environment setting

The code is tested on Python 3.10.8, Pytorch 1.13.0, CUDA 11.8.
The packages used in our project is provided in ```requirements.txt``` for reference.


## File and Folder Description

- `conf/global_settings.py`: parameters used for all scripts include paths of data and model, etc.
- `models/`: model architecture files.
- `dataset_preparation/`: jupyter notebooks for preprocess datasets.
- `dataset_similarity/`: notebooks for reproduce the results of dataset similarity estimation.
- `case_study/`: notebooks and scripts for results of the tiny case study on facial attribute classification.
- `model_similarity`: notebooks for reproduce the results of model similarity estimation.
- `train_cv.py`: script to train models on CIFAR-10/100 and Tiny-ImageNet.
- `train_facial.py`: script to train models on facial attribute datasets (UTKFace and FairFace). The models trained are used in the case study.
- `train_nlp.py`: script to finetune BERT-based language models.
- `utils.py`: script containing auxillary functions like data loading, network initialization and training, etc.

## Preparation



First, please modify variables in format of `{}_PATH` in `conf/global_settings.py` to set storing location of your models and data.


### Dataset preparation

For CIFAR-10/100 and Tiny-ImageNet, check `DatasetSimilarity.ipynb` to obtain intersected datasets. The files should be in format of `{dataset}_intersect_{similarity}.pkl` (e.g. CIFAR10_intersect_0.7, CIFAR10_intersect_0.8).

For facial data, you can download [FairFace](https://github.com/dchen236/FairFace) and [UTKFace](https://susanqq.github.io/UTKFace/) or directly use our pre-processed dataset in format of .pkl.

Then, run `ProcessFacialAttribute.ipynb` to obtain .pkl files.
*Note:* The step is to accelerate data loading during training. It is ok to integrate this part into training scripts.


### Model preparation
Run `train_cv.py`, `train_facial.py` or `'train_nlp.py` to obtain tested and surrogate models.
The parameter description are provided in scripts. 

Here are some examples:

```
python train_cv.py -net resnet18 -dataset cifar10 -subset 1 -inter_propor 0.2  -copy_id 0 -gpu_id 0

python train_facial.py -net regnet_y_8gf -inter_propor 0.1 -copy_id 1

python train_nlp.py -inter_propor 0.3 -save_path int0.3 -archi mini
```

*Note*: Before running the notebooks, it is necessary to prepare the both victim's, surrogate and adversary models with the scripts and well organize the models into correct folders (and with correct file name).

## Similarity Estimation

### Dataset
The notebooks are under folder `dataset_similarity/`.
The scripts `similarity_cv_predict.py` generates model outputs as intermediate results for faster result analysis in notebooks.
The analysis notebook is in `InferSimilarity.ipynb`.
`ablation_similarity_model_predict.py` generates intermediate results for adversary's different learning rate and training epochs.
The analysis notebook is `AblationStudy.ipynb`.

For text classification, the notebook `NLPDatasimilarity.ipynb` takes care of intermediate results and final analysis.

Similarly, under `case_study/`, the script `similarity_facial_predict.py` outputs intermediate results and `SimilarityEstimation.ipynb` outputs final results.

### Model

Under `model_similarity/`, the folder `Quantization/` contains notebooks to quantize the models (ResNet-18, VGG-16 and MobileNet).
The notebooks starting with `ModelHash_` contains code for detection of different model modifications.

