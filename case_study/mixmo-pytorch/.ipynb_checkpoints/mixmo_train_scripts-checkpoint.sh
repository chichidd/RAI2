# 51602 gpu0
python train.py -g 0 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.2 --saveplace int_0.2 --copy_id 0
python train.py -g 0 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.2 --saveplace int_0.2 --copy_id 1
# 51602 gpu1
python train.py -g 1 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.2 --saveplace int_0.2 --copy_id 2
python train.py -g 1 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.2 --saveplace int_0.2 --copy_id 3

# 51800 gpu0
python train.py -g 0 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.2 --saveplace int_0.2 --copy_id 4
# 51800 gpu1
python train.py -g 1 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.4 --saveplace int_0.4 --copy_id 0
# 51800 gpu2
python train.py -g 2 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.4 --saveplace int_0.4 --copy_id 1
# 51800 gpu3
python train.py -g 3 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.4 --saveplace int_0.4 --copy_id 2

# 51601 gpu0
python train.py -g 0 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.4 --saveplace int_0.4 --copy_id 3
python train.py -g 0 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.4 --saveplace int_0.4 --copy_id 4
# 51601 gpu1
python train.py -g 1 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.6 --saveplace int_0.6 --copy_id 0
python train.py -g 1 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.6 --saveplace int_0.6 --copy_id 1

#nsec gpu0
python train.py -g 0 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.6 --saveplace int_0.6 --copy_id 2
python train.py -g 0 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.6 --saveplace int_0.6 --copy_id 3
#nsec gpu1
python train.py -g 1 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.6 --saveplace int_0.6 --copy_id 4
python train.py -g 1 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.8 --saveplace int_0.8 --copy_id 0
#nsec gpu2
python train.py -g 2 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.8 --saveplace int_0.8 --copy_id 1
python train.py -g 2 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.8 --saveplace int_0.8 --copy_id 2
#nsec gpu3
python train.py -g 3 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.8 --saveplace int_0.8 --copy_id 3
python train.py -g 3 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 0.8 --saveplace int_0.8 --copy_id 4



# nsec gpu1
python train.py -g 1 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 1.0 --saveplace int_1.0 --copy_id 2
# nsec gpu2
python train.py -g 2 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 1.0 --saveplace int_1.0 --copy_id 3
# nsec gpu3
python train.py -g 3 --config_path config/tiny/resnet182-2_cutmixmo-p5.yaml --dataplace dataplace --inter_propor 1.0 --saveplace int_1.0 --copy_id 4