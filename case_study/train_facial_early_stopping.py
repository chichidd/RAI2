import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchmetrics import Accuracy
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append("../")
import utils 
from conf import settings

class FaceDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage):
        # called on every GPU
        # X_tensor_tr, y_tensor_tr = pickle.load(open(f'data/facial_attribute/fairface_utk_mix/intersect_{self.args.inter_propor}.pkl', 'rb'))
        fair_train_img_set1_tensor, fair_train_label_set1 = pickle.load(open(os.path.join(settings.DATA_PATH, "facial_attribute", "fairface_set1_tensor.pkl"), "rb"))
        fair_train_img_set2_tensor, fair_train_label_set2 = pickle.load(open(os.path.join(settings.DATA_PATH, "facial_attribute", "utk_tensor.pkl"), "rb"))
        set1_num = len(fair_train_img_set1_tensor)

        shift = int(self.args.inter_propor * set1_num)
        print(f"Sample {set1_num - shift} to {set1_num} from set1, 0 to {set1_num - shift} from set2.")
        X_tensor_tr = torch.cat([fair_train_img_set1_tensor[set1_num - shift:], fair_train_img_set2_tensor[:set1_num - shift]])
        y_tensor_tr = torch.cat([fair_train_label_set1[set1_num - shift:], fair_train_label_set2[:set1_num - shift]])
        del fair_train_img_set1_tensor
        del fair_train_label_set1
        del fair_train_img_set2_tensor
        del fair_train_label_set2
            
        mean = X_tensor_tr.mean(dim=[0, 2, 3])
        std = X_tensor_tr.std(dim=[0, 2, 3])
        self.num_classes = len(torch.unique(y_tensor_tr))
        transform_train = transforms.Compose([
            transforms.RandomCrop(128, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Normalize(mean, std)
        ])
        self.train_dst = utils.SubTrainDataset(X_tensor_tr, y_tensor_tr, transform=transform_train)

        transform_val = transforms.Compose([
            transforms.Normalize(mean, std)
            ])
        X_tensor_val, y_tensor_val = pickle.load(open(os.path.join(settings.DATA_PATH, 'facial_attribute', 'fairface_val_tensor.pkl'), 'rb'))
        self.val_dst = utils.SubTrainDataset(X_tensor_val, y_tensor_val, transform=transform_val)

    def train_dataloader(self):
        return DataLoader(self.train_dst, batch_size=self.args.b, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dst, batch_size=self.args.b, shuffle=False)


class LitModel(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()


    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.args.lr,  momentum=0.9, weight_decay=self.args.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=settings.CASE_STUDY_MILESTONES, gamma=0.2)
        return [opt], [scheduler]



if __name__ == '__main__':
    bs_times = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-inter_propor', type=float, default=0.6)
    parser.add_argument('-copy_id', type=int, default=0, help='different copy id')
    parser.add_argument('-b', type=int, default=128 * bs_times, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1 * bs_times, help='initial learning rate')
    parser.add_argument('-wd', type=float, default=5e-5, help='weight_decay')
    parser.add_argument('-early_stop', type=bool, default=False, help='whether apply early stopping')
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.copy_id)
    
    checkpoint_path = os.path.join(settings.CASE_STUDY_CHECKPOINT_PATH, 
    'early_stopping', 'early' if args.early_stop else 'nonearly', 
    f"{args.net}_wd{args.wd}_{args.inter_propor}{f'_{args.copy_id}' if args.copy_id is not None else ''}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)


    dm = FaceDataModule(args)
    dm.setup(stage='train')
    net = getattr(models, args.net)(num_classes=dm.num_classes)
    model = LitModel(args, net)
    model.datamodule = dm

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc", 
        dirpath=checkpoint_path, 
        filename="model_best_{val_acc:.2f}", save_top_k=1, mode="max")

    if args.early_stop:
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max")
        
        trainer = pl.Trainer(
        # progress_bar_refresh_rate=10, 
        default_root_dir=checkpoint_path,
        max_epochs=settings.CASE_STUDY_EPOCHS, 
        gpus=torch.cuda.device_count(), 
        callbacks=[early_stop_callback, checkpoint_callback]
        )
    else:
        regular_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=None, 
            dirpath=checkpoint_path, 
            filename="model_{epoch}", every_n_epochs=10)
        
        trainer = pl.Trainer(
            # progress_bar_refresh_rate=10, 
            default_root_dir=checkpoint_path,
            max_epochs=settings.CASE_STUDY_EPOCH, 
            gpus=torch.cuda.device_count(), 
            callbacks=[checkpoint_callback, regular_checkpoint_callback]
        )
    
    trainer.fit(model, dm)
    
    
