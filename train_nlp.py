import warnings
warnings.filterwarnings('ignore')
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import argparse


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-inter_propor', type=float, default=None, help='intersection')
    parser.add_argument('-save_path', type=str, default=None, help='checkpoint file')
    parser.add_argument('-archi', type=str, default='tiny', help='architecture')
    parser.add_argument('-gpuid', type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid  

    start = 60000 - int(60000 * args.inter_propor)
    trainset1 = load_dataset("ag_news", split="train[{}:{}]".format(start, start + 60000))
    testset = load_dataset("ag_news", split="test")

    train_dataset1 = trainset1.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = testset.map(lambda examples: {'labels': examples['label']}, batched=True)


    model_id = 'prajjwal1/bert-{}'.format(args.archi)
    if 'base' in args.archi:
        model_id = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_id)


    MAX_LENGTH = 256
    train_dataset1 = train_dataset1.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    # train_dataset2 = train_dataset2.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)



    train_dataset1.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    # train_dataset2.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])


    for i in range(10):
        # note that we need to specify the number of classes for this task
        # we can directly use the metadata (num_classes) stored in the dataset
        model = AutoModelForSequenceClassification.from_pretrained(model_id, 
                    num_labels=train_dataset1.features["label"].num_classes)
        

        output_dir = os.path.join('/data1/checkpoint/nlp/bert-{}/'.format(args.archi), args.save_path, 'model_{}'.format(i))
        training_args = TrainingArguments(
            output_dir=output_dir,          # output directory
            learning_rate=5e-4,
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=64,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            # logging_dir='../results/nlp/logs',            # directory for storing logs
            # logging_steps=10000,
            logging_strategy="no",
            do_train=True,
            do_eval=True,
            no_cuda=False,
            load_best_model_at_end=False,
            # eval_steps=100,
            save_strategy="epoch",
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=model,                         # the instantiated ? Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset1,         # training dataset
            eval_dataset=test_dataset,            # evaluation dataset
            compute_metrics=compute_metrics
        )

        train_out = trainer.train()
        model.cpu()
        trainer.save_model(os.path.join(output_dir, "final"))


