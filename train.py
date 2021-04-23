import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
# Bert 관련
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
# electra 관련
from transformers import ElectraForSequenceClassification, ElectraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from load_data import *
import argparse

# 주요 huggingface model 정의 
model_name_dict = {'bert' : "bert-base-multilingual-cased",
                  'electra' : 'monologg/koelectra-base-v3-discriminator',
                  'roberta' : 'xlm-roberta-large',
                  'kobert' : 'monologg/kobert'}

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }


def train(arg):
  # load model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name_dict[arg.m])
  
  # load dataset
  if arg.train_data == 'val':
    train_dataset = load_data("/opt/ml/input/data/train/new_train.tsv")
  elif arg.train_data == 'ner':
    train_dataset = pd.read_csv("/opt/ml/input/data/train/new_train_ner.tsv", sep='\t')
  elif arg.train_data == 'train':
    train_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  
  # load validation set
  if arg.train_data == 'ner':
    dev_dataset = pd.read_csv("/opt/ml/input/data/train/val_train_ner.tsv", sep='\t')
  else:
    dev_dataset = load_data("/opt/ml/input/data/train/val_train.tsv")
  
  train_label = train_dataset['label'].values
  dev_label = dev_dataset['label'].values
  
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter

  config = AutoConfig.from_pretrained(model_name_dict[arg.m])
  config.num_labels = 42
  model = AutoModelForSequenceClassification.from_pretrained(model_name_dict[arg.m], config=config)  

  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=arg.a,          # output directory
    save_total_limit=3,              # number of total save model.
    # save_steps=500,                 # model saving step.
    save_strategy='epoch',
    num_train_epochs=arg.e,              # total number of training epochs
    learning_rate=arg.lr,               # learning_rate
    per_device_train_batch_size=arg.b,  # batch size per device during training
    per_device_eval_batch_size=40,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='epoch' , # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    # load_best_model_at_end=True,
    # metric_for_best_model=compute_metrics,
    # greater_is_better=True,
    eval_steps = 500,            # evaluation step.
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()
  trainer.save_model(arg.o)
  trainer.save_state()


def main(arg):
  train(arg)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Set some train option')
  parser.add_argument('-a', default='./results', type=str, help='save ckpt address (default : ./results)')
  parser.add_argument('-lr', default=5e-5, type=float, help='learning rate (default : 5e-5)')
  parser.add_argument('-o', default='./results/output', type=str, help='save ckpt address (default : ./results/output)')
  parser.add_argument('-b', default=16, type=int, help='batch size (default : 16)')
  parser.add_argument('-e', default=4, type=int, help='epoch_num (default : 4)')
  parser.add_argument('--train_data', default='val', help='training data (default : val)')
  parser.add_argument('-m', default="bert", type=str, help='model name(default : bert)')
  arg = parser.parse_args()
  main(arg)
