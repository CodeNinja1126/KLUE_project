from transformers import AutoTokenizer,  Trainer, TrainingArguments, AutoModelForSequenceClassification

# Bert 관련 함수
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
# electra 관련
from transformers import ElectraForSequenceClassification, ElectraConfig
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from os import makedirs

model_name_dict = {'bert' : "bert-base-multilingual-cased", 
                  'electra1' : 'monologg/koelectra-base-v3-generator',
                  'electra2' : 'monologg/koelectra-base-v3-discriminator',
                  'roberta' : 'xlm-roberta-large'}


def inference(model, tokenized_sent, device, args):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      if args.model_type == 'roberta':
        outputs = model(
        input_ids=data['input_ids'].to(device),
        attention_mask=data['attention_mask'].to(device),
        )
      else:
        outputs = model(**data)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
  
  return np.array(output_pred).flatten()


def load_test_dataset(dataset_dir, tokenizer):
  # test_dataset = load_data(dataset_dir) # 일반 테스트 데이터셋
  test_dataset = pd.read_csv(dataset_dir, sep='\t') # ner 테스트 데이터셋
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label


def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = model_name_dict[args.model_type]  
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

  # load my model
  MODEL_NAME = args.model_dir # model dir.
  model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

  # load test datset
  # test_dataset_dir = '/opt/ml/input/data/test/test.tsv' # 일반 테스트 데이터셋
  test_dataset_dir = '/opt/ml/input/data/test/ner_test_ver2.tsv' # ner 테스트 데이터셋
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  pred_answer = inference(model, test_dataset, device, args)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  
  output = pd.DataFrame(pred_answer, columns=['pred'])
  makedirs('prediction', exist_ok=True)
  output.to_csv('./prediction/submission.csv', index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, required=True)
  parser.add_argument('--model_type', type=str, required=True)
  args = parser.parse_args()
  print(args)
  main(args)
  
