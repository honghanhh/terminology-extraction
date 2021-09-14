import os
import glob 
import spacy
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
torch.cuda.empty_cache()
# !pip install stanza
import stanza
stanza.download('fr')
nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma')
# !python -m spacy download fr_core_news_sm
import fr_core_news_sm
nlp1 = fr_core_news_sm.load()
# from tqdm.notebook import tqdm
from tqdm import tqdm_notebook as tqdm
from simpletransformers.ner import NERModel, NERArgs

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Training language models.')
  parser.add_argument('-in_train_path', type=str, dest='train_input', default="../terminology-extraction/processed_data/fr/ann_train_lem_1c.csv")
  parser.add_argument('-in_groundtruth_path', type=str, dest='train_data', default="../terminology-extraction/ACTER/fr/htfl/texts/annotated/")
  parser.add_argument('-out_path', type=str, dest='output', default="../terminology-extraction/results/weighted_results/fr/fr_model.pkl")
  args = parser.parse_args()

  
  train = pd.read_csv(args.train_input)
  train['words'] = [str(x) for x in train['words']]
  train_df, eval_df = train_test_split(train, test_size=0.15, random_state=2021)
  class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_df.labels),
                                                  train_df.labels)
  model_args = NERArgs(
                      labels_list = ['B','I', 'O'],
                      manual_seed = 2021,
                      num_train_epochs = 4,
                      max_seq_length = 512,
                      use_early_stopping = True,
                      overwrite_output_dir = True,
                      train_batch_size = 8#,
                      # do_lower_case = True
                      )

  model  = NERModel(
      "camembert", "camembert-base", args=model_args, weight = list(class_weights), use_cuda=True, cuda_device=-1
  )

  model.train_model(train_df)

  result, model_outputs, wrong_predictions = model.eval_model(
      eval_df
  )

  list_of_files = os.listdir(args.in_groundtruth_path)
  lines=[]
  for file in list_of_files:
      f = open(path+file, "r")
      lines.append(f.readlines())
      f.close()

  terms = []
  preds = []
  for lines_ in lines:
    sentences = [[token.text for token in nlp1(line.strip())] for line in lines_]
    predictions, raw_outputs = model.predict(sentences, split_on_space=False)
    preds.extend(predictions)

  with open(args.output, 'wb') as f: 
      pkl.dump(preds, f)