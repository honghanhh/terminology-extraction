# RUN on Collab
import os
import pandas as pd
import pickle
import torch
from simpletransformers.ner import NERModel, NERArgs
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import glob
import spacy
import stanza
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
import en_core_web_sm
nlp1 = en_core_web_sm.load()
torch.cuda.empty_cache()

groundtruth = pd.read_csv('/home/tranthh/terminology-extraction/ACTER/en/htfl/annotations/htfl_en_terms.ann', sep='	', engine='python',header=None)
gt = list(groundtruth[0]) 

# P = TP/(TP+FP)
# R =  TP/(TP+FN)
def evaluation_metrics(pred, gt):
  TP = len(set(pred) & set(gt)) 
  FP = len(set(pred)-set(gt))
  FN = len(set(gt)-set(pred))
  print(TP,FP, FN)
  precision = round((TP/(TP+FP))*100, 2)
  recall = round((TP/(TP+FN))*100,2)
  f1_score = round((2 * precision * recall) / (precision + recall),2)
  return precision, recall, f1_score



model_args = NERArgs(
                    labels_list = ['B', 'O', 'I'],
                    manual_seed = 2021,
                    num_train_epochs = 4,
                    max_seq_length = 512,
                    use_early_stopping = True,
                    overwrite_output_dir = True,
                    train_batch_size = 16
                    )
train = pd.read_csv('/home/tranthh/terminology-extraction/processed_data/en/ann_train_lem.csv')
train['words'] = [str(x) for x in train['words']]
train_df, eval_df = train_test_split(train, test_size=0.2, random_state=2021)
model  = NERModel(
    "roberta", "roberta-base", args=model_args
)

model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(
    eval_df
)

def get_term(predictions):
    all_term = []
    for sentence in predictions:
        tokens = []
        labels = []
        for d in sentence:
            tokens.extend(d.keys())
            labels.extend(d.values())

        for i, label in enumerate(labels):
            if labels[i] == 'I' and (i == 0 or labels[i - 1] == 'O'):
                labels[i] = 'O'

        terms = []
        term = []
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label == 'B': 
                #Lưu vị trí B
                b_pos = i
                term = [token]
            elif label == 'I':
                term.append(token)
            elif len(term) > 0:
                terms.append(' '.join(term))
                # if nlp(str(tokens[b_pos])).sentences[0].word[0].upos == 'NOUN':
                # Check b_pos = 0 không
                if b_pos != 0:
                    # print(tokens[b_pos - 1])
                    if (tokens[b_pos - 1] != '') and (tokens[b_pos - 1] != ' '):
                        if len(nlp(str(tokens[b_pos - 1])).sentences) > 0:
                            b_word = nlp(str(tokens[b_pos - 1])).sentences[0].words[0]
                            # Check vị trí b_pos - 1: terms.append()
                            if  (b_word.text != 'None') and ((b_word.upos == 'NOUN') or (b_word.upos == 'ADJ')):
                                terms.append(' '.join([b_word.text] + term))
                    if (tokens[i] != '') and (tokens[i] != ' '):
                        # Check vị trí i: terms.append()
                        if len(nlp(str(tokens[i])).sentences) > 0:
                            a_word = nlp(str(tokens[i])).sentences[0].words[0]
                            if (a_word.text != 'None') and (a_word.upos == 'NOUN'):
                                terms.append(' '.join(term + [a_word.text]))
                term = []
        if len(term) > 0:
            terms.append(' '.join(term))
            # check b_pos - 1
        all_term.append(terms)

    return all_term                


path = "/home/tranthh/terminology-extraction/ACTER/en/htfl/texts/annotated/"
list_of_files = os.listdir("/home/tranthh/terminology-extraction/ACTER/en/htfl/texts/annotated/")
lines=[]
for file in list_of_files:
    f = open(path+file, "r")
    #append each line in the file to a list
    lines.append(f.readlines())
    f.close()

terms = []
for lines_ in lines:
  sentences = [[token.text for token in nlp1(line.strip())] for line in lines_]
  predictions, raw_outputs = model.predict(sentences, split_on_space=False)
  terms.extend(get_term(predictions))

final_terms = []
for i in terms:
  final_terms.extend(i)

final_terms = [x.lower() for x in final_terms]
print(len(final_terms))
final_terms = set(final_terms)
print(len(final_terms))

precision, recall, f1_score = evaluation_metrics(final_terms, gt)
print(precision, recall, f1_score)

preds = pd.DataFrame (final_terms,columns=['predictions'])
preds.to_csv('/home/tranthh/roberta_ann_lem_predictions_final.csv', index=False)