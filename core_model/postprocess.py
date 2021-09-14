import re
import pandas as pd
import pickle as pkl
from collections import Counter
import nltk
from nltk.corpus import stopwords 
import stanza
nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma')

def evaluation_metrics(pred, gt):
    TP = len(set(pred) & set(gt)) 
    FP = len(set(pred)-set(gt))
    FN = len(set(gt)-set(pred))
    precision = round((TP/(TP+FP))*100, 2)
    recall = round((TP/(TP+FN))*100,2)
    f1_score = round((2 * precision * recall) / (precision + recall),2)
    return precision, recall, f1_score 

def get_term_(predictions):
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
        for token, label in zip(tokens, labels):
            if label == 'B':
                #Lưu vị trí B
                b_pos = i
                term = [token]
            elif label == 'I':
                term.append(token)
            elif len(term) > 0:
                terms.append(' '.join(term))
                term = []
        if len(term) > 0:
            terms.append(' '.join(term))
            # Check b_pos = 0 không
        all_term.append(terms)
    
    final_terms = []
    for i in all_term:
        final_terms.extend(i)

    final_terms = [x.lower().strip() for x in final_terms]
    return final_terms  


domain_path ='/Users/hanh/Documents/Github/terminology-extraction/ACTER/fr/'
preds_path = '/Users/hanh/Documents/Github/terminology-extraction/results/weighted_results/fr/'

def term_evaluation(domain_path, preds_path, rule=None):
    groundtruth = pd.read_csv(domain_path, sep='	', engine='python',header=None)
    gt = list(groundtruth[0])
    predictions = pkl.load(open(preds_path, 'rb'))
    preds =  get_term_(predictions)
    stop_words = set(stopwords.words('french'))
    pred_terms =  set(preds) - set(stop_words)
    pred_terms = [x for x in pred_terms if len(x)>1]
    pred_terms = [x.lower().strip() for x in pred_terms]
    pred_terms = [re.sub(' -','-', x) for x in pred_terms]
    pred_terms = [re.sub('- ','-', x) for x in pred_terms]
    pred_terms = [re.sub('\(','', x) for x in pred_terms]
    pred_terms = [re.sub('\/','', x) for x in pred_terms]
    print(evaluation_metrics(pred_terms, gt))
    return set(pred_terms), set(gt)

stop_words = set(stopwords.words('french'))
predictions, groundtruth =  term_evaluation(domain_path+'htfl/annotations/htfl_fr_terms.ann', preds_path+'ann_weighted_camembert.pkl')
predictions2 = [' '.join(x.split()[:-1]) if x.split()[-1] in stop_words else  x for x in predictions]
print(evaluation_metrics(set(predictions2), groundtruth),evaluation_metrics(set(predictions2), groundtruth))
