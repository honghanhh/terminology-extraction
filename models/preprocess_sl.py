import os
import pandas as pd
import spacy
import pickle

class KeyTerm():
    def __init__(self, data_dir = "../ACTER", language = 'fr', term = "equi", nes=False):
        data_file = os.path.join(data_dir, language, term, 'annotations')
        if nes:
            data_file = os.path.join(data_file, '{0}_{1}_terms_nes.ann'.format(term, language))
        else:
            data_file = os.path.join(data_file, '{0}_{1}_terms.ann'.format(term, language))
        self.df = pd.read_csv(data_file, sep='\t', names=['word', 'class'], header=None)
        self.keys = self.df['word'].to_list()
        self.keys = [str(x) for x in self.keys]
    
    def extract(self, tokens, text = None):
        if text == None:
            text = ' '.join(tokens)
        z = ['O'] * len(tokens)
        for k in self.keys:
            if k in text:
                if len(k.split())==1:
                    try:
                        z[tokens.index(k.lower().split()[0])] = 'B'
                    except ValueError:
                        continue
                elif len(k.split())>1:
                    try:
                        if tokens.index(k.lower().split()[0]) and tokens.index(k.lower().split()[-1]):
                            z[tokens.index(k.lower().split()[0])] = 'B'
                            for j in range(1, len(k.split())):
                                z[tokens.index(k.lower().split()[j])] = 'I'
                    except ValueError:
                        continue
        for m, n in enumerate(z):
            if z[m] == 'I' and z[m-1] == 'O':
                z[m] = 'O'
        
        terms = []
        term = []
        for token, label in zip(tokens, z):
            if label == 'B':
                term = [token]
            elif label == 'I':
                term.append(token)
            elif len(term) > 0:
                terms.append(' '.join(term))
                term = []
        if len(term) > 0:
            terms.append(' '.join(term))
        return z, terms

class ActerDataset():
    def __init__(self, data_dir = "../ACTER", language = 'fr', nes=False):
        if language == 'en':
            nlp = spacy.load("en_core_web_sm")
        elif language == 'fr':
            nlp = spacy.load("fr_core_news_sm")
        elif language == 'nl':
            nlp = spacy.load("nl_core_news_sm")
        self.sentences = []
        self.labels = []
        self.tokens = []
        self.terms = []

        language_dir = os.path.join(data_dir, language)
        for term in ['corp','equi','wind']:
            keyterm = KeyTerm(data_dir = data_dir, language=language, term = term, nes=nes)

            sentences, labels, tokens, terms = self.extract_term(language_dir, term, keyterm, nlp)

            self.sentences.extend(sentences)
            self.labels.extend(labels)
            self.tokens.extend(tokens)
            self.terms.extend(terms)

    def extract_term(self, data_dir, term, keyterm, nlp):
        data_dir = os.path.join(data_dir, term, 'texts', "annotated")
        # print(term)
        # print(data_dir)
        sentences = []
        labels = []
        all_token = []
        terms = []
        for file in os.listdir(data_dir):
            if file.endswith('.txt') and file.startswith(term):
                data_file = os.path.join(data_dir, file)
                print(data_file)
                with open(data_file) as f:
                    for line in f:
                        doc = nlp(line.strip().lower())
                        for sent in doc.sents:
                            tokens = [token.text for token in sent]
                            label, t = keyterm.extract(tokens)
                            if set(label) != {'O'}:
                                sentences.append(sent.text)
                                labels.append(label)
                                all_token.append(tokens)
                                terms.append(t)

        return sentences, labels, all_token, terms

if __name__ == '__main__':
    dataset = ActerDataset()
    path = "../processed_data/fr/"
    if not os.path.exists(path):
            os.mkdir(path) 
    with open(path + "ann_train.pkl", "wb") as output_file:
        pickle.dump((dataset.sentences, dataset.labels, dataset.tokens, dataset.terms), output_file)

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