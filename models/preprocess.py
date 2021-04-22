import os
import re
import pandas as pd
import spacy
import pickle
import stanza
# stanza.download('en')
class KeyTerm():
    def __init__(self, data_dir = "../ACTER", language = 'en', term = "equi", nes=False):
        data_file = os.path.join(data_dir, language, term, 'annotations')
        if nes:
            data_file = os.path.join(data_file, '{0}_{1}_terms_nes.ann'.format(term, language))
        else:
            data_file = os.path.join(data_file, '{0}_{1}_terms.ann'.format(term, language))
        self.df = pd.read_csv(data_file, sep='\t', names=['word', 'class'], header=None)
        self.df['len'] = [len(x) for x in self.df['word']]
        self.df = self.df[self.df['len'] > 1][['word', 'class']]
        self.nlp = stanza.Pipeline(lang='en')
        self.keys = self.df['word'].to_list()
        self.keys = [str(x) for x in self.keys]
        self.keys_lemma = list(set([self.lemma(x) for x in self.keys]))
    
    def lemma(self, doc):
        lemma_word = self.nlp(str(doc))
        lemma_word = ' '.join([w.lemma for sent in lemma_word.sentences for w in sent.words])
        lemma_word = re.sub(' -','-',lemma_word)
        lemma_word = re.sub('- ','-',lemma_word)
        # lemma_word = re.sub(' -','-',lemma_word)
        # lemma_word = re.sub('- ','-',lemma_word)
        # lemma_word = re.sub('/ ','/',lemma_word)
        # lemma_word = re.sub(' /','/',lemma_word)
        lemma_word = re.sub(' \)', ' ',lemma_word)
        lemma_word = re.sub('\( ', ' ',lemma_word)
        lemma_word = re.sub(' +', ' ',lemma_word)
        return lemma_word

    def extract_doc(self, doc, use_lemma=True):
        doc = self.nlp(doc)
        results = []
        for sent in doc.sentences:
            if use_lemma:
                lemma_word = ' '.join([w.lemma for w in sent.words])
                lemma_word = re.sub(' -','-',lemma_word)
                lemma_word = re.sub('- ','-',lemma_word)
                # lemma_word = re.sub(' -','-',lemma_word)
                # lemma_word = re.sub('- ','-',lemma_word)
                # lemma_word = re.sub('/ ','/',lemma_word)
                # lemma_word = re.sub(' /','/',lemma_word)
                lemma_word = re.sub(' \)', ' ',lemma_word)
                lemma_word = re.sub('\( ', ' ',lemma_word)
                lemma_word = re.sub(' +', ' ',lemma_word)

                tokens = lemma_word.split()
                text = lemma_word
                keys = self.keys_lemma
            else:
                tokens = [token.text for token in sent.tokens]
                text = sent.text
                keys = self.keys

            label, term = self.extract(tokens, text=text, keys=keys)

            if set(label) != {'O'}:
                results.append({
                    "tokens": tokens,
                    "sent": sent.text,
                    "labels": label,
                    "terms": term
                })
        return results

    def extract(self, tokens, text = None, keys = None):
        # tokens = [self.lemma(x) for x in tokens]
        if text == None:
            text = ' '.join(tokens)

        if keys == None:
            keys = self.keys

        z = ['O'] * len(tokens)
        # text = self.lemma(text)
        for k in keys:
            # k = self.lemma(k)
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
    def __init__(self, data_dir = "../ACTER", language = 'en', nes=False):
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
                        '''
                        doc = nlp(line.strip().lower())
                        for sent in doc.sents:
                            tokens = [token.text for token in sent]
                            label, t = keyterm.extract(tokens)
                            if set(label) != {'O'}:
                                sentences.append(sent.text)
                                labels.append(label)
                                all_token.append(tokens)
                                terms.append(t)
                        '''
                        results = keyterm.extract_doc(line.strip().lower(), use_lemma=True)
                        for result in results:
                            if set(result['labels']) != {'O'}:
                                sentences.append(result['sent'])
                                labels.append(result['labels'])
                                all_token.append(result['tokens'])
                                terms.append(result['terms'])

        return sentences, labels, all_token, terms

if __name__ == '__main__':
    dataset = ActerDataset()
    path = "../processed_data/en/"
    if not os.path.exists(path):
            os.mkdir(path) 
    with open(path + "ann_train_lem1.pkl", "wb") as output_file:
        pickle.dump((dataset.sentences, dataset.labels, dataset.tokens, dataset.terms), output_file)