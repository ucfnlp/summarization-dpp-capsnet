from __future__ import print_function

import argparse
import io
import json
import os
import pickle
import re
#from collections import defaultdict as defdict
import operator
import string

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from util import get_snli_file_path, get_multinli_file_path, get_word2vec_file_path, get_word2vec_100d_file_path, ChunkDataManager
from cnn_dm import cnn_dm_data


def pad(x, maxlen):
    if len(x) <= maxlen:
        pad_width = ((0, maxlen - len(x)), (0, 0))
        return np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    res = x[:maxlen]
    return np.array(res, copy=False)

def preprocess_word(word):
    if 0:
        # lower case, punctuation
        punc = set(string.punctuation)
        return filter(lambda x: ''.join(ch for ch in x if ch not in punc), word.lower())
    else:
        # lower case
        return word.lower().decode('utf-8')

def add_START_END_token(data, id_start_num, id_end_num):
    # data: list of numpy array [2, sentences, words] zero padded at the end of sentence sequence
    # add <START>, <END> to the sentence - input 42 in length    
    data_shape = data[0].shape
    num_sents = data_shape[0]
    num_words = data_shape[1] + 2   # 44 in length    
    data_added = [np.zeros((num_sents, num_words)), np.zeros((num_sents, num_words))]
    for r in range(2):
        for i, sent in enumerate(data[r]):
            inserted = False
            data_added[r][i][0] = id_start_num
            #np.insert(data[r][i], 0, id_start_num)        # now 43 in length
            for j, word in enumerate(sent):
                if word == 0.:
                    data_added[r][i][j+1]= id_end_num
                    #data[r][i][j] = id_end_num  # change last 0 token with <END>                    
                    inserted = True
                    break
                else:
                    data_added[r][i][j+1]= word         # copy data
                    
            if not inserted:
                data_added[r][i][-1] = id_end_num
                #data[r][i].append(id_end_num)

    return data_added[0], data_added[1]

def add_START_token(data, id_num):
    # not working -> data is np.array
    for st in range(2):
        for i, sents in enumerate(data[st]):
            data[st][i].insert(0, id_num)
    

def add_END_token(data, id_num):
    # add <END> at the end of sentence
    for r in range(2):
        for i, sents in enumerate(data[r]):
            inserted = False            
            for j, word in enumerate(sents):
                if data[r][i][j] == 0.:
                    data[r][i][j] = id_num
                    inserted = True
                    break
            if not inserted:
                data[r][i][-1] = id_num
    return data[0], data[1]


class BasePreprocessor(object):

    def __init__(self):
        self.word_to_id = {}
        self.char_to_id = {}
        self.words = []         # from word vector
        self.vectors = []       # from word vector
        self.part_of_speech_to_id = {}
        self.unique_words = set()
        self.unique_words_freq = dict()     # [word, freq]
        self.unique_words_voca = []        
        self.unique_parts_of_speech = set()
        
        self.stop_words = set(stopwords.words('english'))

    @staticmethod
    def load_data(file_path):
        """
        Load jsonl file by default
        """
        with open(file_path) as f:
            lines = f.readlines()
            text = '[' + ','.join(lines) + ']'
            return json.loads(text)

    @staticmethod
    def load_word_vectors(file_path, separator=' ', normalize=True, max_words=None):
        """
        :return: words[], np.array(vectors)
        """
        seen_words = set()
        words = []
        vectors = []
        vector_size = None
        print('Loading', file_path)
        with io.open(file_path, mode='r', encoding='utf-8') as f:
            for line in f: #tqdm(f):
                values = line.replace(' \n', '').split(separator)
                word = values[0]
                if len(values) < 10 or word in seen_words:
                    print('Invalid word:', word)
                    continue

                seen_words.add(word)
                vec = np.asarray(values[1:], dtype='float32')
                if normalize:
                    vec /= np.linalg.norm(vec, ord=2)

                if vector_size is None:
                    vector_size = len(vec)
                elif len(vec) != vector_size:
                    print('Skipping', word)
                    continue

                words.append(word)
                vectors.append(vec)
                if max_words and len(words) >= max_words:
                    break

        vectors = np.array(vectors, dtype='float32', copy=False)
        return words, vectors

    def get_words_with_part_of_speech(self, sentence):
        """
        :return: words, parts_of_speech
        """
        raise NotImplementedError

    def get_sentences(self, sample):
        """
        :param sample: sample from data
        :return: premise, hypothesis
        """
        raise NotImplementedError

    def get_all_words_with_parts_of_speech(self, file_paths):
        """
        :param file_paths: paths to files where the data is stored
        :return: words, parts_of_speech
        """
        all_words = []
        all_parts_of_speech = []
        for file_path in file_paths:
            data = self.load_data(file_path=file_path)

            for sample in tqdm(data):
                premise, hypothesis = self.get_sentences(sample)
                premise_words,    premise_speech    = self.get_words_with_part_of_speech(premise)
                hypothesis_words, hypothesis_speech = self.get_words_with_part_of_speech(hypothesis)
                all_words           += premise_words  + hypothesis_words
                all_parts_of_speech += premise_speech + hypothesis_speech
        
        self.unique_words           = set(all_words)
        
        for w in all_words:
            if w in self.unique_words_freq:
                self.unique_words_freq[w] += 1
            else:
                self.unique_words_freq[w] = 1
            
        self.unique_parts_of_speech = set(all_parts_of_speech)
    
    def load_txt_data(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()
        return lines
    
    def get_all_words_DUC(self, dir_paths):
        all_words = []        
        for dir in dir_paths:
            file_name = os.path.basename(dir)
            file_path = os.path.join(dir, file_name+'.txt')
            sents = self.load_txt_data(file_path = file_path)
            
            words = []
            for sent in sents:
                word_tokens = word_tokenize(sent)                
                for word in word_tokens: #sent.split():                    
                    words.append(preprocess_word(word))               
            all_words += words        
        
        self.unique_words = set(all_words)        
        
        for w in all_words:
            if w in self.unique_words_freq:
                self.unique_words_freq[w] += 1
            else:
                self.unique_words_freq[w] = 1

    @staticmethod
    def get_not_present_word_vectors(not_present_words, word_vector_size, normalize):
        res_words = []
        res_vectors = []
        for word in not_present_words:
            vec = np.random.uniform(size=word_vector_size)
            if normalize:
                vec /= np.linalg.norm(vec, ord=2)
            res_words.append(word)
            res_vectors.append(vec)
        return res_words, res_vectors
    
    def call_load_word_vector(self, file_path, normalize=False, max_words=None):
        self.words, self.vectors = self.load_word_vectors(file_path=file_path,
                                                     normalize=normalize,
                                                     max_words=max_words)

    def init_word_to_vectors(self, needed_words, normalize=False):
        """
        Initialize:
            {word -> vec} mapping
            {word -> id}  mapping
            [vectors] array
        :param max_loaded_word_vectors: maximum number of words to load from word-vec file
        :param vectors_file_path: file where word-vectors are stored (Glove .txt file)
        :param needed_words: words for which to keep word-vectors
        :param normalize: normalize word vectors
        """
        needed_words.append('<START>')
        needed_words.append('<END>')
        needed_words.append('<UNK>')
        needed_words = set(needed_words)
        
        word_vector_size = self.vectors.shape[-1]
        self.vectors = list(self.vectors)

        present_words = needed_words.intersection(self.words)
        not_present_words = needed_words - present_words
        print('#Present words:', len(present_words), '\t#Not present words:', len(not_present_words))

        not_present_words, not_present_vectors = self.get_not_present_word_vectors(not_present_words=not_present_words,
                                                                                   word_vector_size=word_vector_size,
                                                                                   normalize=normalize)
        words, self.vectors = zip(*[(word, vec) for word, vec in zip(self.words, self.vectors) if word in needed_words])
        words = list(words) + not_present_words
        self.vectors = list(self.vectors) + not_present_vectors        

        print('Initializing word mappings...')
        self.word_to_id  = {word: i   for i, word   in enumerate(words)}
        self.vectors = np.array(self.vectors, copy=False)

        assert len(self.word_to_id) == len(self.vectors)
        print(len(self.word_to_id), 'words in total are now initialized!')

    def init_chars(self, words):
        """
        Init char -> id mapping
        """
        chars = set()
        for word in words:
            chars = chars.union(set(word))        

        self.char_to_id = {char: i+1 for i, char in enumerate(chars)}
        print('Chars:', chars)

    def init_parts_of_speech(self, parts_of_speech):
        self.part_of_speech_to_id = {part: i+1 for i, part in enumerate(parts_of_speech)}
        print('Parts of speech:', parts_of_speech)

    def save_word_vectors(self, file_path):
        np.save(file_path, self.vectors)
    
    def save_word2id_dict(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.word_to_id, f)
    
    def load_word2id_dict(self, file_path):
        with open(file_path, 'rb') as f:
            self.word_to_id = pickle.load(f)

    def get_label(self, sample):
        return NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def label_to_one_hot(self, label):
        label_set = self.get_labels()
        res = np.zeros(shape=(len(label_set)), dtype=np.bool)
        i = label_set.index(label)
        res[i] = 1
        return res
    
    def calculate_exact_match(self, source_words, target_words):
        source_words = [word.lower() for word in source_words if word.lower() not in self.stop_words]
        target_words = [word.lower() for word in target_words if word.lower() not in self.stop_words]
        target_words = set(target_words)

        res = [(word in target_words) for word in source_words]
        return np.array(res, copy=False)
        
    def parse_sentence(self, sentence, max_words, chars_per_word):
        # Words
        words, parts_of_speech = self.get_words_with_part_of_speech(sentence)        
        #words.append('<END>')
        
        #word_ids = [self.word_to_id[word] for word in words]
        word_ids = []
        word_ids.append(self.word_to_id['<START>'])
        for i, word in enumerate(words):
            if word in self.word_to_id:
                word_ids.append(self.word_to_id[word])
            else:
                word_ids.append(self.word_to_id['<UNK>'])
        word_ids.append(self.word_to_id['<END>'])

        # Syntactical features
        syntactical_features = [self.part_of_speech_to_id[part] for part in parts_of_speech]
        syntactical_one_hot = np.eye(len(self.part_of_speech_to_id) + 2)[syntactical_features]  # Convert to 1-hot

        # Chars
        chars = [[self.char_to_id[c] for c in word] for word in words]
        chars = pad_sequences(chars, maxlen=chars_per_word, padding='post', truncating='post')

        return (words, parts_of_speech, np.array(word_ids, copy=False),
                syntactical_features, pad(syntactical_one_hot, max_words),
                pad(chars, max_words))

    def parse_one(self, premise, hypothesis, max_words_p, max_words_h, chars_per_word):
        """
        :param premise: sentence
        :param hypothesis: sentence
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word
        :return: (premise_word_ids, hypothesis_word_ids,
                  premise_chars, hypothesis_chars,
                  premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                  premise_exact_match, hypothesis_exact_match)
        """
        (premise_words, premise_parts_of_speech, premise_word_ids,
         premise_syntactical_features, premise_syntactical_one_hot,
         premise_chars) = self.parse_sentence(sentence=premise, max_words=max_words_p, chars_per_word=chars_per_word)

        (hypothesis_words, hypothesis_parts_of_speech, hypothesis_word_ids,
         hypothesis_syntactical_features, hypothesis_syntactical_one_hot,
         hypothesis_chars) = self.parse_sentence(sentence=hypothesis, max_words=max_words_h, chars_per_word=chars_per_word)

        premise_exact_match    = self.calculate_exact_match(premise_words, hypothesis_words)
        hypothesis_exact_match = self.calculate_exact_match(hypothesis_words, premise_words)

        return (premise_word_ids, hypothesis_word_ids,
                premise_chars, hypothesis_chars,
                premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                premise_exact_match, hypothesis_exact_match)

    def parse(self, input_file_path, max_words_p=33, max_words_h=20, chars_per_word=13):
        """
        :param input_file_path: file to parse data from
        :param max_words_p: maximum number of words in premise
        :param max_words_h: maximum number of words in hypothesis
        :param chars_per_word: number of chars in each word (padding is applied if not enough)
        :return: (premise_word_ids, hypothesis_word_ids,
                  premise_chars, hypothesis_chars,
                  premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
                  premise_exact_match, hypothesis_exact_match)
        """
        # res = [premise_word_ids, hypothesis_word_ids, premise_chars, hypothesis_chars,
        # premise_syntactical_one_hot, hypothesis_syntactical_one_hot, premise_exact_match, hypothesis_exact_match]
        res = [[], [], [], [], [], [], [], [], []]

        data = self.load_data(input_file_path)
        for sample in tqdm(data):
            # As stated in paper: The labels are "entailment", "neutral", "contradiction" and "-".
            # "-"  shows that annotators can't reach consensus with each other, thus removed during training and testing
            label = self.get_label(sample=sample)
            if label == '-':
                continue
            premise, hypothesis = self.get_sentences(sample=sample)
            sample_inputs = self.parse_one(premise, hypothesis,
                                           max_words_p=max_words_p, max_words_h=max_words_h,
                                           chars_per_word=chars_per_word)
            label = self.label_to_one_hot(label=label)

            sample_result = list(sample_inputs) + [label]
            for res_item, parsed_item in zip(res, sample_result):
                res_item.append(parsed_item)

        res[0] = pad_sequences(res[0], maxlen=max_words_p+2, padding='post', truncating='post', value=0.)  # input_word_p
        res[1] = pad_sequences(res[1], maxlen=max_words_h+2, padding='post', truncating='post', value=0.)  # input_word_h
        res[6] = pad_sequences(res[6], maxlen=max_words_p, padding='post', truncating='post', value=0.)  # exact_match_p
        res[7] = pad_sequences(res[7], maxlen=max_words_h, padding='post', truncating='post', value=0.)  # exact_match_h
        
        return res


class SNLIPreprocessor(BasePreprocessor):
    def get_words_with_part_of_speech(self, sentence):
        parts = sentence.split('(')
        words = []
        parts_of_speech = []
        for p in parts:
            if ')' in p:
                res = p.split(' ')
                parts_of_speech.append(res[0])
                ############  ADDED  ############
                word = res[1].replace(')', '')                
                words.append(preprocess_word(word))
                ############  ADDED  ############
        return words, parts_of_speech

    def get_sentences(self, sample):
        return sample['sentence1_parse'], sample['sentence2_parse']

    def get_label(self, sample):
        return sample['gold_label']

    def get_labels(self):
        return 'entailment', 'neutral', 'contradiction'
   
   
def preprocess(p, h, chars_per_word, preprocessor, save_dir, data_paths,
               word_vector_save_path, normalize_word_vectors, 
               max_loaded_word_vectors=None, word_vectors_load_path=None, word2id_save_path=None,
               include_word_vectors=True, include_chars=True,
               include_syntactical_features=True, include_exact_match=True):

    preprocessor.get_all_words_with_parts_of_speech([data_path[1] for data_path in data_paths])
    print('Found', len(preprocessor.unique_words), 'unique words')
    print('Found', len(preprocessor.unique_parts_of_speech), 'unique parts of speech')

    # Init mappings of the preprocessor
    preprocessor.init_word_to_vectors(vectors_file_path=get_word2vec_file_path(word_vectors_load_path),
                                      needed_words=preprocessor.unique_words,
                                      normalize=normalize_word_vectors,
                                      max_loaded_word_vectors=max_loaded_word_vectors)
    preprocessor.init_chars(words=preprocessor.unique_words)
    preprocessor.init_parts_of_speech(parts_of_speech=preprocessor.unique_parts_of_speech)

    # Process and save the data
    preprocessor.save_word2id_dict(word2id_save_path)
    preprocessor.save_word_vectors(word_vector_save_path)
    for dataset, input_path in data_paths:
        data = preprocessor.parse(input_file_path=input_path,
                                  max_words_p=p,
                                  max_words_h=h,
                                  chars_per_word=chars_per_word)

        # Determine which part of data we need to dump
        if not include_exact_match:             del data[6:8]  # Exact match feature
        if not include_syntactical_features:    del data[4:6]  # Syntactical POS tags
        if not include_chars:                   del data[2:4]  # Character features
        if not include_word_vectors:            del data[0:2]  # Word vectors

        data_saver = ChunkDataManager(save_data_path=os.path.join(save_dir, dataset))
        data_saver.save([np.array(item) for item in data])


def preprocess_2sent(sents, p, h, preprocessor, need_exact_match,
               word_vector_save_path, word2id_save_path
               ):

    # get word_to_id
    if os.path.exists(word2id_save_path) and os.path.exists(word_vector_save_path): 
        preprocessor.load_word2id_dict(word2id_save_path)    
        preprocessor.vectors = np.load(word_vector_save_path)        
    else:
        print('Error - MUST have word_vector & word2id files')
        exit(0)
        
    # parse sentences
    sentp, senth = sents
    
    raw_tokens_p = []
    raw_tokens_h = []
    word_ids_p = []
    word_ids_h = []
    unk_id = preprocessor.word_to_id['<UNK>']
    start_id = preprocessor.word_to_id['<START>']
    end_id = preprocessor.word_to_id['<END>']
    for i, _ in enumerate(sentp):
        words_p = []
        words_p.append(start_id)
        words_h = []
        words_h.append(start_id)
        
        raw_tokens = word_tokenize(sentp[i])
        raw_tokens_p.append(raw_tokens)
        for word in raw_tokens:
            word = word.lower()
            if word in preprocessor.word_to_id:
                words_p.append(preprocessor.word_to_id[word])
            else:
                words_p.append(unk_id)
        
        raw_tokens = word_tokenize(senth[i])
        raw_tokens_h.append(raw_tokens)
        for word in raw_tokens:
            word = word.lower()
            if word in preprocessor.word_to_id:
                words_h.append(preprocessor.word_to_id[word])
            else:
                words_h.append(unk_id)
        words_p.append(end_id)
        words_h.append(end_id)
        
        word_ids_p.append(words_p)
        word_ids_h.append(words_h)
            
    
#    word_ids_p = [[preprocessor.word_to_id[word.lower()] for word in word_tokenize(sent)] for sent in sentp]  # list of list
#    word_ids_h = [[preprocessor.word_to_id[word.lower()] for word in word_tokenize(sent)] for sent in senth]
    
    p_padded = pad_sequences(word_ids_p, maxlen=p, padding='post', truncating='post', value=0.)
    h_padded = pad_sequences(word_ids_h, maxlen=p, padding='post', truncating='post', value=0.)
    
    if need_exact_match:
        # exact words
        sents_exact_pair_p = []
        sents_exact_pair_h = []        
        #stop_words = set(stopwords.words('english'))
        
        for i, _ in enumerate(sentp):
            premise_exact_match = preprocessor.calculate_exact_match(raw_tokens_p[i], raw_tokens_h[i])
            hypothesis_exact_match = preprocessor.calculate_exact_match(raw_tokens_h[i], raw_tokens_p[i])
            sents_exact_pair_p.append(premise_exact_match)
            sents_exact_pair_h.append(hypothesis_exact_match)
        
        sents_exact_pair_p = pad_sequences(sents_exact_pair_p, maxlen=p-2, padding='post', truncating='post', value=0.)
        sents_exact_pair_h = pad_sequences(sents_exact_pair_h, maxlen=h-2, padding='post', truncating='post', value=0.)
        
        return [p_padded, h_padded, sents_exact_pair_p, sents_exact_pair_h]
    else:
        return [p_padded, h_padded]


def preprocess_DUC(p, h, preprocessor, data_path, save_dir,
               word_vector_save_path, word_vectors_load_path, word2id_save_path,
               normalize_word_vectors, max_loaded_word_vectors=None):
    """    
    :param p:                       maximum number of words in text
    :param h:                       maximum number of words in hypothesis
    :param preprocessor:            preprocessor
    :param data_path:               root directory of dataset
    :param word_vector_save_path:   path to save a word_vector (only vectors)
    :param word_vectors_load_path:  path to load a Glove word_vector
    :param word2id_save_path:       path to save a word2id
    :param normalize_word_vectors:  normalize word_vector or not
    :param max_loaded_word_vectors: maximum limitation of number of words
    
    :return: (premise_word_ids, hypothesis_word_ids,
              premise_chars, hypothesis_chars,
              premise_syntactical_one_hot, hypothesis_syntactical_one_hot,
              premise_exact_match, hypothesis_exact_match)
    """ 
    
    #dirs = [x[0] for x in os.walk(data_path)]  # os.walk finds all sub-directoreis
    #folders = [d for d in dirs if os.path.basename(d).startswith('d')]
    folders = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,d)) and re.search('^d[0-9]+', d)]
    
    if os.path.exists(word2id_save_path) and os.path.exists(word_vector_save_path):
        preprocessor.load_word2id_dict(word2id_save_path)    
        preprocessor.vectors = np.load(word_vector_save_path)
    else:
        
        preprocessor.get_all_words_DUC(folders)
        print('Found', len(preprocessor.unique_words), 'unique words from DUC')
        
        preprocessor.init_word_to_vectors(vectors_file_path=get_word2vec_file_path(word_vectors_load_path),
                                          needed_words=preprocessor.unique_words,
                                          normalize=normalize_word_vectors,
                                          max_loaded_word_vectors=max_loaded_word_vectors)
        preprocessor.save_word2id_dict(word2id_save_path)
        preprocessor.save_word_vectors(word_vector_save_path)
    
    save_path = os.path.join(data_path, save_dir)
    
    for dir in folders:
        file_name = os.path.basename(dir)
        file_path = os.path.join(dir, file_name+'.txt')
        sents = preprocessor.load_txt_data(file_path = file_path)
        
        # list of list: processed words in sentences
        sents_processed = []
        for sent in sents:
            sent_tmp = []
            for word in sent.split():
                w = preprocessor.word_to_id[word.translate(None, ',.')]
                sent_tmp.append(w)
            sents_processed.append(sent_tmp)
            
        sents_pair_p = []
        sents_pair_h = []
        sent_len = len(sents_processed)
        for i in range(sent_len):
            for j in range(i+1, sent_len):
                sents_pair_p.append(sents_processed[i])
                sents_pair_h.append(sents_processed[j])
        
        data = [sents_pair_p, sents_pair_h]
        data[0] = pad_sequences(data[0], maxlen=p, padding='post', truncating='post', value=0.)
        data[1] = pad_sequences(data[1], maxlen=h, padding='post', truncating='post', value=0.)
        
        # save as npy
        data_saver = ChunkDataManager(save_data_path=os.path.join(save_path, file_name))
        data_saver.save([np.array(item) for item in data])
        
        
def preprocess_unified(p, h, preprocessors, save_dir, 
                       data_paths, dataset_to_save,
                       word_vectors_load_path=None, normalize_word_vectors=False,                       
                       voca_size=[50000], voca_dim=300,
                       data_root_dir='data',
                       word_vector_save_path=None, word2id_save_path=None,
                       max_loaded_word_vectors=None,  
                       include_word_vectors=True,
                       include_exact_match=False,
                       include_chars=False,
                       include_syntactical_features=False):    
    
    unified_preprocessor = SNLIPreprocessor()
    
    # load word vector
    if voca_dim == 300:
        word2vec_func = get_word2vec_file_path
    elif voca_dim == 100:
        word2vec_func = get_word2vec_100d_file_path
    unified_preprocessor.call_load_word_vector(file_path=word2vec_func(word_vectors_load_path),
                                                     normalize=normalize_word_vectors,
                                                     max_words=max_loaded_word_vectors)
    
    if False and os.path.exists(word2id_save_path) and os.path.exists(word_vector_save_path):
        unified_preprocessor.load_word2id_dict(word2id_save_path)    
        unified_preprocessor.vectors = np.load(word_vector_save_path)
    else:
        # get all tokenized words from all dataset    
        for dataset, ppr in preprocessors.iteritems():
            data_path = data_paths[dataset]
            if dataset.endswith('nli'):
                if dataset.startswith('s'):
                    train_path = os.path.join(data_path, dataset+'_1.0_train.jsonl')
                    test_path  = os.path.join(data_path, dataset+'_1.0_test.jsonl')
                    dev_path   = os.path.join(data_path, dataset+'_1.0_dev.jsonl')                
                    paths = [('train', train_path), ('test', test_path), ('dev', dev_path)]
                elif dataset.startswith('m'):
                    train_path = os.path.join(data_path, dataset+'_1.0_train.jsonl')
                    test_matched_path  = os.path.join(data_path, dataset+'_1.0_dev_matched.jsonl')
                    test_mismatched_path   = os.path.join(data_path, dataset+'_1.0_dev_mismatched.jsonl')                
                    paths = [('train', train_path), ('test_matched', test_matched_path), ('test_mismatched', test_mismatched_path)]
                
                ppr.get_all_words_with_parts_of_speech([path[1] for path in paths])
                print('Found {} unique words, {} unique parts of speech from {}'.format(len(ppr.unique_words), 
                      len(ppr.unique_parts_of_speech), dataset))
                unified_preprocessor.unique_parts_of_speech = unified_preprocessor.unique_parts_of_speech.union(ppr.unique_parts_of_speech)
            elif dataset.startswith('DUC'):
                folders = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,d)) and re.search('^d[0-9]+', d)]
                ppr.get_all_words_DUC(folders)
                print('Found {} unique words from {}'.format(len(ppr.unique_words), dataset))
            elif dataset.startswith('cnn'):                
                paths = [('train', data_path), ('test', data_path), ('val', data_path)]
                ppr.gen_all_pairs(paths)
                ppr.get_all_words()
                print('Found {} unique words from {}'.format(len(ppr.unique_words), dataset))
            
            unified_preprocessor.unique_words = unified_preprocessor.unique_words.union(ppr.unique_words)
            
            temp_dict = dict()
            for d in (unified_preprocessor.unique_words_freq, ppr.unique_words_freq):
                for word, freq in d.items():
                    if word in temp_dict:
                        temp_dict[word] += freq
                    else:
                        temp_dict[word] = freq
            unified_preprocessor.unique_words_freq = temp_dict
        
        # sort to get 50k frequent words
        sorted_dict = sorted(unified_preprocessor.unique_words_freq.items(), key=operator.itemgetter(1), reverse=True)
        
    
    for v_size in voca_size:        
        voca_name = 'voca'+str(v_size/1000)+'k_'+str(voca_dim)+'d'
        
        selected_dict = sorted_dict[:v_size]     # list of tuples
        unified_preprocessor.unique_words_voca = [word[0] for word in selected_dict]
        
        print('Found {} unique words, {} unique parts of speech from unified'.format(len(unified_preprocessor.unique_words),
                                                                                     len(unified_preprocessor.unique_parts_of_speech)))
        print('Found {} unique words freq(dict), {} {} words(list)'.format(len(unified_preprocessor.unique_words_freq), 
                                                                    len(unified_preprocessor.unique_words_voca), voca_name))
        n_print = 20
        print('Top {} frequent words'.format(n_print))
        for it in range(n_print):
            print('{} - {}:{}'.format(it+1, selected_dict[it][0], selected_dict[it][1]))
        
        # initialize w2v, word2id
        # ADD <START>, <END>, UNK, ZERO into the vocabulary
        unified_preprocessor.init_word_to_vectors(needed_words=unified_preprocessor.unique_words_voca, normalize=normalize_word_vectors)        
        unified_preprocessor.init_chars(words=unified_preprocessor.unique_words)
        unified_preprocessor.init_parts_of_speech(parts_of_speech=unified_preprocessor.unique_parts_of_speech)
        
        #dirname = os.path.dirname(word2id_save_path)        
        word2id_path = os.path.join(args.data_root_dir, 'word2id_'+voca_name+'_unified.pkl')
        wordvec_path = os.path.join(args.data_root_dir, 'word-vectors_'+voca_name+'_unified.npy')
        unified_preprocessor.save_word2id_dict(word2id_path)
        unified_preprocessor.save_word_vectors(wordvec_path)
    
        # assign word id and save 
        for dataset in dataset_to_save:
            print('***** [{}] data saving *****'.format(dataset))
            data_path = data_paths[dataset]
            
            if dataset.endswith('nli'):
                if dataset.startswith('s'):
                    train_path = os.path.join(data_path, dataset+'_1.0_train.jsonl')
                    test_path  = os.path.join(data_path, dataset+'_1.0_test.jsonl')
                    dev_path   = os.path.join(data_path, dataset+'_1.0_dev.jsonl')                
                    paths = [('train_'+voca_name, train_path), ('test_'+voca_name, test_path), ('dev_'+voca_name, dev_path)]
                elif dataset.startswith('m'):
                    train_path = os.path.join(data_path, dataset+'_1.0_train.jsonl')
                    test_matched_path  = os.path.join(data_path, dataset+'_1.0_dev_matched.jsonl')
                    test_mismatched_path   = os.path.join(data_path, dataset+'_1.0_dev_mismatched.jsonl')                
                    paths = [('train_'+voca_name, train_path), ('test_matched_'+voca_name, test_matched_path), 
                             ('test_mismatched_'+voca_name, test_mismatched_path)]
                    
                for dataset_var, input_path in paths:
                    data = unified_preprocessor.parse(input_file_path=input_path,
                                                      max_words_p=p,
                                                      max_words_h=h)
            
                    # Determine which part of data we need to dump
                    if not include_exact_match:             del data[6:8]  # Exact match feature
                    if not include_syntactical_features:    del data[4:6]  # Syntactical POS tags
                    if not include_chars:                   del data[2:4]  # Character features
                    if not include_word_vectors:            del data[0:2]  # Word vectors
            
                    data_saver = ChunkDataManager(save_data_path=os.path.join(data_path, dataset_var))
                    data_saver.save([np.array(item) for item in data])
                    
            elif dataset.startswith('cnn'):                
                cnn_dm = preprocessors['cnn_dm']
                cnn_dm.assign_w2id(unified_preprocessor.word_to_id)
                
                data_path_1up = os.path.dirname(data_path)    # data/cnn_dm                
                paths = ['train_'+voca_name, 'test_'+voca_name, 'val_'+voca_name]
                
                for i, path_name in enumerate(paths):
                    data_saver = ChunkDataManager(save_data_path=os.path.join(data_path_1up, path_name))                    
                    data_saver.save([np.array(item) for item in cnn_dm.data_all[i]])
                    
            elif dataset.startswith('DUC'):
                DUC_save_dir = 'data_'+voca_name
                save_path = os.path.join(data_path, DUC_save_dir) #save_dir
                folders = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,d)) and re.search('^d[0-9]+', d)]
                for dir in folders:
                    file_name = os.path.basename(dir)
                    file_path = os.path.join(dir, file_name+'.txt')
                    sents = unified_preprocessor.load_txt_data(file_path = file_path)
                    
                    # list of list: processed words in sentences
                    sents_processed = []
                    sents_raw_words = []
                    for sent in sents:
                        sent_tmp = []
                        sent_tmp.append(unified_preprocessor.word_to_id['<START>'])
                        word_tokens = word_tokenize(sent)
                        
                        for word in word_tokens:
                            word = preprocess_word(word)
                            if word in unified_preprocessor.word_to_id:
                                sent_tmp.append(unified_preprocessor.word_to_id[word])
                            else:
                                sent_tmp.append(unified_preprocessor.word_to_id['<UNK>'])
                        sent_tmp.append(unified_preprocessor.word_to_id['<END>'])
                        sents_processed.append(sent_tmp)
                        sents_raw_words.append(word_tokens)
                    
                    # test pair for similarity measure
                    sents_pair_p = []
                    sents_pair_h = []
                    sents_exact_pair_p = []
                    sents_exact_pair_h = []
                    sent_len = len(sents_processed)
                    for i in range(sent_len):
                        for j in range(i+1, sent_len):
                            sents_pair_p.append(sents_processed[i])
                            sents_pair_h.append(sents_processed[j])
                    
                            # exact words
                            premise_exact_match = unified_preprocessor.calculate_exact_match(sents_raw_words[i], sents_raw_words[j])
                            hypothesis_exact_match = unified_preprocessor.calculate_exact_match(sents_raw_words[j], sents_raw_words[i])
                            sents_exact_pair_p.append(premise_exact_match)
                            sents_exact_pair_h.append(hypothesis_exact_match)
                    
                    data = []
                    w2id_p = pad_sequences(sents_pair_p, maxlen=p+2, padding='post', truncating='post', value=0.)
                    w2id_h = pad_sequences(sents_pair_h, maxlen=h+2, padding='post', truncating='post', value=0.)
                    data.append(w2id_p)
                    data.append(w2id_h)
                    
                    sents_exact_pair_p = pad_sequences(sents_exact_pair_p, maxlen=p, padding='post', truncating='post', value=0.)
                    sents_exact_pair_h = pad_sequences(sents_exact_pair_h, maxlen=h, padding='post', truncating='post', value=0.)
                    data.append(sents_exact_pair_p)
                    data.append(sents_exact_pair_h)
                    
                    # save as npy
                    data_saver = ChunkDataManager(save_data_path=os.path.join(save_path, file_name))
                    data_saver.save([np.array(item) for item in data])
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p',              default=42,         help='Maximum words in premise +2 for <START> & <END>',  type=int)
    parser.add_argument('--h',              default=42,         help='Maximum words in hypothesis',         type=int)    
    parser.add_argument('--chars_per_word', default=16,         help='Number of characters in one word',    type=int)
    parser.add_argument('--max_word_vecs',  default=None,       help='Maximum number of word vectors',      type=int)
    
    # not used for unified
    parser.add_argument('--save_dir',       default='data_voca50k_100d/',    help='Save directory of data',      type=str)
    parser.add_argument('--word_vec_load_path', default=None,   help='Path to load word vectors',           type=str)    
    parser.add_argument('--word_vec_save_path', default='data/word-vectors_voca50k_100d_unified.npy', help='Path to save vectors', type=str)
    parser.add_argument('--word2id_save_path',  default='data/word2id_voca50k_100d_unified.pkl', help='Path to save word2id',    type=str)
    
    
    parser.add_argument('--data_root_dir',  default='data/',    help='data directory',                      type=str)
    parser.add_argument('--voca_size',      default=[50000],      help='vocabulary size') # [10000,50000]
    parser.add_argument('--voca_dim',       default=300,        help='dimension of voca embedding',         type=int)    
    parser.add_argument('--dataset',        default='unified',  help='Which preprocessor to use',           type=str)
    
    parser.add_argument('--normalize_word_vectors',      action='store_true')
    parser.add_argument('--omit_word_vectors',           action='store_true')
    parser.add_argument('--omit_exact_match',            action='store_true')
    parser.add_argument('--omit_chars',                  action='store_true')
    parser.add_argument('--omit_syntactical_features',   action='store_true')
    
    args = parser.parse_args()

    start_timer = time.time()
    if args.dataset == 'snli':
        snli_preprocessor = SNLIPreprocessor()
        path = get_snli_file_path()
        train_path = os.path.join(path, 'snli_1.0_train.jsonl')
        test_path  = os.path.join(path, 'snli_1.0_test.jsonl')
        dev_path   = os.path.join(path, 'snli_1.0_dev.jsonl')

        preprocess(p=args.p, h=args.h, chars_per_word=args.chars_per_word,
                   preprocessor=snli_preprocessor,
                   save_dir=args.save_dir,
                   data_paths=[('train', train_path), ('test', test_path), ('dev', dev_path)],
                   word_vectors_load_path=args.word_vec_load_path,
                   normalize_word_vectors=args.normalize_word_vectors,
                   word_vector_save_path=args.word_vec_save_path,
                   word2id_save_path=args.word2id_save_path,
                   max_loaded_word_vectors=args.max_word_vecs,
                   include_word_vectors=not args.omit_word_vectors,
                   include_chars=not args.omit_chars,
                   include_syntactical_features=not args.omit_syntactical_features,
                   include_exact_match=not args.omit_exact_match)
    if args.dataset == 'mnli':
        mnli_preprocessor = SNLIPreprocessor()
        path = get_multinli_file_path()
        train_path = os.path.join(path, 'multinli_1.0_train.jsonl')
        test_path  = os.path.join(path, 'multinli_1.0_dev_matched.jsonl')
        dev_path   = os.path.join(path, 'multinli_1.0_dev_mismatched.jsonl')

        preprocess(p=args.p, h=args.h, chars_per_word=args.chars_per_word,
                   preprocessor=mnli_preprocessor,
                   save_dir=args.save_dir,
                   data_paths=[('train', train_path), ('test', test_path), ('dev', dev_path)],
                   word_vectors_load_path=args.word_vec_load_path,
                   normalize_word_vectors=args.normalize_word_vectors,
                   word_vector_save_path=args.word_vec_save_path,
                   word2id_save_path=args.word2id_save_path,
                   max_loaded_word_vectors=args.max_word_vecs,
                   include_word_vectors=not args.omit_word_vectors,
                   include_chars=not args.omit_chars,
                   include_syntactical_features=not args.omit_syntactical_features,
                   include_exact_match=not args.omit_exact_match)
    elif args.dataset == 'DUC':
        duc_preprocessor = BasePreprocessor()

        preprocess_DUC(p=args.p, h=args.h, 
                       preprocessor=duc_preprocessor,
                       data_path='/media/swcho/352843D4280F4AF5/Research/text_summarization/data/2004',
                       save_dir=args.save_dir,
                       word_vector_save_path=args.word_vec_save_path,
                       word_vectors_load_path='data/glove.840B.300d.txt',
                       word2id_save_path=args.word2id_save_path, #'data/word2id_DUC2004.pkl',
                       normalize_word_vectors=False,
                       max_loaded_word_vectors=None)
    elif args.dataset == 'unified':
        # pre-processor
        snli_preprocessor     = SNLIPreprocessor()
        mnli_preprocessor     = SNLIPreprocessor()
        duc2003_preprocessor  = BasePreprocessor()
        duc2004_preprocessor  = BasePreprocessor()
        cnn_dm_preprocessor  = cnn_dm_data(args.p)
        
        preprocessors = {}
#        preprocessors['snli'] = snli_preprocessor
#        preprocessors['multinli'] = mnli_preprocessor
        preprocessors['DUC2003'] = duc2003_preprocessor
        preprocessors['DUC2004'] = duc2004_preprocessor
        preprocessors['cnn_dm'] = cnn_dm_preprocessor
        
        # data path
#        path_snli = get_snli_file_path()
#        path_mnli = get_multinli_file_path()
        path_DUC2003 = os.path.join(args.data_root_dir, '2003')
        path_DUC2004 = os.path.join(args.data_root_dir, '2004')
        path_cnn_dm = os.path.join(args.data_root_dir, 'cnn_dm', 'bin')
        
        data_paths = {}
#        data_paths['snli'] = path_snli
#        data_paths['multinli'] = path_mnli
        data_paths['DUC2003'] = path_DUC2003
        data_paths['DUC2004'] = path_DUC2004
        data_paths['cnn_dm'] = path_cnn_dm
        
        # dataset to save
        dataset_to_save = set()
#        dataset_to_save.add('snli')
#        dataset_to_save.add('multinli')
        dataset_to_save.add('DUC2003')
        dataset_to_save.add('DUC2004')
        dataset_to_save.add('cnn_dm')

        preprocess_unified(p=args.p, h=args.h,                           
                           preprocessors=preprocessors,
                           save_dir=args.save_dir,
                           data_paths=data_paths,
                           dataset_to_save=dataset_to_save,
                           word_vectors_load_path=args.word_vec_load_path,
                           normalize_word_vectors=args.normalize_word_vectors,
                           voca_size=args.voca_size,
                           voca_dim=args.voca_dim,
                           data_root_dir=args.data_root_dir,
                           word_vector_save_path=args.word_vec_save_path,
                           word2id_save_path=args.word2id_save_path,
                           max_loaded_word_vectors=args.max_word_vecs,
                           include_word_vectors=not args.omit_word_vectors,
                           include_exact_match=not args.omit_exact_match,
                           include_chars=args.omit_chars,
                           include_syntactical_features=args.omit_syntactical_features
                           )
    else:
        raise ValueError('couldn\'t find implementation for specified dataset')

    end_timer = time.time()
    print('Elapsed time for {} preprocessing: {:4.3f}'.format(args.dataset, end_timer-start_timer))
