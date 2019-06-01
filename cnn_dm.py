from tqdm import tqdm
import glob
#from data import example_generator    # The module "data" is from Abigail See's code
import json
import numpy as np

import random
import struct
from tensorflow.core.example import example_pb2
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import np_utils

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


names_to_types = [('raw_article_sents', 'string_list'), ('similar_source_indices', 'delimited_list_of_lists'), 
                  ('summary_text', 'string'), ('corefs', 'json')]


class cnn_dm_data(object):
    def __init__(self, max_len):
        self.max_len = max_len
        
        self.unique_words = set()
        self.unique_words_freq = {}
        
        self.x_train_sent_pairs = []        
        self.y_train = []
        self.x_test_sent_pairs = []
        self.x_test = []
        self.x_dev_sent_pairs = []
        self.y_dev = []
        
        self.unique_words = set()
        self.unique_words_freq = {}
        
        self.stop_words = set(stopwords.words('english'))
    
    def assign_w2id(self, w2id):
        def calculate_exact_match(source_words, target_words):
            source_words = [word.lower() for word in source_words if word.lower() not in self.stop_words]
            target_words = [word.lower() for word in target_words if word.lower() not in self.stop_words]
            target_words = set(target_words)

            res = [(word in target_words) for word in source_words]
            return np.array(res, copy=False)
        
        self.data_all = []
        for i, sent_pairs in enumerate([self.x_train_sent_pairs, self.x_test_sent_pairs, self.x_dev_sent_pairs]):
            sents_w2id_p = []
            sents_w2id_h = []
            exact_match_ph = []
            exact_match_hp = []
            for sents in tqdm(sent_pairs):                
                for s, sent in enumerate(sents):     # [p, h]
                    word_tokens = word_tokenize(sent)
                    sent_w2ids = []
                    sent_w2ids.append(w2id['<START>'])
                    for word in word_tokens:
                        word = word.lower()
                        if word in w2id:
                            sent_w2ids.append(w2id[word])
                        else:
                            sent_w2ids.append(w2id['<UNK>'])
                    sent_w2ids.append(w2id['<END>'])
                    if s == 0:
                        sents_w2id_p.append(sent_w2ids)
                        raw_words_p = word_tokens
                    else:
                        sents_w2id_h.append(sent_w2ids)
                        raw_words_h = word_tokens
                        
                exact_match_ph.append( calculate_exact_match(raw_words_p, raw_words_h) )
                exact_match_hp.append( calculate_exact_match(raw_words_h, raw_words_p) )
                
        
            data = []
            w2id_p = pad_sequences(sents_w2id_p, maxlen=self.max_len+2, padding='post', truncating='post', value=0.)
            w2id_h = pad_sequences(sents_w2id_h, maxlen=self.max_len+2, padding='post', truncating='post', value=0.)
            data.append(w2id_p)
            data.append(w2id_h)
            
            exact_ph = pad_sequences(exact_match_ph, maxlen=self.max_len, padding='post', truncating='post', value=0.)
            exact_hp = pad_sequences(exact_match_hp, maxlen=self.max_len, padding='post', truncating='post', value=0.)
            data.append(exact_ph)
            data.append(exact_hp)
            
            if i==0:
                data.append( np.array(self.y_train, dtype=np.bool) )
                print ('[cnn_dm] train data assigned')
            elif i==1:
                data.append( np.array(self.y_test, dtype=np.bool) )
                print ('[cnn_dm] test data assigned')
            elif i==2:
                data.append( np.array(self.y_dev, dtype=np.bool) )
                print ('[cnn_dm] val data assigned')
            self.data_all.append(data)
                            
    def gen_all_pairs(self, file_paths):
        '''
        file_paths: [['train', path], ['test', path], ['val', path]]
        '''
        for split, file_path in file_paths:
            if split == 'train':
                print ('[cnn_dm] begin generating sentence pairs - train')
                self.x_train_sent_pairs, self.y_train = self.generate_sentence_pairs(file_path+'/'+split+'*')
            elif split == 'test':
                print ('[cnn_dm] begin generating sentence pairs - test')
                self.x_test_sent_pairs, self.y_test = self.generate_sentence_pairs(file_path+'/'+split+'*')
            elif split == 'val':
                print ('[cnn_dm] begin generating sentence pairs - val')
                self.x_dev_sent_pairs, self.y_dev = self.generate_sentence_pairs(file_path+'/'+split+'*')
        
    def generate_sentence_pairs(self, file_path):        
        source_files = sorted(glob.glob(file_path))
        
        sent_pairs = []
        label = []
        #prev_raw_article_sents = []            
        total = len(source_files) * 1000
        example_generator = self.example_generator(file_path, True)            
        for example in tqdm(example_generator, total=total):
            raw_article_sents, similar_source_indices_list, summary_text = self.unpack_tf_example(example, names_to_types[:-1])            
            
            summary_text_list = [sent.strip() for sent in summary_text.split('\n') if len(sent.strip())>0]            
            
            #print ('len summary:{}\t len_indices:{}'.format(len(summary_text_list), len(similar_source_indices_list)))
            #assert (len(summary_text_list) == len(similar_source_indices_list))
                        
            # sentences in same document except summary sentences
            sum_index = set([idx for si in similar_source_indices_list for idx in si])
            all_index = range(len(raw_article_sents))
            random.shuffle(all_index)
            cand_index = [ind for ind in all_index if ind not in sum_index]
            
            #print similar_source_indices_list, len(raw_article_sents), len(cand_index)
            
            for si, sum_sent in enumerate(summary_text_list):
                neg_ind = 0
                if len(similar_source_indices_list[si]) > 0:
                    if neg_ind < len(cand_index):
                        # positive sentence pair
                        pos_index = similar_source_indices_list[si][0]      # most similar one
                        sent_pairs.append([sum_sent, raw_article_sents[pos_index]])
                        label.append(1)
                                            
                        # negative sentence pair                    
                        neg_index = cand_index[neg_ind]
                        sent_pairs.append([sum_sent, raw_article_sents[neg_index]])
                        label.append(0)
                        neg_ind += 1
                        
                        if 0:   # debug print out
                            print (sum_sent, raw_article_sents[pos_index])
                            print (sum_sent, raw_article_sents[neg_index])
                            raw_input('enter')
        
        print ('[cnn_dm] # of generated pairs: {}'.format(len(sent_pairs)))
        return sent_pairs, label        

    def get_all_words(self):
        all_words = []
        
        for sent_pairs in [self.x_train_sent_pairs, self.x_test_sent_pairs, self.x_dev_sent_pairs]:            
            for sents in tqdm(sent_pairs):
                for sent in sents:
                    word_tokens = word_tokenize(sent)                
                    for word in word_tokens: #sent.split():                    
                        all_words.append(word.lower())                        
            
        self.unique_words = set(all_words)        
        
        for word in all_words:
            if word in self.unique_words_freq:
                self.unique_words_freq[word] += 1
            else:
                self.unique_words_freq[word] = 1
        
        #print('Found {} unique words'.format(len(self.unique_words)))

    def example_generator(self, data_path, single_pass):
      """Generates tf.Examples from data files.
    
        Binary data format: <length><blob>. <length> represents the byte size
        of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
        the tokenized article text and summary.
    
      Args:
        data_path:
          Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
        single_pass:
          Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.
    
      Yields:
        Deserialized tf.Example.
      """
      while True:
        filelist = glob.glob(data_path) # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
        if single_pass:
          filelist = sorted(filelist)
        else:
          random.shuffle(filelist)
        for f in filelist:
          reader = open(f, 'rb')
          while True:
            len_bytes = reader.read(8)
            if not len_bytes: break # finished reading this file
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)
        if single_pass:
          print "example_generator completed reading all datafiles. No more data."
          break
    
    def decode_text(self, text):
        try:
            text = text.decode('utf-8')
        except:
            try:
                text = text.decode('latin-1')
            except:
                raise
        return text
    
    def unpack_tf_example(self, example, names_to_types):
        def get_string(name):
            return self.decode_text(example.features.feature[name].bytes_list.value[0])
        def get_string_list(name):
            texts = get_list(name)
            texts = [self.decode_text(text) for text in texts]
            return texts
        def get_list(name):
            return example.features.feature[name].bytes_list.value
        def get_delimited_list(name):
            text = get_string(name)
            return text.split(' ')
        def get_delimited_list_of_lists(name):
            text = get_string(name)
            return [[int(i) for i in (l.split(' ') if l != '' else [])] for l in text.split(';')]
        def get_delimited_list_of_tuples(name):
            list_of_lists = get_delimited_list_of_lists(name)
            return [tuple(l) for l in list_of_lists]
        def get_json(name):
            text = get_string(name)
            return json.loads(text)
        func = {'string': get_string,
                'list': get_list,
                'string_list': get_string_list,
                'delimited_list': get_delimited_list,
                'delimited_list_of_lists': get_delimited_list_of_lists,
                'delimited_list_of_tuples': get_delimited_list_of_tuples,
                'json': get_json}
    
        res = []
        for name, type in names_to_types:
            if name not in example.features.feature:
                raise Exception('%s is not a feature of TF Example' % name)
            res.append(func[type](name))
        return res


if __name__ == '__main__':    
    dataset_split = 'test'
    source_dir = './data/cnn_dm/bin'
    
    cnn_dm = cnn_dm_data(42)
    paths = [('test', source_dir), ('train', source_dir), ('val', source_dir)]
#    paths = [('train', source_dir)]
    cnn_dm.gen_all_pairs(paths)
#    cnn_dm.get_all_words()
  