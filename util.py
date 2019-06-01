from __future__ import print_function

import os
from os import path
import re
from shutil import copy

import numpy as np
from keras import backend as K
from keras.utils import get_file
from tqdm import tqdm

from matplotlib import pyplot as plt
import csv
import math
import pandas


def load_sts_dataset(filename):
    # Loads a subset of the STS dataset into a DataFrame. In particular both
    # sentences and their human rated similarity score.
    sent_pairs = []
    with open(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            # (sent_1, sent_2, similarity_score)
            sent_pairs.append((ts[0], ts[5], ts[6], float(ts[4])))
    return pandas.DataFrame(sent_pairs, columns=["genre", "sent_1", "sent_2", "sim"])


def download_and_load_sts_data(STS_train=False):
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')
    stsbm_dir = cache_dir + '/stsbenchmark/'
    
    if not os.path.exists(stsbm_dir):           
        sts_dataset = get_file(
          fname="/tmp/Stsbenchmark.tar.gz",
          origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
          cache_dir=cache_dir,
          cache_subdir='',
          extract=True)        
        #sts_dataset: "/tmp/Stsbenchmark.tar.gz"
        
    sts_dev = load_sts_dataset(os.path.join(stsbm_dir, "sts-dev.csv"))
    sts_test = load_sts_dataset(os.path.join(stsbm_dir, "sts-test.csv"))
    if STS_train:
        sts_train = load_sts_dataset(os.path.join(stsbm_dir, "sts-train.csv"))
        return sts_dev, sts_test, sts_train

    return sts_dev, sts_test


def get_snli_file_path():
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')
    download_url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    snli_dir = cache_dir + '/snli_1.0/'

    if os.path.exists(snli_dir):
        return snli_dir

    get_file('/tmp/snli_1.0.zip',
             origin=download_url,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    return snli_dir

def get_multinli_file_path():
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')
    download_url = 'https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'
    multinli_dir = cache_dir + '/multinli_1.0/'    

    if os.path.exists(multinli_dir):
        return multinli_dir

    get_file('/tmp/multinli_1.0.zip',
             origin=download_url,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    return multinli_dir

def get_word2vec_file_path(file_path):
    if file_path is not None and path.exists(file_path):
        return file_path

    download_url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')
    glove_file_path = cache_dir + '/glove.840B.300d.txt'

    if path.exists(glove_file_path):
        return glove_file_path

    filename = '/tmp/glove.zip'
    get_file(filename,
             origin=download_url,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    os.remove(filename)
    return glove_file_path

def get_word2vec_100d_file_path(file_path):
    if file_path is not None and path.exists(file_path):
        return file_path

    download_url = 'http://nlp.stanford.edu/data/wordvecs/glove.6B.zip'  #'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')
    glove_file_path = cache_dir + '/glove.6B.100d.txt'  #glove.twitter.27B.100d.txt'

    if path.exists(glove_file_path):
        return glove_file_path

    filename = '/tmp/glove.zip'
    get_file(filename,
             origin=download_url,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    os.remove(filename)
    return glove_file_path


class ChunkDataManager(object):
    def __init__(self, load_data_path=None, save_data_path=None):
        self.load_data_path = load_data_path
        self.save_data_path = save_data_path

    def load(self, load_list=None):
        data = []
        for file in tqdm(sorted(os.listdir(self.load_data_path))):
            if not file.endswith('.npy'):
                continue
            if (load_list is not None) and (int(os.path.splitext(file)[0]) in load_list):
                data.append(np.load(self.load_data_path + '/' + file))
        return data

    def save(self, data):
        if not os.path.exists(self.save_data_path):
            os.makedirs(self.save_data_path)
        print('Saving data of shapes:', [item.shape for item in data])
        for i, item in tqdm(enumerate(data)):
            np.save(self.save_data_path + '/' + str(i) + '.npy', item)


def broadcast_last_axis(x):
    """
    :param x tensor of shape (batch, a, b)
    :returns broadcasted tensor of shape (batch, a, b, a)
    """
    y = K.expand_dims(x, 1) * 0
    y = K.permute_dimensions(y, (0, 1, 3, 2))
    return y + K.expand_dims(x)


def plot_log(filename, show=True, filesave=True):
    # load data
    keys = []
    values = []
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        
        # find index column        
        for i, key in enumerate(keys):
            if not key.find('loss') >=0 and not key.find('acc') >=0:
                index_key = i        
        values[:,index_key] += 1

    fig = plt.figure(figsize=(10,4)) # figsize=(4,6)
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(121)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 : #and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, index_key], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation loss')

    fig.add_subplot(122)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, index_key], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()
    
    if filesave:
        path = os.path.dirname(filename)
        pdf_name = '/loss_acc.pdf'
        jpg_name = '/loss_acc.jpg'        
        fig.savefig(path+pdf_name, bbox_inches='tight')
        fig.savefig(path+jpg_name, bbox_inches='tight')


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

    
def copy_predfile(src_dir, dest_dir, re_exp):
    # src_dir = './data/2003/data'    
    mat_filename = 'prediction.mat'
    folders = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,d)) and re.search(re_exp, d)]
    
    i = 0
    for folder in folders:
        src_matfile = os.path.join(src_dir, folder, mat_filename)
        
        dest_folder = os.path.join(dest_dir, folder)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        
        dest_matfile = os.path.join(dest_folder, mat_filename)
        copy(src_matfile, dest_matfile)
        i += 1        
    print ('total {} files are copied...'.format(i))


def get_STS_sents():
    sts_dev, sts_test, sts_train = download_and_load_sts_data(STS_train=True)
    text_a = sts_train['sent_1'].tolist()
    text_b = sts_train['sent_2'].tolist()
    for i in range(len(text_a)):
        print (text_a[i])
        print (text_b[i])
        raw_input('Enter to continue')
    
def copy_DUC_TAC_data():
    if 1:
        for dataset in ['2003', '2004']:
            copy_predfile('./data/'+dataset+'/predictions/capsnet', 
                          './data/'+dataset+'/data_voca50k_300d', '^d\d+') # cnndm-negsame / snli
    else:
        for dataset in ['./data/TAC_Data/s080910_gen_proc', './data/TAC_Data/s11_gen_proc']:
            copy_predfile(dataset+'/predictions/capsnet',
                          dataset+'/data_voca50k_300d', '^D\d+\-[A-B]')

if __name__=="__main__":
    #plot_log('result/log.csv')
#    copy_predfile('./data/2003/data', './data/2003/predictions/model2')
    
    copy_DUC_TAC_data()
#    get_STS_sents()