#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 21:17:09 2018

@author: swcho
"""

from __future__ import print_function

import os
import numpy as np
import argparse
import time
import logging
import re
import scipy.io as sio
import scipy
import json

from keras import callbacks
from keras.utils import plot_model

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from util import ChunkDataManager, plot_log, copy_predfile, download_and_load_sts_data
from model import CapsNetTextSim
from preprocess import preprocess_2sent, SNLIPreprocessor



def generate_batch(data, batch_size, shuffle=False):
    while True:
        size = len(data)       
        
        # loop once per epoch
        if shuffle:
            indices = np.random.permutation(np.arange(size))            
        else:
            indices = np.arange(size)            
        
        num_batches = size // batch_size
        if size % batch_size != 0:
            num_batches += 1
        for bid in range(num_batches):
            # loop once per batch            
            batch_index = indices[bid * batch_size : min((bid + 1) * batch_size, size)]
            
            x_p, x_h, y_labels, y_dec_p, y_dec_h = [], [], [], [], []
            for i in range(batch_size):
                p, h, label, sequences_p, sequences_h = data[batch_index[i]]
                                
                x_p.append(p)
                x_h.append(h)
                y_labels.append(label)
                y_dec_p.append(sequences_p)
                y_dec_h.append(sequences_h)
            
            X_p = np.array(x_p, dtype='int32', copy=False)
            X_h = np.array(x_h, dtype='int32', copy=False)
            Y = np.array(y_labels, dtype='int32', copy=False)
            Y_p = np.array(y_dec_p, copy=False)
            Y_h = np.array(y_dec_h, copy=False)
            
#            print (type(X_p[0][0]))    # <type 'numpy.int32'>
#            print (type(X_h[0][0]))    # <type 'numpy.int32'>
#            print (type(Y[0]))         # <type 'numpy.int32'>
#            print (X_p.shape)   # (None, 44)
#            print (X_h.shape)   # (None, 44)
#            print (Y.shape)     # (None,)
#            print (Y_p.shape)   # (None, 44, 50004)
#            print (Y_h.shape)   # (None, 44, 50004)
            
            yield ([X_p, X_h], [Y, Y_p, Y_h])


def get_mask(seq):
    # seq: [Batch, sequence length] - [100, 44]
    batch, seq_len = seq.shape
    
    mask = np.zeros((batch, seq_len))
    for i, bat in enumerate(batch):
        for j, sl in enumerate(seq_len):
            if seq[i][j] != 0:
                mask[i][j] = 1
            else:
                break
    return mask                
                

def generate_batch_from_data(x_data, y_data, batch_size, voca_size, shuffle=False):
    while True:        
        sample_size, seq_len = x_data[0].shape
        
        # loop once per epoch
        if shuffle:
            indices = np.random.permutation(np.arange(sample_size))            
        else:
            indices = np.arange(sample_size)            
        
        num_batches = sample_size // batch_size
        if sample_size % batch_size != 0:
            num_batches += 1
        for bid in range(num_batches):
            # loop once per batch            
            batch_index = indices[bid * batch_size : min((bid + 1) * batch_size, sample_size)]
            batch_size_cur = len(batch_index)
            
            # p, h
            p = x_data[0][batch_index, :]
            h = x_data[1][batch_index, :]
            p = p.astype(np.int32)
            h = h.astype(np.int32)
            label = y_data[batch_index]            
            
            # exact match
            p_exact_match = x_data[2][batch_index, :]
            h_exact_match = x_data[3][batch_index, :]
            p_exact_match = p_exact_match.astype(np.float32)
            h_exact_match = h_exact_match.astype(np.float32)              
            
            sequences_p = np.zeros((batch_size_cur, seq_len, voca_size))
            sequences_h = np.zeros((batch_size_cur, seq_len, voca_size))
            for b in range(batch_size_cur):
                for s in range(1,seq_len):
                    #if s>0:
                    sequences_p[b, s-1, p[b,s]] = 1
                    sequences_h[b, s-1, h[b,s]] = 1
            
            yield ([p, h, p_exact_match, h_exact_match], [label, sequences_p, sequences_h])


def train(model, data, save_folder, args):    
    # {'entailment': 183416, 'neutral': 182764, '-': 785, 'contradiction': 183187}
    # train_data: (N, 44)  (N, 44)  (N, 3)
    (train_data, test_data) = data    
    
    x_train = train_data[:-1]
    x_test = test_data[:-1]
    y_train = train_data[-1]
    y_test  = test_data[-1]

    # callbacks
    log = callbacks.CSVLogger(save_folder + '/log.csv')
    monitor = 'val_loss'
    tb = callbacks.TensorBoard(log_dir=save_folder + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug), write_images=True)
    checkpoint = callbacks.ModelCheckpoint(save_folder + '/weights-{epoch:02d}.h5', monitor=monitor,
                                           save_best_only=False, save_weights_only=True, verbose=1, mode='auto')
#    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    reduce_lr = callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=1, verbose=1, 
                                            mode='auto') #, cooldown=2)
    
    train_gen = generate_batch_from_data(x_train, y_train, args.batch_size, args.voca_size, shuffle=True)
    val_gen   = generate_batch_from_data(x_test,  y_test,  args.batch_size, args.voca_size, shuffle=False)
    
    num_train_samples = x_train[0].shape[0]
    num_train_steps = num_train_samples // args.batch_size
    if num_train_samples % args.batch_size != 0:
        num_train_steps += 1
    num_test_samples = x_test[0].shape[0]
    num_val_steps = num_test_samples // args.batch_size
    if num_test_samples % args.batch_size != 0:
        num_val_steps += 1
    
    model.fit_generator(
        train_gen,
        steps_per_epoch=num_train_steps,
        validation_data=val_gen,
        validation_steps=num_val_steps,
        epochs=args.epochs,
        callbacks=[log, tb, checkpoint, reduce_lr],
        initial_epoch=args.initial_epoch
        )        
    
    model.save_weights(save_folder + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % save_folder)
   
    plot_log(save_folder + '/log.csv', show=True)


def test(model, test_data, args):
    x_test = test_data[:-1]
    y_test = test_data[-1]
    print ('test sampels: {}'.format(y_test.shape[0]))
    
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    y_pred_bin = np.array(y_pred.ravel() >= 0.5)
    
    if len(y_pred) == 3:
        y_pred_multiclass = np.argmax(y_pred[1],1)
        y_test_multiclass = np.argmax(y_test,1)
        
    if args.test_mode == 'pred':
        print('-'*30 + 'Begin: test' + '-'*30)
        if len(y_pred) == 3:
            [precision, recall, F1, support] = \
                precision_recall_fscore_support(y_test_multiclass, y_pred_multiclass)
            print ('Precision: {}  Recall: {}  F1: {}'.format(precision, recall, F1))
            print (confusion_matrix(y_test_multiclass, y_pred_multiclass))
            acc = accuracy_score(y_test_multiclass, y_pred_multiclass)
            print ('ACC(multi-class): {:.4f}'.format(acc))
        
        acc = accuracy_score(y_test, y_pred_bin)
        print ('ACC(binary-class): {:.4f}'.format(acc))        
        print('-' * 30 + 'End: test' + '-' * 30)


def test_twosents(model, voca_name, args):
    word2id_path = os.path.join(args.load_dir, 'word2id_'+voca_name+'_unified.pkl')
    wordvec_path = os.path.join(args.load_dir, 'word-vectors_'+voca_name+'_unified.npy')
    
    snli_preprocessor = SNLIPreprocessor()
    testsents = preprocess_2sent(args.testsents, p=train_data[0].shape[-1], h=train_data[1].shape[-1],
                              preprocessor=snli_preprocessor, need_exact_match=True,
                              word_vector_save_path=wordvec_path, word2id_save_path=word2id_path)
        
    y_pred = model.predict(testsents, batch_size=args.batch_size)            
    print ('Similarity score of two sentences is {}.'.format(y_pred))
    
        
def test_STS(model, voca_name, args):
    sts_dev, sts_test = download_and_load_sts_data()
    
    for sts_data, split in [[sts_dev, 'dev'], [sts_test, 'test']]:
        text_a = sts_data['sent_1'].tolist()
        text_b = sts_data['sent_2'].tolist()
        dev_scores = sts_data['sim'].tolist()
        genre = sts_data['genre'].tolist()
        
        genre_name = ['main-captions', 'main-forums', 'main-news', 'all']
        ids_captions = [i for i, g in enumerate(genre) if g==genre_name[0]]
        ids_forums   = [i for i, g in enumerate(genre) if g==genre_name[1]]
        ids_news     = [i for i, g in enumerate(genre) if g==genre_name[2]]
        ids_all      = range(len(genre))
        
        word2id_path = os.path.join(args.load_dir, 'word2id_'+voca_name+'_unified.pkl')
        wordvec_path = os.path.join(args.load_dir, 'word-vectors_'+voca_name+'_unified.npy')
        
        print('-' * 30 + 'BEGIN: STS ' +split+ '-' * 30)        
        for i, ids in enumerate([ids_captions, ids_forums, ids_news, ids_all]):
            ta = [text_a[id] for id in ids]
            tb = [text_b[id] for id in ids]
            
            snli_preprocessor = SNLIPreprocessor()
            parsed = preprocess_2sent([ta, tb], p=args.max_len, h=args.max_len, 
                                      preprocessor=snli_preprocessor, need_exact_match=True,
                                      word_vector_save_path=wordvec_path, word2id_save_path=word2id_path)
            
            y_pred = model.predict(parsed, batch_size=args.batch_size)
            y_pred = [item for sublist in y_pred for item in sublist]
            
            scores = [dev_scores[id] for id in ids]
            pearson_correlation = scipy.stats.pearsonr(y_pred, scores)
            print('\t', genre_name[i])
            print('Pearson correlation coefficient = {0}\tp-value = {1}'.format(pearson_correlation[0], 
                  pearson_correlation[1]))
        print('-' * 30 + 'End: STS ' +split+ '-' * 30)


def test_DUCTAC(model, data_path, voca_name, re_exp, args):
#    data_list = [0,1]
    data_path = data_path + '_' + voca_name
    data_list = [0,1,2,3]
    
    folders = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,d)) and re.search(re_exp, d)]
    print('{} folders are found from {}'.format(len(folders), data_path))
    
    time_takes = []
    for i, folder in enumerate(folders):
        x_test  = ChunkDataManager(load_data_path=folder).load(load_list=data_list)
        
        start_test = time.time()
        y_pred = model.predict(x_test, batch_size=args.batch_size)
        finish_test = time.time() - start_test
        
        base_name = os.path.basename(folder)
        time_takes.append(finish_test)
        print ('[{} {}] elapsed testing time: {:3.3f} secs'.format(i+1, base_name, finish_test))
        
        matfile_name = os.path.join(folder, 'prediction.mat')
        sio.savemat(matfile_name, {'pred_siamese':y_pred})
        
    print ('average elapsed testing time of 1 doc: {}'.format(np.mean(time_takes)))
    
    # copy prediction files
    dest_path = os.path.dirname(data_path)    
    aux_name = '_'+voca_name
    dest_path = os.path.join(dest_path, 'predictions', 'model' + aux_name)
    copy_predfile(data_path, dest_path, re_exp)


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="CapsNet Sentence Similarity")
    
    # Learning 
    parser.add_argument('--epochs',             default=10,             type=int,
                        help="Number of epochs to train")
    parser.add_argument('--batch_size',         default=50,             type=int,
                        help="Size of batch")
    parser.add_argument('--lr',                 default=0.001,          type=float, 
                        help="Initial learning rate")
    parser.add_argument('--lr_decay',           default=0.9,            type=float,
                        help="The value multiplied by lr at each epoch.")
        
    # Architecture
    parser.add_argument('--voca_size',          default=50003,          type=int, 
                        help="3 for <UNK>, <START>, <END>; 10003 for 10k")
    parser.add_argument('--voca_dim',           default=300,            type=int)
    parser.add_argument('--lstm_layer_num',     default=1,              type=int)
    parser.add_argument('--lstm_hidden_unit',   default=256,            type=int)
    parser.add_argument('--routings',           default=3,              type=int,
                        help="Number of iterations used in routing algorithm. should > 0")    
    parser.add_argument('--filters',            default=[3,4,5,6,7],  type=int,   
                        help='List of 1D Conv filters')
    parser.add_argument('--dr_rate',            default=0.2,            type=float, 
                        help="Dropout rate")    
    parser.add_argument('--filter_out',         default=100,            type=int,
                        help="Output size of 1D Conv filters")
    parser.add_argument('--capsule_num',        default=12,             type=int,
                        help="Output capsule number")
    parser.add_argument('--capsule_dim',        default=15,             type=int,
                        help="Output capsule dimension")
    
    # Data
    parser.add_argument('--data_list',          default=[0,1,2,3,4],    type=int,
                        help="Name list of splited numpy data")
    parser.add_argument('--load_dir',           default='data',         type=str,   
                        help="Directory of the data")
    parser.add_argument('--dataset_list',      default=['cnn_dm'],     type=str,   
                        help="List of dataset to train or test") # cnn_dm,STS
    parser.add_argument('--logdir',             default='logs',         type=str,   
                        help="Directory for Tensorboard logs")
    parser.add_argument('--save_dir',           default='./result',
                        help="directory to training outputs")
    parser.add_argument('--debug',              action='store_true',    
                        help="Save weights by TensorBoard")
    parser.add_argument('--max_len',            default=44,             type=int,
                        help="Max length of sentence")
    
    # Testing    
    parser.add_argument('--force_loadweight',   default=False, 
                        help="set to load a given weigths no matter what")
    parser.add_argument('--initial_epoch',      default=0,              type=int,
                        help='initial epoch for resuming training (0-index)')
    parser.add_argument('--testing',            default=True, 
                        help="Test the trained model on testing dataset; set to False when Training")
    parser.add_argument('--weights',            default='weights-06.h5',
                        help="Trained weights to be loaded. Should be specified when testing")
    parser.add_argument('--test_mode',          default='pred', # pred, sents, STS, DUC, TAC
                        help="Test mode: pred-prediction for a given dataset, sents-prediction of two sentence similarity, STS-prediction for STS dataset, DUC/TAC-similarity prediction for pair sentences in DUC/TAC dataset")
    parser.add_argument('--DUC_data_path',          default=['./data/2003/data', './data/2004/data'])
    parser.add_argument('--TAC_data_path',          default=['./data/TAC_Data/s080910_gen_proc/data', './data/TAC_Data/s11_gen_proc/data'])    
    parser.add_argument('--testsents',          default=[['Snowstorm slams east-ern US on Friday'], ['A strong wintry storm wasdumping snow in eastern US after creating traffichavoc that claimed at least eight lives']])
                        #default=[['i saw john was eating an apple'], ['i was eating an apple with john']]) 
        
    args = parser.parse_args()
        
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    folder_name = 'capsnet_sim'
    save_folder = os.path.join(args.save_dir, folder_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    params = vars(args)
    print(json.dumps(params, indent = 2))
    config_fn = os.path.join(save_folder, 'config.json')
    if not args.testing:
        with open(config_fn, 'w') as outfile:
            json.dump(params, outfile)      
    
    ''' Prepare data '''
    voca_name = 'voca'+str(args.voca_size/1000)+'k_'+str(args.voca_dim)+'d'
    dataset_list = args.dataset_list
    for dataset in dataset_list:
        if dataset.startswith('cnn'):
            train_cnndm = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'cnn_dm', 'train_'+voca_name)).load(load_list=args.data_list)
            test_cnndm  = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'cnn_dm', 'test_'+voca_name)).load(load_list=args.data_list)
            val_cnndm   = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'cnn_dm', 'val_'+voca_name)).load(load_list=args.data_list)
            
            if not args.testing:
                test_cnndm = val_cnndm            
            print('# cnn_dm samples: train:{}, test:{}'.format(train_cnndm[0].shape[0], test_cnndm[0].shape[0]))
        
        elif dataset.startswith('STS'):
            train_sts = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'stsbenchmark', 'train_'+voca_name)).load(load_list=args.data_list)            
            test_sts   = ChunkDataManager(load_data_path=os.path.join(args.load_dir, 'stsbenchmark', 'dev_'+voca_name)).load(load_list=args.data_list)
            print('# sts samples: train:{}, test:{}'.format(train_sts[0].shape[0], test_sts[0].shape[0]))            
    
    train_data = []
    test_data = []
    if len(dataset_list) > 1:
        train_tmp = []
        test_tmp = []
        for dataset in dataset_list:
            if dataset.startswith('cnn'):
                train_tmp.append(train_cnndm)
                test_tmp.append(test_cnndm)
            elif dataset.startswith('STS'):
                train_tmp.append(train_sts)
                test_tmp.append(test_sts)
        
        for i in range(len(args.data_list)):
            train_concat = []
            test_concat = []
            for d in range(len(dataset_list)):
                train_concat.append(train_tmp[d][i])
                test_concat.append(test_tmp[d][i])
            train_data.append(np.concatenate(train_concat, axis=0))
            test_data.append(np.concatenate(test_concat, axis=0))
        
    elif len(dataset_list) == 1:
        if dataset_list[0].startswith('cnn'):
            train_data = train_cnndm
            test_data  = test_cnndm
        elif dataset_list[0].startswith('STS'):
            train_data = train_sts
            test_data  = test_sts
    
    ''' Getting dimensions of the input '''
    for td in train_data:
        print (td.shape)
    
    ''' Load model '''
    word2vec_path = os.path.join(args.load_dir, 'word-vectors_'+voca_name+'_unified.npy')
    word_embedding_weights = np.load(word2vec_path) #args.word_vec_path)
    print ('word vectors loaded from [{}]'.format(word2vec_path))
    
    if not args.testing:
        draw_summary_network = True
    else:
        draw_summary_network = False        
    
    net = CapsNetTextSim(logger, p=train_data[0].shape[-1], h=train_data[1].shape[-1],
                      save_folder=save_folder, folder_name=folder_name,  
                      word_embedding_weights=word_embedding_weights,
                      filters=args.filters, n_filter_out=args.filter_out,
                      capsule_num=args.capsule_num, capsule_dim=args.capsule_dim, routings=args.routings,
                      lr=args.lr, dr_rate=args.dr_rate,
                      voca_size=args.voca_size, lstm_layer_num=args.lstm_layer_num, lstm_hidden_unit=args.lstm_hidden_unit,
                      draw_summary_network=draw_summary_network)
    net()   # call __call__    
    model = net.model    
    pred_model = net.pred_model
    plot_model(model, to_file='{}/{}.pdf'.format(save_folder, folder_name), show_shapes=True)    
    
    # train or test
    if args.weights is not None:
        if args.force_loadweight or (not args.testing and args.initial_epoch>0) or (args.testing):  # init the model weights with provided one
            model.load_weights('{}/{}'.format(save_folder, args.weights))
            print ('weigth is loaded... from {}/{}'.format(save_folder, args.weights))
        
    if not args.testing:
        start_train = time.time()
        train(model=model, data=(train_data, test_data), save_folder=save_folder, args=args)
        print ('elapsed training time: {:3.3f} hrs'.format((time.time()-start_train)/3600))
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Terminate...')
            exit
        if args.test_mode == 'pred':
            test(model=pred_model, test_data=test_data, args=args)
        elif args.test_mode == 'sents':            
            test_twosents(model=pred_model, voca_name=voca_name, args=args)
        elif args.test_mode == 'DUC':
            for data_path in args.DUC_data_path:
                test_DUCTAC(model=pred_model, data_path=data_path, voca_name=voca_name, re_exp='^d[0-9]+', args=args)
        elif args.test_mode == 'TAC':
            for data_path in args.TAC_data_path:
                test_DUCTAC(model=pred_model, data_path=data_path, voca_name=voca_name, re_exp='^D[0-9]+\-[A-B]', args=args)
        elif args.test_mode == 'STS':
            test_STS(model=pred_model, voca_name=voca_name, args=args)
