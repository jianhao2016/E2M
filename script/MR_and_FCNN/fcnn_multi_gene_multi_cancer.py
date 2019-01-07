#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script try to infer from expression data to methlaytion data.
input:

p2data_root = '/data/GBM/data/'
with open(p2data_root + 'dataset/gbm_all', 'r') as f:
    DS = json.load(f)
    for key1 in DS.keys():
        for key2 in DS[key1].keys():
            DS[key1][key2] = np.array(DS[key1][key2], dtype = np.float32)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
import os
import shutil
import json
from collections import Counter

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def get_num_CpG_sites(gtype):
    if gtype == 'mgmt':
        num_sites = 7
    elif gtype == 'mlh1' or gtype == 'kras':
        num_sites = 6
    elif gtype == 'atm':
        num_sites = 4
    elif gtype == 'casp8' or gtype == 'tp53':
        num_sites = 3
    elif gtype == 'gata6':
        num_sites = 2
    else:
        print('not valid gene type {}'.format(gtype))
        num_sites = 0
    return num_sites

def convert_array_int(arr):
    val = 0
    for idx in range(len(arr)):
        val += 2**idx * arr[idx]
    return int(val)
    
def longest_burst(arr):
    max_burst = 0
    rl_count = 0
    for val in arr:
        if val == 1:
            rl_count += 1
        elif val == 0:
            rl_count = 0
        max_burst = max(max_burst, rl_count)

    return max_burst

def hamming_distance_of_int_numbers(num1, num2):
    bin_1 = '{0:07b}'.format(num1)
    bin_2 = '{0:07b}'.format(num2)
    hamming_d = 0
    for b1, b2 in zip(bin_1, bin_2):
        if b1 != b2:
            hamming_d += 1
    return hamming_d

def hamming_error(pred_labels, true_labels):
    hamming_error = np.zeros(8)
    for pred, true in zip(pred_labels, true_labels):
        hamming_d = hamming_distance_of_int_numbers(int(pred), int(true))
        hamming_error[hamming_d] += 1
    hamming_error = hamming_error / sum(hamming_error)
    return hamming_error

def dnn_model_fn(features, labels, mode):
    """
    model function for nerual network.
    used in tf.estimator
    """
    print('\nMode = ', mode)
    print('type of labels = ', type(labels))
    input_layer = tf.reshape(features['x'], [-1, 978])
    print(input_layer.shape.as_list())
    training_flag = (mode == tf.estimator.ModeKeys.TRAIN)


    # first layer.
    fc1 = tf.layers.dense(inputs = input_layer, 
                          units = 1000,
                          activation = tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs = fc1, 
                                 rate = 0.5,
                                 training = training_flag)

    # second layer.
    fc2 = tf.layers.dense(inputs = dropout1,
                          units = 500,
                          activation = tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs = fc2,
                                 rate = 0.5,
                                 training = training_flag)

    # thrid layer.
    fc3 = tf.layers.dense(inputs = dropout2,
                          units = 250,
                          activation = tf.nn.relu)
    dropout3 = tf.layers.dropout(inputs = fc3,
                                 rate = 0.5,
                                 training = training_flag)

    # logit layer.
    logits = tf.layers.dense(inputs = dropout3,
                           units = N_CLASSES)

    print('logits shape = ', logits.get_shape().as_list())
    # general predictions for PREDICT and EVAL mode.
    # predictions = { "classes": tf.argmax(logits, axis = 1),
    #                 "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")}

    output_classes = tf.argmax(logits, axis = 1, name = 'nn_output_classes')
    predictions = { "classes": output_classes}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions = predictions)

    # calculate loss for TRAIN and EVAL
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), 
            depth = N_CLASSES)

    cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits = logits)

    tf.identity(cross_entropy, name = 'cross_entropy_tensor')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    # configure the training Op.
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        starter_learning_rate = init_learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                global_step, decay_steps = 1000, decay_rate = 0.96, 
                staircase = True, name = 'expo_decay')

        tf.identity(learning_rate, name = 'learning_rate_tensor')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(
                loss = loss,
                global_step = global_step)
        # global_step in the train_op is an optional Variable.
        # which will automatically increment one after the train_op update.
        # train_op = optimizer.minimize(
        #         loss = loss,
        #         global_step = tf.train.get_global_step())
        # return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op)
    else:
        train_op = None

    # the code below will only be called when 
    # mode = tf.estimator.ModeKeys.EVAL
    # eval_metric_op = {"accuracy": acc}
    # eval_metric_op = {
    #         "accuracy": tf.metrics.accuracy(
    #             labels = labels, predictions = predictions["classes"])}
    accuracy = tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
    eval_metric_ops = {'accuracy': accuracy}
    
    # correct_prediction = tf.equal(tf.argmax(predictions["probabilities"],1), tf.argmax(labels,1))
    correct_prediction = tf.equal(predictions["classes"], tf.cast(labels, tf.int64))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy_tensor")
    tf.identity(acc, name = 'train_accuracy_tensor')
    tf.summary.scalar('train_accuracy', acc)

    tf.identity(accuracy[1], name='train_accuracy_running_tensor')
    tf.summary.scalar('train_accuracy_running_average', accuracy[1])

    # eval_metric_op = {
    #         "accuracy": tf.metrics.accuracy(
    #             labels = labels, predictions = predictions["classes"])}
    # return tf.estimator.EstimatorSpec(
    #         mode = mode, loss = loss, eval_metric_ops = eval_metric_op)
    return tf.estimator.EstimatorSpec(
            mode = mode, 
            predictions = predictions,
            loss = loss,
            train_op = train_op,
            eval_metric_ops = eval_metric_ops)


def main(unused_argv):
    # load data.
    
    with open(p2data_root + file_name, 'r') as f:
        DS = json.load(f)
        for key1 in DS.keys():
            for key2 in DS[key1].keys():
                DS[key1][key2] = np.array(DS[key1][key2], dtype = np.float32)
    
    labels_dict = {'train':[], 'test':[]}
    for key1 in DS.keys():
        for label_idx in range(len(DS[key1]['label'])):
            bin_label = DS[key1]['label'][label_idx]
            if error_type == 'word_error':
                labels_dict[key1].append(convert_array_int(bin_label))
            elif error_type == 'count_error':
                labels_dict[key1].append(sum(bin_label))
            elif error_type == 'runlen_error':
                labels_dict[key1].append(longest_burst(bin_label))
            elif error_type == 'threshold_error':
                tmp_label = sum(bin_label)
                if tmp_label >= meth_threshold:
                    thres_label = 1.0
                else:
                    thres_label = 0.0
                labels_dict[key1].append(thres_label)

    train_data = DS['train']['data']
    train_labels = np.array(labels_dict['train'], np.float32)

    test_data = DS['test']['data']
    test_labels = np.array(labels_dict['test'], np.float32)

    # create the estimator.
    runConfig = tf.contrib.learn.RunConfig(session_config = config)
    runConfig.replace(save_checkpoints_secs=1e9)
    gene_classifier = tf.estimator.Estimator(
            model_fn = dnn_model_fn,
            model_dir = model_saved_path,
            config = runConfig)

    tensors_to_log = {'learning_rate': 'learning_rate_tensor',
                      # 'cross_entropy': 'cross_entropy_tensor',
                      'batch_accuracy': 'train_accuracy_tensor',
                      'running_acc': 'train_accuracy_running_tensor'}
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # tensors_to_log = {"acc": "accuracy_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=batch_size,
            num_epochs=epochs,
            shuffle=True)
    gene_classifier.train(
            input_fn=train_input_fn,
            hooks=[logging_hook])

    # evaluate the model and print the results.
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            y=test_labels,
            num_epochs=1,
            shuffle=False)
    eval_result = gene_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_result)

    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_data},
            y=None,
            num_epochs=1,
            shuffle=False)
    
    predict_result = gene_classifier.predict(input_fn=pred_input_fn)
    pred = {'class':[]}
    for idx in predict_result:
        pred['class'].append(idx['classes'])
    print('pred set = ', Counter(pred['class']))

    # variable_names = [v for v in tf.trainable_variables()]
    # print(variable_names)
    # print(set(train_labels))
    print('test set =', Counter(test_labels))
    print('hamming error = \n', hamming_error(pred['class'], test_labels))
    # print('one bit error = ', sum(hamming_error(pred['class'], test_labels)[:2]))
    print(ctype, '\t', gtype, '\t', error_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p2root', type=str,
                        default='../data/', help='root path to data')
    parser.add_argument('--cancer_type', type=str,
            choices=['lgg', 'gbm', 'brain', 'luad', 'lusc', 'lung', 'stad'], 
            default='gbm',
            help='cancer type of training data. default brain, mixed of lgg & gbm')
    parser.add_argument('--gene', type = str,
            choices = ['mgmt', 'mlh1', 'atm', 'gata6', 'casp8', 'kras', 'tp53'], default = 'gata6',
            help='which gene to predict, default mgmt')
    parser.add_argument('--error_type', type=str,
            choices=['word_error', 'count_error', 'runlen_error', 'threshold_error'], 
            default='word_error', help='error metric to use')

    parser.add_argument('--batch', type=int, default=64,
            help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
            help='learning_rate')
    parser.add_argument('--epoch', type=int, default=500,
            help='number of epoches')
    parser.add_argument('--model', type=str, default='saved_model_tmp',
            help='name of saved models')
    parser.add_argument('--quantized', type=int,
                        choices=[0, 4], default=0,
                        help='1 for quantized input, 0 for not. Default 1')

    FLAG, _ = parser.parse_known_args()


    p2data_root = FLAG.p2root
    gtype = FLAG.gene
    ctype = FLAG.cancer_type
    if ctype == 'lgg':
        p2data_root += '{}/data/dataset/'.format(ctype.upper())
        file_name = 'lgg_all_' + gtype
    elif ctype == 'gbm':
        p2data_root += '{}/data/dataset/'.format(ctype.upper())
        file_name = 'gbm_all_' + gtype
    elif ctype == 'brain':
        p2data_root += 'merge_GBM_LGG/'
        file_name = 'brain_all_' + gtype

    elif ctype == 'luad':
        p2data_root += '{}/data/dataset/'.format(ctype.upper())
        file_name = 'luad_all_' + gtype
    elif ctype == 'lusc':
        p2data_root += '{}/data/dataset/'.format(ctype.upper())
        file_name = 'lusc_all_' + gtype
    elif ctype == 'lung':
        p2data_root += 'merge_LUAD_LUSC/'
        file_name = 'lung_all_' + gtype

    elif ctype == 'stad':
        p2data_root += '{}/data/dataset/'.format(ctype.upper())
        file_name = 'stad_all_' + gtype

    error_type = FLAG.error_type
    num_CpG_sites = get_num_CpG_sites(gtype)
    if error_type == 'word_error':
        # if gtype == 'mgmt' or gtype == 'mlh1':
        #     N_CLASSES = 128
        # elif gtype == 'atm':
        #     N_CLASSES = 16
        # elif gtype == 'gata6':
        #     N_CLASSES = 2 ** 2
        N_CLASSES = 2 ** num_CpG_sites
    elif error_type == 'count_error' or error_type == 'runlen_error':
        # if gtype == 'mgmt' or gtype == 'mlh1':
        #     N_CLASSES = 8
        # elif gtype == 'atm':
        #     N_CLASSES = 5
        # elif gtype == 'gata6':
        #     N_CLASSES = 2 + 1
        N_CLASSES = num_CpG_sites + 1
    elif error_type == 'threshold_error':
        N_CLASSES = 2
        # if gtype == 'mgmt' or gtype == 'mlh1':
        #     meth_threshold = 4
        # elif gtype == 'atm':
        #     meth_threshold = 2
        # elif gtype == 'gata6':
        #     meth_threshold = 1
        meth_threshold = num_CpG_sites // 2
    
    _WEIGHT_DECAY = 2e-4
    batch_size = FLAG.batch
    epochs = FLAG.epoch
    init_learning_rate = FLAG.lr

    model_name = '{}_{}_{}_fcnn'.format(ctype, gtype, error_type)
    model_saved_path = p2data_root + 'saved_models_fcnn/' + model_name

    if FLAG.quantized == 4:
        file_name += '_quantized_4bits'
        model_name += '_Q_4bits'
        model_saved_path = p2data_root + 'saved_models_fcnn/' + model_name

    if os.path.isdir(model_saved_path):
        shutil.rmtree(model_saved_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

    print(FLAG)
