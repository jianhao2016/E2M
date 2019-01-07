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

p2data_root = '../data/GBM/data/'
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
import json
import os
import shutil
from collections import Counter
# from check_meth_status import convert_array_int

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
    hamming_err = np.zeros(8)
    for pred, true in zip(pred_labels, true_labels):
        hamming_d = hamming_distance_of_int_numbers(int(pred), int(true))
        hamming_err[hamming_d] += 1
    hamming_err = hamming_err / sum(hamming_err)
    return hamming_err

def dnn_model_fn(features, labels, mode):
    """
    model function for nerual network.
    used in tf.estimator
    """
    print('\nMode = ', mode)
    # print('type of labels = ', type(labels))
    # label type is a tensor!!
    input_layer = tf.reshape(features['x'], [-1, 978, 1])
    # print(input_layer.shape.as_list())
    training_flag = (mode == tf.estimator.ModeKeys.TRAIN)

    simple_conv1 = tf.layers.conv1d(inputs = input_layer,
            filters = 32,
            kernel_size = 7,
            strides = 1,
            padding = 'same',
            activation = tf.nn.relu)
    simple_max_pool1 = tf.layers.max_pooling1d(inputs = simple_conv1,
            pool_size = 3,
            strides = 2,
            padding = 'same')
    simple_conv2 = tf.layers.conv1d(inputs = simple_max_pool1,
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            activation = tf.nn.relu)
    simple_max_pool2 = tf.layers.max_pooling1d(inputs = simple_conv2,
            pool_size = 3,
            strides = 2,
            padding = 'same')

    # first inception layer.
    conv1x_layer1 = tf.layers.conv1d(inputs = simple_max_pool2,
            filters = 32,
            kernel_size = 1,
            strides = 1,
            padding = 'same',
            activation = tf.nn.relu)

    conv3x_layer1 = tf.layers.conv1d(inputs = simple_max_pool2,
            filters = 8,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            activation = tf.nn.relu)

    conv5x_layer1 = tf.layers.conv1d(inputs = simple_max_pool2,
            filters = 16,
            kernel_size = 5,
            padding = 'same',
            activation = tf.nn.relu)

    max_pool_ahead_layer1 = tf.layers.max_pooling1d(inputs = simple_max_pool2,
            pool_size = 3,
            strides = 1,
            padding = 'same')

    conv7x_layer1 = max_pool_ahead_layer1
    # conv7x_layer1 = tf.layers.conv1d(inputs = max_pool_ahead_layer1,
    #         filters = 16,
    #         kernel_size = 7,
    #         padding = 'same',
    #         activation = tf.nn.relu)
    layer1_concat = tf.concat([conv1x_layer1, conv3x_layer1, conv5x_layer1, 
                                conv7x_layer1], axis = 2)

    max_pool_after_layer1 = tf.layers.max_pooling1d(inputs = layer1_concat,
            pool_size = 5,
            strides = 3)

    # second inception layer.
    conv1x_layer2 = tf.layers.conv1d(inputs = max_pool_after_layer1,
            filters = 16,
            kernel_size = 1,
            strides = 1,
            padding = 'same',
            activation = tf.nn.relu)

    conv3x_layer2 = tf.layers.conv1d(inputs = max_pool_after_layer1,
            filters = 32,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            activation = tf.nn.relu)

    conv5x_layer2 = tf.layers.conv1d(inputs = max_pool_after_layer1,
            filters = 64,
            kernel_size = 5,
            padding = 'same',
            activation = tf.nn.relu)

    max_pool_ahead_layer2 = tf.layers.max_pooling1d(inputs = max_pool_after_layer1,
            pool_size = 3,
            strides = 1,
            padding = 'same')

    conv7x_layer2 = max_pool_ahead_layer2
    # conv7x_layer2 = tf.layers.conv1d(inputs = max_pool_ahead_layer2,
    #         filters = 8,
    #         kernel_size = 7,
    #         padding = 'same',
    #         activation = tf.nn.relu)
    layer2_concat = tf.concat([conv1x_layer2, conv3x_layer2, conv5x_layer2, 
                                conv7x_layer2], axis = 2)

    max_pool_after_layer2 = tf.layers.max_pooling1d(inputs = layer2_concat,
            pool_size = 5,
            strides = 3)

    # flatten inception output
    inception_output = tf.contrib.layers.flatten(inputs = max_pool_after_layer2)

    # first fc layer.
    fc1 = tf.layers.dense(inputs = inception_output, 
                          units = 128,
                          activation = tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs = fc1, 
                                 rate = 0.5,
                                 training = training_flag)

    # logit layer.
    logits = tf.layers.dense(inputs = dropout1,
                           units = N_CLASSES)

    # print('logits shape = ', logits.get_shape().as_list())
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
                global_step, decay_steps = 500, decay_rate = 0.96, 
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

    print('hamming error = ', hamming_error(pred['class'], test_labels))
    print('word error = {:.4f}'.format(hamming_error(pred['class'], test_labels)[0]))
    print('one bit error = {:.4f}'.format(sum(hamming_error(pred['class'], test_labels)[:2])))
    print('{} {} {}'.format(ctype, gtype, FLAG.quantization))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p2root', type=str,
            default='../data/', help='root path to data')
    parser.add_argument('--cancer_type', type=str,
            choices=['lgg', 'gbm', 'brain', 'luad', 'lusc', 'lung', 'stad'], 
            default='brain',
            help='cancer type of training data. default brain, mixed of lgg & gbm')
    parser.add_argument('--gene', type = str,
            choices = ['mgmt', 'mlh1', 'atm', 'gata6', 'casp8', 'kras', 'tp53'], default = 'mgmt',
            help='which gene to predict, default mgmt')
    parser.add_argument('--error_type', type=str,
            choices=['word_error', 'count_error', 'runlen_error', 'threshold_error'], 
            default='word_error', help='error metric to use')

    parser.add_argument('--batch', type=int, default=64,
            help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
            help='learning_rate')
    parser.add_argument('--epoch', type=int, default=4000,
            help='number of epoches')
    parser.add_argument('--model', type=str, default='',
            help='name of saved models')
    parser.add_argument('--quantization', type=int,
            choices = [0, 4, 8], default = 8, help='4/8 for quantized bits, 0 for not, default 8')

    FLAG, _ = parser.parse_known_args()


    p2data_root = FLAG.p2root
    gtype = FLAG.gene
    ctype = FLAG.cancer_type
    if ctype == 'lgg':
        file_name = 'LGG/data/dataset/lgg_all_' + gtype
    elif ctype == 'gbm':
        file_name = 'GBM/data/dataset/gbm_all_' + gtype
    elif ctype == 'brain':
        file_name = 'merge_GBM_LGG/brain_all_' + gtype
    elif ctype == 'luad':
        file_name = 'LUAD/data/dataset/luad_all_' + gtype
    elif ctype == 'lusc':
        file_name = 'LUSC/data/dataset/lusc_all_' + gtype
    elif ctype == 'lung':
        file_name = 'merge_LUAD_LUSC/lung_all_' + gtype
    elif ctype == 'stad':
        file_name = 'STAD/data/dataset/stad_all_' + gtype


    error_type = FLAG.error_type
    num_CpG_sites = get_num_CpG_sites(gtype)
    if error_type == 'word_error':
        # if gtype == 'mgmt' or gtype == 'mlh1':
        #     N_CLASSES = 128
        # elif gtype == 'atm':
        #     N_CLASSES = 16
        # elif gtype == 'gata6':
        #     N_CLASSES = 4
        N_CLASSES = 2 ** num_CpG_sites
    elif error_type == 'count_error' or error_type == 'runlen_error':
        # if gtype == 'mgmt' or gtype == 'mlh1':
        #     N_CLASSES = 8
        # elif gtype == 'atm':
        #     N_CLASSES = 5
        # elif gtype == 'gata6':
        #     N_CLASSES = 3
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

    if FLAG.model == '':
        model_name = '{}_{}_{}_inception'.format(ctype, gtype, error_type)
        model_saved_path = FLAG.p2root + 'modified_data_retraining/' + model_name
    else:
        model_saved_path = FLAG.p2root + 'modified_data_retraining/' + FLAG.model

    if FLAG.quantization != 0:
        file_name = '{}_quantized_{}bits'.format(file_name, FLAG.quantization)
        model_saved_path = '{}_Q_{}bits'.format(model_saved_path, FLAG.quantization)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

    # print(FLAG)
