# this script runs multi-class regression on the dataset.

import tensorflow as tf
import numpy as np
import json
import argparse

from fcnn_multi_gene_multi_cancer import get_num_CpG_sites

def convert_array_int(arr):
    val = 0
    for idx in range(len(arr)):
        val += 2**idx * arr[idx]
    return int(val)

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


def logistic_regressor(feature, labels):
    logits = tf.layers.dense(feature, N_CLASSES, kernel_initializer=tf.zeros_initializer)
    predicted_label = tf.argmax(logits, axis = 1)

    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32),
                               depth = N_CLASSES)
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    reg_loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
            # [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            [tf.norm(v, ord=1) for v in tf.trainable_variables()])

    correct_prediction = tf.equal(predicted_label, tf.cast(labels, tf.int64))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy_tensor")
    return acc, reg_loss, predicted_label

def main(unused_argv):

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
            # elif error_type == 'runlen_error':
            #     labels_dict[key1].append(longest_burst(bin_label))
            elif error_type == 'threshold_error':
                tmp_label = sum(bin_label)
                if tmp_label >= meth_threshold:
                    thres_label = 1.0
                else:
                    thres_label = 0.0
                labels_dict[key1].append(thres_label)

    train_data = DS['train']['data']
    train_labels = np.array(labels_dict['train'], np.float32)
    print('train labels:', train_labels[:10])

    test_data = DS['test']['data']
    test_labels = np.array(labels_dict['test'], np.float32)
    print('test_labels:', test_labels[:10])

    feature_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 978])
    label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None])

    train_dict = {feature_placeholder: train_data, label_placeholder: train_labels}
    test_dict = {feature_placeholder: test_data, label_placeholder: test_labels}

    acc_t, loss_t, predicted_t = logistic_regressor(feature_placeholder, label_placeholder)

    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_t)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print_counter = 0
        for iteration in range(2000):
            print_counter += 1
            _ = sess.run(train_step, feed_dict = train_dict)
            if print_counter == 100:
                print_counter = 0
                train_accuracy, train_loss = sess.run([acc_t, loss_t],
                                                        feed_dict = train_dict)
                print('iteration {}, train accuracy = {:.6f}, train_loss = {:.4f}'.format(
                        iteration, train_accuracy, train_loss))

        test_accuracy, test_loss, test_predicted = sess.run([acc_t, loss_t, predicted_t],
                                                feed_dict = test_dict)
        print('Test result: accuracy = {:.6f}, loss = {:.4f}'.format(test_accuracy, test_loss))
        print('Test output:', test_predicted)
        print('test labels:', test_labels.astype(np.int))
        # print('one bit error = {:.6f}'.format(sum(hamming_error(test_predicted, test_labels)[:2])))
        # print('Hamming distance: ', hamming_error(test_predicted, test_labels))
        print(FLAG)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cancer_type', type=str,
                        choices=['lgg', 'gbm', 'brain', 'luad', 'lusc', 'lung', 'stad'],
                        default='gbm', help='cancer type, default lgg')
    parser.add_argument('--gene', type=str,
                        choices=['mgmt', 'mlh1', 'atm', 'gata6', 'kras', 'casp8', 'tp53'],
                        default='gata6', help='gene type, default mgmt')
    parser.add_argument('--p2root', type=str,
                        default='../data/', help='root path to data')

    parser.add_argument('--error_type', type=str,
                        choices=['word_error', 'count_error', 'runlen_error', 'threshold_error'],
                        default='word_error', help='error metric to use, default word_error')
    FLAG, _ = parser.parse_known_args()

    p2data_root = FLAG.p2root
    ctype = FLAG.cancer_type
    gtype = FLAG.gene

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

    # file_name += '_quantized'

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
    print(FLAG)

    tf.app.run()
