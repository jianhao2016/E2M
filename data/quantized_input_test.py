import json
import numpy as np
import matplotlib.pyplot as plt

p2f_root = './'
cancer_list = ['lgg', 'gbm', 'brain', 'luad', 'lusc', 'lung', 'stad']
# cancer_list = ['lgg', 'gbm', 'brain']
gene_list = ['mgmt', 'mlh1', 'atm', 'gata6', 'casp8', 'kras', 'tp53']
quantization_bit = 4
for ctype in cancer_list:
    for gtype in gene_list:
        p2f = p2f_root + ctype.upper() + '/data/dataset/' + ctype + '_all_' + gtype

        with open(p2f, 'r') as f:
            DS = json.load(f)
            for key1 in DS.keys():
                for key2 in ['data']:
                    DS[key1][key2] = np.array(DS[key1][key2], dtype = np.float32)

        print('train, 95 percentile expression of {} {} is {}'.format(ctype, gtype, 
            np.percentile(DS['train']['data'], 95)))
        print('test, 95 percentile expression of {} {} is {}'.format(ctype, gtype, 
            np.percentile(DS['test']['data'], 95)))
        print('-'*7)

        num_bins = 2 ** quantization_bit
        quantized_range = np.arange(num_bins)
        normal_factor = num_bins / 64
        # quantizer = np.arange(16)/0.25
        quantizer = quantized_range / normal_factor

        for key in DS.keys():
            DS[key]['data'] = np.digitize(DS[key]['data'], bins = quantizer)
            DS[key]['data'] = DS[key]['data'].tolist()

        quantized_suffix = '_quantized_{}bits'.format(quantization_bit)
        # p2f_quantized = p2f + '_quantized_4bits'
        p2f_quantized = p2f + quantized_suffix
        with open(p2f_quantized, 'w') as f:
            json.dump(DS, f)

        # for idx in range(3):
        #     idx1_train = DS['train']['data'][:, idx]
        #     idx1_test = DS['test']['data'][:, idx]

        #     plt.hist(idx1_train, bins = 'auto')
        #     plt.title('histogram of idx ' + str(idx))
        #     plt.savefig('pics/' + ctype + '/pos' + str(idx) + '_train_quantized')
        #     # plt.show()
        #     plt.hist(idx1_test, bins = 'auto')
        #     plt.savefig('pics/' + ctype + '/pos' + str(idx) + '_test_quantized')
        #     plt.close()
        #     # plt.show()
        # break
    # break

