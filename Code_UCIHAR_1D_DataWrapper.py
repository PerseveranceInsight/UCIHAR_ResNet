import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

class UCIHARDataset(Dataset):
    '''
        Dataset for loading and preprocessing the UCI-HAR dataset
    '''
    def __init__(self,
                 signal_path_prefix,
                 signal_path_suffix,
                 signal_labelfile,
                 mode='train'):
        def read_signals(filename):
            with open(filename, 'r') as fp:
                signal = fp.read().splitlines()
            signal = map(lambda x: x.rstrip().lstrip().split(), signal)
            signal = [list(map(float, line)) for line in signal]
            signal = np.array(signal, dtype=np.float32)
            return signal

        def read_signals_label(filename):
            with open(filename, 'r') as fp:
                actions = fp.read().splitlines()
                actions = list(map(int, actions))
            return np.array(actions)

        def signals_shuffle(signals, labels, shuffle=True):
            shuffled_signals = signals
            shuffled_labels = labels
            if shuffle:
                shuffle_ind = np.random.permutation(labels.shape[0])
                shuffled_signals = shuffled_signals[shuffle_ind, :, :]
                shuffled_labels = shuffled_labels[shuffle_ind]
            return shuffled_signals, shuffled_labels
        
        self.mode = mode
        self.np_signals = []
        for file_name in signal_path_suffix:
            signal_sample = read_signals(signal_path_prefix + file_name)
            self.np_signals.append(signal_sample)
        
        self.np_signals = np.transpose(np.array(self.np_signals), (1, 2, 0))
        self.labels = read_signals_label(signal_labelfile)
        self.labels = self.labels - 1

        num_samples = self.np_signals.shape[0]

        if mode == 'test':
            self.torch_signal = torch.FloatTensor(self.np_signals)
            self.torch_labels = torch.FloatTensor(self.labels)
        else:
            self.np_signals, self.labels = signals_shuffle(self.np_signals,
                                                           self.labels,
                                                           shuffle=True)
            if mode == 'train':
                indices = [i for i in range(num_samples) if i % 10 != 0]
            elif mode == 'val':
                indices = [i for i in range(num_samples) if i % 10 == 0]

            self.np_signals = self.np_signals[indices,:,:]
            self.labels = self.labels[indices]
            self.torch_signal = torch.FloatTensor(self.np_signals)
            self.torch_labels = torch.LongTensor(self.labels)
            print('Shape of np_signals {0}'.format(self.np_signals.shape))
            print('Shape of labels {0}'.format(self.labels.shape))

    def __getitem__(self, index):
        # Returns one sample at a time
        return self.torch_signal[index, :, :], self.torch_labels[index]

    def __len__(self):
        return len(self.labels)
