import torch
import torch.nn as nn
import torch.nn.functional as nn_func

class UCIHAR_1D_model(nn.Module):
    def __init__(self, filter_parameters, pool_parameters):
        super(UCIHAR_1D_model, self).__init__()
        filter_input_channel = filter_parameters['input_channel']
        filter_output_channel = filter_parameters['output_channel']
        filter_size = filter_parameters['filter_size']
        filter_stride = filter_parameters['stride']
        filter_padding = filter_parameters['padding']
        filter_dilation = filter_parameters['dilation']
        filter_groups = filter_parameters['group']
        filter_bias = filter_parameters['bias']
        filter_padding_mode = filter_parameters['padding_mode']
        filter_num_label = filter_parameters['num_label']

        pool_size = pool_parameters['pool_size']
        pool_stride = pool_parameters['stride']
        pool_padding = pool_parameters['padding']
        pool_dilation = pool_parameters['dilation']
        pool_ceil = pool_parameters['ceil_mode']
        
        self.layer1 = nn.Conv1d(in_channels = filter_input_channel,
                                out_channels = filter_output_channel,
                                kernel_size = filter_size,
                                stride = filter_stride,
                                padding = filter_padding,
                                dilation = filter_dilation,
                                groups = filter_groups,
                                bias = filter_bias,
                                padding_mode = filter_padding_mode)
        self.max_pool1 = nn.MaxPool1d(kernel_size = pool_size,
                                      stride = pool_stride,
                                      padding = pool_padding,
                                      dilation = pool_dilation,
                                      return_indices = False,
                                      ceil_mode = pool_ceil)
        self.layer2 = nn.Conv1d(in_channels = filter_output_channel,
                                out_channels = filter_output_channel,
                                kernel_size = filter_size,
                                stride = filter_stride,
                                padding = filter_padding,
                                dilation = filter_dilation,
                                groups = filter_groups,
                                bias = filter_bias,
                                padding_mode = filter_padding_mode)
        self.max_pool2 = nn.MaxPool1d(kernel_size = pool_size,
                                      stride = pool_stride,
                                      padding = pool_padding,
                                      dilation = pool_dilation,
                                      return_indices = False,
                                      ceil_mode = pool_ceil)
        self.layer3 = nn.Conv1d(in_channels = filter_output_channel,
                                out_channels = filter_output_channel,
                                kernel_size = filter_size,
                                stride = filter_stride,
                                padding = filter_padding,
                                dilation = filter_dilation,
                                groups = filter_groups,
                                bias = filter_bias,
                                padding_mode = filter_padding_mode)
        self.max_pool3 = nn.MaxPool1d(kernel_size = pool_size,
                                      stride = pool_stride,
                                      padding = pool_padding,
                                      dilation = pool_dilation,
                                      return_indices = False,
                                      ceil_mode = pool_ceil)
        self.layer4 = nn.Conv1d(in_channels = filter_output_channel,
                                out_channels = filter_output_channel,
                                kernel_size = filter_size,
                                stride = filter_stride,
                                padding = filter_padding,
                                dilation = filter_dilation,
                                groups = filter_groups,
                                bias = filter_bias,
                                padding_mode = filter_padding_mode)
        self.max_pool4 = nn.MaxPool1d(kernel_size = pool_size,
                                      stride = pool_stride,
                                      padding = pool_padding,
                                      dilation = pool_dilation,
                                      return_indices = False,
                                      ceil_mode = pool_ceil)
        self.mlp = nn.Linear(in_features = 1040,
                             out_features = filter_num_label,
                             bias = True)
        self.softmax = nn.Softmax(dim=0)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input):
        y1 = nn_func.relu(self.layer1(input))
        h1 = self.max_pool1(y1)
        y2 = nn_func.relu(self.layer2(h1))
        h2 = self.max_pool2(y2)
        y3 = nn_func.relu(self.layer3(h2))
        h3 = self.max_pool3(y3)
        y4 = nn_func.relu(self.layer4(h3))
        h4 = self.max_pool4(y4)
        h4f = torch.flatten(h4,start_dim=1)
        h5 = self.mlp(h4f)
        return h5

    def cal_loss(self, pred_pro, tar):
        return self.criterion(pred_pro,tar) 


if __name__ == '__main__':
    import numpy as np
    test_input = torch.randn(1, 9, 128)
    test_output = np.array([5])
    test_target = torch.LongTensor(test_output)
    print('Shape of test_input : {0}'.format(test_input.shape))
    print('Content of target : {0}'.format(test_target))
    print('Shape of target : {0}'.format(test_target.shape))
    filter_parameters = {'input_channel': 9,
                         'output_channel': 10,
                         'filter_size': 5,
                         'stride': 1,
                         'padding': 0,
                         'dilation': 1,
                         'group': 1,
                         'bias': True,
                         'padding_mode': 'zeros',
                         'num_label': 6}
    pool_parameters = {'pool_size': 3,
                       'stride': 1,
                       'padding': 0,
                       'dilation': 1,
                       'ceil_mode': False}
    test_model = UCIHAR_1D_model(filter_parameters = filter_parameters,
                                 pool_parameters = pool_parameters)
    prob = test_model.forward(input = test_input)
    loss = test_model.cal_loss(pred_pro = prob,
                               tar = test_target)

