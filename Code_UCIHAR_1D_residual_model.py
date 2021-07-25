import torch
import torch.nn as nn
import torch.quantization as quantization

class UCIHAR_1D_residual_block(nn.Module):
    def __init__(self, filter_parameters, quantized_switch=False):
        super(UCIHAR_1D_residual_block, self).__init__()
        filter_output_channel = filter_parameters['output_channel']
        filter_size = filter_parameters['filter_size']
        filter_stride = filter_parameters['stride']
        filter_padding = filter_parameters['padding']
        filter_dilation = filter_parameters['dilation']
        filter_groups = filter_parameters['group']
        filter_bias = filter_parameters['bias']
        filter_padding_mode = filter_parameters['padding_mode']

        self.layer1 = nn.Conv1d(in_channels = filter_output_channel,
                                out_channels = filter_output_channel, 
                                kernel_size = filter_size,
                                stride = filter_stride,
                                padding = 2,
                                dilation = filter_dilation,
                                groups = filter_groups,
                                bias = filter_bias,
                                padding_mode = filter_padding_mode)
        self.batch_norm_layer1 = nn.BatchNorm1d(num_features = filter_output_channel)
        self.relu1 = nn.ReLU(inplace=False)
        self.layer2 = nn.Conv1d(in_channels = filter_output_channel,
                                out_channels = filter_output_channel,
                                kernel_size = filter_size,
                                stride = filter_stride,
                                padding = 2,
                                dilation = filter_dilation,
                                groups = filter_groups,
                                bias = filter_bias,
                                padding_mode = filter_padding_mode)
        self.batch_norm_layer2 = nn.BatchNorm1d(num_features = filter_output_channel)
        self.relu2 = nn.ReLU(inplace=False)
        self.quantized_switch = quantized_switch
        self.shortcut_q = nn.quantized.FloatFunctional()

    def forward(self, input):
        hidden1 = self.batch_norm_layer1(self.layer1(input))
        hidden2 = self.relu1(hidden1)
        hidden3 = self.batch_norm_layer2(self.layer2(hidden2))
        hidden4 = self.relu2(hidden3)
        if self.quantized_switch:
            return self.shortcut_q.add(input, hidden4)
        else:
            return torch.add(hidden4, input, alpha=1.0)

class UCIHAR_1D_residual_model(nn.Module):
    def __init__(self, filter_parameters, quantized_switch = False):
        super(UCIHAR_1D_residual_model, self).__init__()
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

        self.quantized_switch = quantized_switch

        self.layer1 = nn.Conv1d(in_channels = filter_input_channel,
                                out_channels = filter_output_channel, 
                                kernel_size = filter_size,
                                stride = filter_stride,
                                padding = filter_padding,
                                dilation = filter_dilation,
                                groups = filter_groups,
                                bias = filter_bias,
                                padding_mode = filter_padding_mode)
        self.batch_norm_layer1 = nn.BatchNorm1d(num_features = filter_output_channel)
        self.relu1 = nn.ReLU(inplace=False)
        self.residual_block = UCIHAR_1D_residual_block(filter_parameters = filter_parameters,
                                                       quantized_switch = quantized_switch)
        self.layer2 = nn.Conv1d(in_channels = filter_output_channel,
                                out_channels = filter_output_channel,
                                kernel_size = filter_size,
                                stride = filter_stride,
                                padding = filter_padding,
                                dilation = filter_dilation,
                                groups = filter_groups,
                                bias = filter_bias,
                                padding_mode = filter_padding_mode)
        self.batch_norm_layer2 = nn.BatchNorm1d(num_features = filter_output_channel)
        self.relu2 = nn.ReLU(inplace=False)
        self.mlp = nn.Linear(in_features = 1200,
                             out_features = filter_num_label,
                             bias = True)
        self.criterion = nn.CrossEntropyLoss()
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, input):
        if self.quantized_switch:
            hidden1 = self.batch_norm_layer1(self.layer1(self.quant(input)))
        else:
            hidden1 = self.batch_norm_layer1(self.layer1(input))
        hidden2 = self.relu1(hidden1)
        res_out = self.residual_block(hidden2)
        hidden3 = self.batch_norm_layer2(self.layer2(res_out))
        hidden4 = self.relu2(hidden3)
        hidden4flatten = torch.flatten(hidden4, start_dim=1)
        if self.quantized_switch:
            hidden5 = self.dequant(self.mlp(hidden4flatten))
        else:
            hidden5 = self.mlp(hidden4flatten)
        return hidden5

    def cal_loss(self, pred_pro, tar):
        return self.criterion(pred_pro, tar)

    def fuse_model(self):
        # print('Module before fused : {0}'.format(self))
        torch.quantization.fuse_modules(self,
                                        [['layer1', 'batch_norm_layer1', 'relu1']],
                                        inplace=True)
        for module in self.modules():
            if type(module) == UCIHAR_1D_residual_block:
                torch.quantization.fuse_modules(module,
                                                [['layer1', 'batch_norm_layer1', 'relu1']],
                                                inplace=True)
                torch.quantization.fuse_modules(module,
                                                [['layer2', 'batch_norm_layer2', 'relu2']],
                                                inplace=True)
        
        torch.quantization.fuse_modules(self,
                                        [['layer2', 'batch_norm_layer2', 'relu2']],
                                        inplace=True)
        # print('Module after fused : {0}'.format(self))


if __name__ == '__main__':
    import numpy as np
    test_input = torch.randn(1, 9, 128)
    test_output = np.array([5])
    test_target = torch.LongTensor(test_output)
    print('Shape of test input : {0}'.format(test_input.shape))
    print('Content of target : {0}'.format(test_target))
    print('Shape if target : {0}'.format(test_target.shape))
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
 
    test_residual_model = UCIHAR_1D_residual_model(filter_parameters = filter_parameters,
                                                   quantized_switch = False)
    test_residual_model.eval()
    test_pred = test_residual_model.forward(input = test_input)
    print('test_pred : {0}'.format(test_pred))
    
    torch.save(test_residual_model.state_dict(), 'test.pt')
    
    test_residual_model = UCIHAR_1D_residual_model(filter_parameters = filter_parameters,
                                                   quantized_switch = True)
    test_residual_model.load_state_dict(torch.load('test.pt'))
    test_residual_model.eval()
    test_residual_model.fuse_model()
    test_residual_model.qconfig = torch.quantization.default_qconfig
    # print(test_residual_model.qconfig)
    torch.quantization.prepare(test_residual_model, inplace=True)
    test_pred_q = test_residual_model.forward(input = test_input)
    print('test_pred_q : {0}'.format(test_pred_q))
    torch.quantization.convert(test_residual_model, inplace=True)
    torch.save(test_residual_model.state_dict(), 'test_q.pt')
    test_pred_qq = test_residual_model.forward(input = test_input)
    



