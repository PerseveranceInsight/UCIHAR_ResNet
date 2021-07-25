import numpy as np
import os
import torch
import torch.nn as nn
import torch.quantization as quantization

import Code_UCIHAR_1D_DataWrapper as DW
import Code_UCIHAR_1D_model as model
import Code_UCIHAR_1D_residual_model as residual_model
from torch.utils.data import Dataset, DataLoader

def get_device(quantized_switch=False):
    if quantized_switch:
        return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(),
                                                          **config['optim_hparas'])
    min_loss = 1000.
    running_loss = 0.
    loss_record = {'train': [], 'dev': [], 'running_loss': [], 'running_epoch': [], 'dev_epoch': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        print('epoch : {0}'.format(epoch))
        running_loss = 0
        model.train()                                  
        for x, y in tr_set:                             
            optimizer.zero_grad()                       
            x, y = x.to(device), y.to(device)           
            pred = model(x.permute(0,2,1))              
            loss = model.cal_loss(pred, y)              
            running_loss += loss
            loss.backward()
            optimizer.step()
            loss_record['train'].append(loss.detach().cpu().item())
        running_loss /= len(tr_set)
        loss_record['running_loss'].append(running_loss.detach().cpu().item())
        loss_record['running_epoch'].append(epoch)
        dev_loss = dev(dv_set, model, device)
        if dev_loss < min_loss: 
            min_loss = dev_loss
            print('Saving model (epoch {:4d}, loss = {:.4f})'.format(epoch + 1, min_loss))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_loss)
        loss_record['dev_epoch'].append(epoch)
        if early_stop_cnt > config['early_stop']:
            break
    print('Finished training after {} epochs'.format(epoch))
    return min_loss, loss_record


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x.permute(0,2,1))
            loss = model.cal_loss(pred, y)
        total_loss += loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)
    return total_loss

def pred(data_set, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    num_pred_success = 0
    success_rate = 0
    for x, y in data_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model.forward(x.permute(0,2,1))
            # pred = softmax(pred)
            pred_class = torch.argmax(pred)
            label = y.numpy().astype(int)[0]
            if (pred_class == label):
                num_pred_success += 1
    success_rate = (num_pred_success*100)/len(data_set)
    print('Success rate : {0:.4f}'.format(success_rate))
    return success_rate

def main():
    load_model = True
    retrain = False
    residual_learning = True
    quantized_switch = True

    model_name = ''
    
    if residual_learning:
        model_name = './models/UCIHAR_1D_Residual_Model.pt'
        quantized_model_name = './models/UCIHAR_1D_Residual_Model_q.pt'
    else:
        model_name = './models/UCIHAR_1D_Model.pt'
        quantized_model_name = './models/UCIHAR_1D_Model_q.pt'

    batch_size = 200

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

    train_labelfile = './UCIHAR/train/y_train.txt'
    test_labelfile = './UCIHAR/test/y_test.txt'
    activities_descriptions = {1: 'walking',
                               2: 'walking upstairs',
                               3: 'walking downstairs',
                               4: 'sitting',
                               5: 'stannding',
                               6: 'laying'}
    train_folder_prefix = './UCIHAR/train/Inertial_Signals/'
    test_folder_prefix = './UCIHAR/test/Inertial_Signals/'
    train_files_suffix = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                          'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                          'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']
    test_files_suffix = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                         'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                         'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']
    train_config = { 'n_epochs': 5000,
                     'batch_size': 300,
                     'optimizer': 'Adagrad',
                     'optim_hparas': { 'lr': 0.005,
                                       'lr_decay': 0.001},
                     'early_stop': 200,
                     'save_path':  model_name} 

    random_seed = 42609
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = get_device(quantized_switch)
    print('device : {0}'.format(device))

    train_dataset = DW.UCIHARDataset(signal_path_prefix = train_folder_prefix,
                                  signal_path_suffix = train_files_suffix,
                                  signal_labelfile = train_labelfile,
                                  mode='train')
    val_dataset = DW.UCIHARDataset(signal_path_prefix = train_folder_prefix,
                                   signal_path_suffix = train_files_suffix,
                                   signal_labelfile = train_labelfile,
                                   mode='val')
    test_dataset = DW.UCIHARDataset(signal_path_prefix = test_folder_prefix,
                                    signal_path_suffix = test_files_suffix,
                                    signal_labelfile = test_labelfile,
                                    mode='test')
    train_set = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False,
                           num_workers=0, pin_memory=True)
    val_set = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=False,
                         num_workers=0, pin_memory=True)
    test_set = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,
                          num_workers=0, pin_memory=True)

    if load_model:
        if os.path.exists(train_config['save_path']):
            if residual_learning:
                ucihar_model = residual_model.UCIHAR_1D_residual_model(filter_parameters = filter_parameters,
                                                                       quantized_switch = quantized_switch).to(device)
                ucihar_model.load_state_dict(torch.load(train_config['save_path']))
            else:
                ucihar_model = model.UCIHAR_1D_model(filter_parameters = filter_parameters,
                                                     pool_parameters = pool_parameters).to(device)
                ucihar_model.load_state_dict(torch.load(train_config['save_path']))
        else:
            ucihar_model = None
    else:
        if residual_learning:
            ucihar_model = residual_model.UCIHAR_1D_residual_model(filter_parameters = filter_parameters).to(device)
        else:
            ucihar_model = model.UCIHAR_1D_model(filter_parameters = filter_parameters,
                                                 pool_parameters = pool_parameters).to(device)

    if ucihar_model is not None:
        if retrain:
            print('Start training')
            model_loss, model_loss_record = train(tr_set = train_set, 
                                                  dv_set = val_set, 
                                                  model = ucihar_model, 
                                                  config = train_config, 
                                                  device = device)
            np.save('./models/train_loss.npy', np.asarray(model_loss_record['train']))
            np.save('./models/val_loss.npy', np.asarray(model_loss_record['dev']))
            np.save('./models/running_loss.npy', np.asarray(model_loss_record['running_loss']))
            np.save('./models/running_epoch.npy', np.asarray(model_loss_record['running_epoch']))
            np.save('./models/val_epoch.npy', np.asarray(model_loss_record['dev_epoch']))
        else:
            if quantized_switch:
                ucihar_model.eval()
                ucihar_model.fuse_model()
                ucihar_model.qconfig = torch.quantization.default_qconfig
                torch.quantization.prepare(ucihar_model, inplace=True)
                dev(val_set, ucihar_model, device)
                ucihar_model = torch.quantization.convert(ucihar_model, inplace=True) 
                torch.save(ucihar_model.state_dict(), quantized_model_name)
                pred_rate = pred(data_set = test_set,
                                 model = ucihar_model,
                                 device = device)
            else:
                pred_rate = pred(data_set = test_set, 
                                 model = ucihar_model, 
                                 device = device)
    else:
        print('Model doesn\' exit')


if __name__ == '__main__':
    main()
