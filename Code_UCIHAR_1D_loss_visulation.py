import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    running_epoch_path = './models/running_epoch.npy'
    running_loss_path = './models/running_loss.npy'
    val_epoch_path = './models/val_epoch.npy'
    val_loss_path = './models/val_loss.npy'
    running_epoch = np.load(running_epoch_path)
    running_loss = np.load(running_loss_path)
    val_epoch = np.load(val_epoch_path)
    val_loss = np.load(val_loss_path)
    
    plt.figure()
    plt.title('Loss')
    plt.plot(running_epoch, running_loss, 'bx-', label='running loss')
    plt.plot(val_epoch,val_loss, 'ro-', label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy loss')
    plt.legend()