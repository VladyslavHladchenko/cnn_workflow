from cProfile import label
from tkinter import font
from unittest import result
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE
import pickle
import os

def load_pickle(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    """ For a more elaborate solution take a look at the EasyDict package https://pypi.org/project/easydict/ """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # these are needed for deepcopy / pickle
    def __getstate__(self): return self.__dict__

    def __setstate__(self, d): self.__dict__.update(d)


def plot_dict_data_figure(data, title, xlabel, ylabel):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    data_x = list(data.keys())
    data_y = list(data.values())
    plt.xticks(data_x)
    plt.plot(data_x, data_y)


def plot_dict_data(title, xlabel, ylabel,legend, data):
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for d in data:
        data_x = list(d.keys())
        data_y = list(d.values())
        # plt.xticks(data_x[:-1:16] + [data_x[-1]])
        plt.plot(data_x, data_y)
    plt.legend(legend)


def display_results_loss(results, title_postfix=""):
    legend = []
    data=[]
    if results.trn_loss:
        title = 'training loss over epochs'
        legend.append('training loss')
        data.append(results.trn_loss)
    if results.val_loss:
        title = 'validation loss over epochs'
        legend.append('validation loss')
        data.append(results.val_loss)

    if results.trn_loss and results.val_loss:
        title = 'training and validation loss over epochs'

    plt.figure()
    plot_dict_data(title + f": {title_postfix}",'epochs', 'loss', legend, data)

def display_results_accuracy(results, title_postfix=""):
    legend = []
    data=[]
    if results.trn_acc:
        title = 'accuracy on training set over epochs'
        legend.append('accuracy on training set')
        data.append(results.trn_acc)
    if results.val_acc:
        title = 'accuracy on validation set over epochs'
        legend.append('accuracy on validation set')
        data.append(results.val_acc)

    if results.trn_acc and results.val_acc:
        title = 'accuracy on training and validation sets over epochs'

    plt.figure()
    plot_dict_data(title + f": {title_postfix}", 'epochs', 'accuracy', legend, data)


def display_results(results, title_postfix=""):
    display_results_loss(results, title_postfix)
    display_results_accuracy(results, title_postfix)


def display_compare(results, name, title, legend):
    data=[]
    plt.figure()
    for r in results:
        attr = getattr(r, name)
        data.append(attr)

    plot_dict_data(title, 'epochs', name, legend, data)


def display_results_compare(results, titles):
    display_compare(results, 'val_acc', 'validation accuracy comparison', titles)
    display_compare(results, 'val_loss', 'validation loss comparison', titles)
    display_compare(results, 'trn_acc', 'training accuracy comparison', titles)
    display_compare(results, 'trn_loss', 'training loss comparison', titles)

def get_confusion(model,device, data_loader, n_classes):
    confusion = np.zeros((n_classes,n_classes), dtype=np.uint64)
    model.eval()

    with torch.no_grad():
        for data, target in data_loader.test_loader:
            data, target = data.to(device), target.to(device)
            predicted = model(data).argmax(dim=1)
            np_targer=target.cpu().detach().numpy()
            np_predicted = predicted.cpu().detach().numpy()

            for t,p in zip(np_targer,np_predicted):
                confusion[t][p] +=1

    return confusion


def plot_confusion(C, title="", fontsize=20):
    f = plt.figure(figsize = (20,20))
    plt.title(f'Confusion matrix: {title}',fontsize=fontsize)
    heatmap=plt.imshow(C, cmap='Blues', interpolation='nearest', vmin=0, vmax=C.max())
    plt.xticks(range(C.shape[1]), fontsize=fontsize)
    plt.yticks(range(C.shape[0]), fontsize=fontsize)
    plt.xlabel('predicted classes', fontsize=fontsize)
    plt.ylabel('true classes', fontsize=fontsize)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            s = str(C[i,j])
            plt.text(j ,i, s,ha="center", va="center", size='xx-large', color='black' if C[i,j] < C.max()//2 else 'white')
    cbar = f.colorbar(heatmap)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.show()


def get_conv_out_size_params(W,K,P,S):
    return ((W-K+2*P)/S)+1


def get_conv_out_size_seq(in_dim, sequential):
    for m in sequential:
        if type(m) == torch.nn.Conv2d:
            in_dim, last_conv_out = get_conv_out_size_layer(in_dim, m)
        elif type(m) == torch.nn.MaxPool2d:
            in_dim, _ = get_conv_out_size_layer(in_dim, m)
        elif type(m) == torch.nn.Sequential:
            in_dim, last_conv_out = get_conv_out_size_seq(in_dim, m)
    return in_dim, last_conv_out


def get_conv_out_size_layer(in_dim, l):
    K = l.kernel_size if type(l.kernel_size) == int else l.kernel_size[0]
    P = l.padding if type(l.padding) == int else l.padding[0]
    S = l.stride if type(l.stride) == int else l.stride[0]

    out_ch = getattr(l, 'out_channels', None)
    return get_conv_out_size_params(in_dim, K, P, S), out_ch


def get_linear_in_size(in_dim, layers):
    featres_out_dim,last_kernel_depth = get_conv_out_size_seq(in_dim, layers)
    linear_in = int(featres_out_dim*featres_out_dim*last_kernel_depth)

    return linear_in

def plot_tsne(features, labels, title=""):
    n_data = len(labels)
    X = features.reshape((n_data,-1))
    tsne = TSNE(n_components=2, random_state=0)
    # Project the data in 2D

    X_2d = tsne.fit_transform(X)
    # Visualize the data

    target_ids = range(n_data)

    plt.figure(figsize=(15, 15))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'dimgray', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, labels):
        plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=c, label=label)
    plt.legend()
    plt.title(f'features : {title}')


def get_model_desc(data_folder):
    model_params_filename = f'runs/{data_folder}/models/params.pickle'
    model_params = load_pickle(model_params_filename)

    return f"{model_params.start_time} {model_params.model_name} {model_params.epochs} lr={model_params.opt_lr}"


def print_pid():
    print(f'[PID {os.getpid()}]')


def load_model(model_name, device):
    import networks
    model_class = getattr(networks, model_name)
    model = model_class().to(device)
    print('info:')
    return model

def get_optimizer(opt_name, *opt_args, **opt_kwargs):
    opt_class = getattr(torch.optim, opt_name)
    opt = opt_class(*opt_args, **opt_kwargs)

    return opt
