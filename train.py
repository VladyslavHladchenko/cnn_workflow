from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pprint

import torch
import networks
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import dotdict, load_pickle
from collections import OrderedDict
import pickle

from tqdm.autonotebook import tqdm

# this is a command line program which can be run with different options
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b","--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("-m", "--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--epochs", type=int, default=1, help="training epochs")


class Data():
    def __init__(self, args=None):
        args = dotdict() if args == None else args

        # transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        val_fraction = args.get('val_fraction',0.1)
        batch_size = args.get('batch_size',64)

        train_set = torchvision.datasets.MNIST('../data', download=True, train=True, transform=transform)
        self.test_set = torchvision.datasets.MNIST('../data', download=True, train=False, transform=transform)

        # split original train set into train and validation
        val_size = int(val_fraction*len(train_set))
        train_size = len(train_set) - val_size
        self.train_set, self.val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

        # dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=val_size, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=True, num_workers=0)


def new_network(args):
    net = args.model_class()
    net.to(args.device)
    return net

def save_network(net:nn.Module, filename):
    torch.save({'model_state_dict': net.state_dict()}, filename)

def load_network(args, filename):
    state = torch.load(open(filename, "rb") , map_location=args.device)
    net = new_network(args)
    net.load_state_dict(state['model_state_dict'])
    return net


def evaluate(model: nn.Module, loader, device):
    ## perform evaluation of the network on the data given by the loader
    model.eval()

    loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum')
            acc += get_acc(output, target)

    loss /= len(loader.dataset)
    acc /= len(loader)

    return loss, acc


def get_acc(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()/len(labels)


def createDirs(do_it):
    if not do_it:
        return None,None

    d = datetime.today().strftime('%Y-%m-%dT%H_%M_%S')
    models = f'runs/{d}/models'
    results = f'runs/{d}/results'
    os.makedirs(models)
    os.makedirs(results)
    print("Created dirs: ")
    print(models)
    print(results)
    return models, results


def saveResults(do_it, results, model, dirs, epoch=None):
    """

    if epoch is not None save data for current epoch 
    else save all results 

    """
    if not do_it:
        return

    models_dir, results_dir = dirs

    # save results of all epochs
    with open(f'{results_dir}/results.pickle', 'wb') as handle:
            pickle.dump(results, handle)

    if epoch == None:
        return

    # save results of current epoch
    result_last_epoch = dotdict()
    result_last_epoch.trn_loss = results.trn_loss[epoch]
    result_last_epoch.trn_acc = results.trn_acc[epoch]
    result_last_epoch.val_loss = results.val_loss[epoch]
    result_last_epoch.val_acc = results.val_acc[epoch]

    with open(f'{results_dir}/results_{epoch}.pickle', 'wb') as handle:
        pickle.dump(result_last_epoch, handle)

    net_name = model.__class__.__name__
    save_network(model, f'{models_dir}/{net_name}_{epoch}.pt')
    save_network(model, f'{models_dir}/{net_name}.pt')


def saveParams(do_it, dirs, model, opt):
    if not do_it:
        return
    models_dir, results_dir = dirs
    
    d = dotdict()
    d.model_class = model.__class__
    d.model_name = model.__class__.__name__
    d.opt_class = opt.__class__
    d.opt_lr = opt.param_groups[0]['lr']

    with open(f'{models_dir}/params.pickle', 'wb') as handle:
        pickle.dump(d, handle)

    with open(f'{models_dir}/params.txt', 'w') as handle:
        handle.write(pprint.pformat(d))

def load_models_and_params(data_folders,device):
    model_titles = []
    models = []
    results = []

    for df in data_folders:
        filename = f'runs/{df}/results/results.pickle'
        results.append(load_pickle(filename))

        model_params_filename = f'runs/{df}/models/params.pickle'
        model_params = load_pickle(model_params_filename)
        model_params.device=device

        model_filename = f'runs/{df}/models/{model_params.model_name}.pt'
        model = load_network(model_params, model_filename)
        models.append(model)

        model_titles.append(f"{model_params.model_name} lr={model_params.opt_lr}")

    return models, results, model_titles


def train_single_batch(model, trn_data, val_data, opt, device):
    model.train()

    trn_x = trn_data[0].to(device)
    trn_labels = trn_data[1].to(device)

    val_x = val_data[0].to(device)
    val_labels = val_data[1].to(device)

    opt.zero_grad()
    output = model(trn_x)
    trn_loss = F.nll_loss(output, trn_labels, reduction='sum')
    trn_loss.backward()
    opt.step()

    model.eval()
    # cumpute accuracy on training batch
    trn_acc = get_acc(output, trn_labels)

    # cumpute accuracy on validation batch
    output = model(val_x)
    val_loss = F.nll_loss(output, val_labels,reduction='sum')
    val_acc = get_acc(output, val_labels)

    # write results
    results = dotdict()
    results.trn_loss = trn_loss.item()/len(trn_labels)
    results.val_loss = val_loss.item()/len(val_labels)
    results.trn_acc = trn_acc
    results.val_acc = val_acc

    return results


def train_single_epoch(args, model, device, data_loader, opt, epoch):
    n_data = len(data_loader.train_set)
    n_batches = len(data_loader.train_loader)

    disable_tqdm = args.get('disable_tqdm', False)

    trn_loss_sum = 0
    val_loss_sum = 0
    trn_acc_sum = 0
    val_acc_sum = 0

    val_data = next(iter(data_loader.val_loader))

    for batch_idx, trn_data in tqdm(enumerate(data_loader.train_loader),
                                    total=n_batches,
                                    desc=f'epoch {epoch+1}',
                                    disable=disable_tqdm):
        out = train_single_batch(model, trn_data, val_data, opt, device)

        trn_loss_sum += out.trn_loss
        val_loss_sum += out.val_loss
        trn_acc_sum += out.trn_acc
        val_acc_sum += out.val_acc


    results = dotdict()
    results.trn_loss = trn_loss_sum/n_batches
    results.val_loss = val_loss_sum/n_batches
    results.trn_acc = trn_acc_sum/n_batches
    results.val_acc = val_acc_sum/n_batches

    return results


def train(model, device, epoch_num, data_loader, opt, save_each_epoch=True):
    dirs = createDirs(save_each_epoch)
    saveParams(save_each_epoch, dirs, model, opt)

    train_args = dotdict()
    train_args.disable_tqdm = True

    results = dotdict()
    results.trn_loss = OrderedDict()
    results.val_loss = OrderedDict()
    results.trn_acc = OrderedDict()
    results.val_acc = OrderedDict()

    for epoch in tqdm(range(1, epoch_num+1), desc="epochs progress"):
        out = train_single_epoch(train_args, model, device, data_loader, opt, epoch)
        results.trn_loss[epoch] = out.trn_loss
        results.val_loss[epoch] = out.val_loss
        results.trn_acc[epoch] = out.trn_acc
        results.val_acc[epoch] = out.val_acc

        saveResults(save_each_epoch, results, model, dirs, epoch)

    saveResults(save_each_epoch, results, model, dirs)

    return results

def range_test(lb, ub, device, data_loader, model, opt_type=optim.SGD):
    train_args = dotdict()
    train_args.log_interval = 600

    num = len(data_loader.train_loader)

    results = OrderedDict()
    results.trn_loss = OrderedDict()
    results.val_loss = OrderedDict()
    results.trn_acc = OrderedDict()
    results.val_acc = OrderedDict()

    lrs = np.linspace(lb, ub, num)

    val_data = next(iter(data_loader.val_loader))
    for trn_data, lr in tqdm(list(zip(data_loader.train_loader, lrs))):
        # print(f"{lr=:.6f}")

        opt=opt_type(model.parameters(),lr=lr)

        out = train_single_batch(model, trn_data, val_data, opt, device)

        results.trn_loss[lr] = out.trn_loss
        results.val_loss[lr] = out.val_loss
        results.trn_acc[lr] = out.trn_acc
        results.val_acc[lr] = out.val_acc

    return results


def test(model, device, test_loader):
    l, acc = evaluate(model, test_loader, device)
    print(f"loss {l:.4f}, accuracy {acc}")


def get_predictions_sorted_by_confidence(model, device, test_loader):
    model.eval()
    
    outputs_list = []
    target_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs_list.append(model(data))
            target_list.append(target)

    output = torch.concat(outputs_list)
    target = torch.concat(target_list)
    
    logsoftmax, pred = torch.max(output,1)
    confidence = torch.exp(logsoftmax)

    sord_idx = torch.argsort(confidence)

    return pred[sord_idx].cpu().numpy(), confidence[sord_idx].cpu().numpy(), target[sord_idx].cpu().numpy()




if __name__ == "__main__":
    args = parser.parse_args()
    # query if we have GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    data_args = dotdict()
    data_args.batch_size = 64
    data_args.val_fraction = 0.1
    data_loader = Data(data_args)

    model = networks.Net().to(device)
    opt = optim.Adadelta(model.parameters(), lr=1)
    epoch_num=30
    results = train(model, device, epoch_num, data_loader, opt, save_each_epoch=True)

    test(model, device, data_loader.test_loader)
