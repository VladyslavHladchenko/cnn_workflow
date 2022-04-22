from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import dotdict, load_pickle
from collections import OrderedDict
import pickle
import json

from tqdm.autonotebook import tqdm

from easydict import EasyDict as edict

try:
    import wandb
except ModuleNotFoundError:
    wandb = False

from args import parser


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


def createDirs(start_time, **kwargs):
    args = edict(kwargs)

    if not args.save_epochs:
        return None,None,None

    d = start_time.strftime('%Y-%m-%d %H_%M_%S')
    subdir = f"{d} {args.model_name} e={args.epoch_num} lr={args.lr}"
    models_dir = f'runs/{subdir}/models'
    results_dir = f'runs/{subdir}/results'
    os.makedirs(models_dir)
    os.makedirs(results_dir)

    print("Created dirs: ")
    print("\t" + models_dir)
    print("\t" + results_dir)

    return subdir, models_dir, results_dir


def saveResults(do_it, results, model, dirs, epoch=None):
    """

    if epoch is not None save data for current epoch 
    else save all results 

    """
    if not do_it:
        return

    subdir, models_dir, results_dir = dirs

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


def saveParams(start_time, dirs, model, opt, **kwargs):
    args = edict(kwargs)

    if not args.save_epochs:
        return

    subdir, models_dir, results_dir = dirs
    
    d = dotdict()
    d.start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    d.data_dir = subdir
    d.epoch_num = args.epoch_num
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

        model_titles.append(f"{model_params.model_name} {model_params.epochs} lr={model_params.opt_lr}")

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

def log_to_wandb(do_it, epoch, result):
    if not do_it:
        return 
    wandb.log({"training loss": result.trn_loss
              ,"validation loss": result.val_loss},
              step=epoch,
              commit=False)
    wandb.log({"training accuracy": result.trn_acc,
               "validation accuracy": result.val_acc},
               step=epoch)


def train(model, device, data_loader, opt, **kwargs):
    args = edict(kwargs)
    args.wandb = args.get('wandb', False)
    args.lr = args.get('lr', opt.param_groups[0]['lr'])

    start_time=datetime.today()
    dirs = createDirs(start_time, model_name = model.__class__.__name__, **kwargs)
    saveParams(start_time, dirs, model, opt, **kwargs)

    train_args = dotdict()
    train_args.disable_tqdm = True

    results = dotdict()
    results.trn_loss = OrderedDict()
    results.val_loss = OrderedDict()
    results.trn_acc = OrderedDict()
    results.val_acc = OrderedDict()

    t=tqdm(range(1, args.epoch_num + 1), disable=kwargs.get('disable_tqdm', False))
    for epoch in t:
        out = train_single_epoch(train_args, model, device, data_loader, opt, epoch)
        results.trn_loss[epoch] = out.trn_loss
        results.val_loss[epoch] = out.val_loss
        results.trn_acc[epoch] = out.trn_acc
        results.val_acc[epoch] = out.val_acc
        s= f"tl {out.trn_loss:.4f} vl {out.val_loss:.4f} ta {out.trn_acc:.4f} va {out.val_acc:.4f}" if kwargs.get('do_log',True) else None

        t.set_postfix(result=s) #TODO: get rif of 'result' key

        log_to_wandb(args.wandb, epoch, out)

        saveResults(args.save_epochs, results, model, dirs, epoch)

    saveResults(args.save_epochs, results, model, dirs)

    return results

def range_test(lb, ub, device, data_loader, model, opt_type=optim.SGD):
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
    return l, acc


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


def get_args():
    args = parser.parse_args()
    args.wandb = args.wandb and wandb != None
    return args

def init_wandb(**kwargs):
    with open("wandb_config.json", "r") as f:
        wandb_config = json.load(f)

    wandb.init(**wandb_config,config=kwargs)
