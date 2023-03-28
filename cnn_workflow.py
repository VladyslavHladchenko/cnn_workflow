import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from .utils import Results, Result
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import dotdict, get_model_desc, log_to_wandb, createDirs, saveParams, set_tqdm_postfix, save_model, saveResults
from collections import OrderedDict

from tqdm.autonotebook import tqdm

from easydict import EasyDict as edict


def get_acc(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()/len(labels)


def train_single_batch(model, trn_data, val_data, opt, device, loss_fn=F.cross_entropy, **kwargs):

    pre_step = kwargs.get('pre_step', None)
    pre_step_sigma = kwargs.get('pre_step_sigma', None)

    model.train()

    trn_x = trn_data[0].to(device)
    trn_labels = trn_data[1].to(device)

    val_x = val_data[0].to(device)
    val_labels = val_data[1].to(device)

    opt.zero_grad()
    output = model(trn_x)
    trn_loss = loss_fn(output, trn_labels, reduction='sum')
    trn_loss.backward()

    if pre_step:
        pre_step(model, loss_fn, trn_x, trn_labels, pre_step_sigma)

    opt.step()

    model.eval()
    # compute accuracy on training batch
    trn_acc = get_acc(output, trn_labels) if not kwargs.get('no_acc', False) else 0

    # compute accuracy on validation batch
    output = model(val_x)
    val_loss = loss_fn(output, val_labels, reduction='sum')
    val_acc = get_acc(output, val_labels) if not kwargs.get('no_acc', False) else 0

    # write result
    result = Result(trn_loss = trn_loss.item()/len(trn_labels),
                     val_loss = val_loss.item()/len(val_labels),
                     trn_acc = trn_acc,
                     val_acc = val_acc)

    return result


def train_single_epoch(args, model, device, data_loader, opt, epoch, loss_fn, **kwargs):
    n_batches = len(data_loader.train_loader)

    disable_tqdm = args.get('disable_tqdm', False)

    result = Result()

    val_data = next(iter(data_loader.val_loader))

    for batch_idx, trn_data in tqdm(enumerate(data_loader.train_loader),
                                    total=n_batches,
                                    desc=f'epoch {epoch+1}',
                                    disable=disable_tqdm):
        out = train_single_batch(model, trn_data, val_data, opt, device, loss_fn, **kwargs)
        result += out
    result/= n_batches

    return result


def _get_train_args(opt, kwargs):
    args = edict(kwargs)
    args.wandb = args.get('wandb', False)
    args.lr = args.get('lr', opt.param_groups[0]['lr'])
    args.loss_fn = args.get('loss_fn', F.cross_entropy)
    args.save_epochs = args.get('save_epochs', False)
    args.disable_tqdm = args.get('disable_tqdm', False)
    args.scheduler = args.get('scheduler', None)
    args.pre_step = args.get('pre_step', None)
    args.pre_step_sigma = args.get('pre_step_sigma', None)
    args.no_acc = args.get('no_acc', False)
    return args

def train(model, device, data_loader, opt, **kwargs):
    args = _get_train_args(opt, kwargs)

    start_time=datetime.today()
    dir = createDirs(start_time, model_desc = get_model_desc(model), **args)
    saveParams(start_time, dir, model, opt, **args)

    train_args = dotdict({'disable_tqdm': True})

    results = Results()

    t=tqdm(range(1, args.epoch_num + 1), disable=args.disable_tqdm)
    for epoch in t:
        out = train_single_epoch(train_args, model, device, data_loader, opt, epoch, loss_fn=args.loss_fn, pre_step=args.pre_step, pre_step_sigma=args.pre_step_sigma, no_acc = args.no_acc)
        results.save_epoch(epoch, out)
        set_tqdm_postfix(t, out)

        args.wandb and log_to_wandb(epoch, out)
        args.save_epochs and save_model(model, dir, epoch)

    args.save_epochs and saveResults(results, dir)

    return results


#TODO: refactor
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
        opt=opt_type(model.parameters(),lr=lr)

        out = train_single_batch(model, trn_data, val_data, opt, device)

        results.trn_loss[lr] = out.trn_loss
        results.val_loss[lr] = out.val_loss
        results.trn_acc[lr] = out.trn_acc
        results.val_acc[lr] = out.val_acc

    return results


def test(model, device, test_loader, loss_fun):
    l, acc = evaluate(model, test_loader, device, loss_fun)
    print(f"loss {l:.4f}, accuracy {acc:.4f}")
    return l, acc

def evaluate(model: nn.Module, device, loader, loss_fun=F.cross_entropy):
    ## perform evaluation of the network on the data given by the loader
    model.eval()

    loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += loss_fun(output, target, reduction='sum').item()
            acc += get_acc(output, target)  

    loss /= len(loader.dataset)
    acc /= len(loader)

    return loss, acc
