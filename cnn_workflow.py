from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pprint
from Results import Results, Result
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import dotdict, get_model_desc, load_pickle, get_model_name, get_model_note
from collections import OrderedDict
import pickle
import json

from tqdm.autonotebook import tqdm

from easydict import EasyDict as edict

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from args import parser


def new_network(args):
    net = args.model_class()
    net.to(args.device)
    return net

def save_model(model:nn.Module, dir, epoch):
    filename = f'{dir}/models/{get_model_name(model)}_{epoch}.pt'
    torch.save({'model_state_dict': model.state_dict(), 'note':get_model_note(model)}, filename)


def get_acc(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()/len(labels)


def createDirs(start_time, **kwargs):
    args = edict(kwargs)
    args.basedir=args.get('basedir','runs')

    if not args.save_epochs:
        return None,None,None

    d = start_time.strftime('%Y_%m_%d %H_%M_%S')
    dir = f"{args.basedir}/{d} {args.model_desc} e={args.epoch_num} lr={args.lr}"
    models_dir = f'{dir}/models'
    os.makedirs(models_dir)

    print(f"Created dir: {dir}")

    return f'{dir}'


def saveResults(results, dir):
    """

    if epoch is not None save data for current epoch 
    else save all results 

    """

    with open(f'{dir}/results.pickle', 'wb') as handle:
        pickle.dump(results, handle)

    with open(f'{dir}/results.json', 'w') as handle:
        json.dump(results, handle, indent=4)


def saveParams(start_time, dir, model, opt, **kwargs):
    args = edict(kwargs)

    if not args.save_epochs:
        return

    d = dotdict()
    d.start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
    d.data_dir = dir
    d.epoch_num = args.epoch_num
    d.model_class = model.__class__
    d.model_name = get_model_name(model)
    d.model_note = get_model_note(model)
    d.opt_class = opt.__class__
    d.opt_lr = opt.param_groups[0]['lr']

    with open(f'{dir}/params.pickle', 'wb') as handle:
        pickle.dump(d, handle)

    with open(f'{dir}/params.json', 'w') as handle:
        json.dump({k:str(d[k]) for k in d }, handle, indent=4)


def load_state_to_network(device, statefilename, model):
    state = torch.load(open(statefilename, "rb") , map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.note = state['note']

#TODO: rework
def __load_models_and_params(data_folders,device, model=None):
    model_titles = []
    models = []
    results = []

    for df in data_folders:
        resultsfilename = f'{df}/results.pickle'
        results.append(load_pickle(resultsfilename))

        model_params_filename = f'{df}/params.pickle'
        model_params = load_pickle(model_params_filename)
        model_params.device=device

        model_state_filename = f'{df}/models/{model_params.model_name}.pt' #TODO: model without epoch is not saved 
        if not model:
            model = new_network(model_params)
        load_state_to_network(device, model_state_filename, model) 
        models.append(model)

        model_titles.append(f"{model_params.model_name} {model_params.model_note} {model_params.epochs} lr={model_params.opt_lr}")

    return models, results, model_titles


def train_single_batch(model, trn_data, val_data, opt, device, loss_fn=F.cross_entropy):
    model.train()

    trn_x = trn_data[0].to(device)
    trn_labels = trn_data[1].to(device)

    val_x = val_data[0].to(device)
    val_labels = val_data[1].to(device)

    opt.zero_grad()
    output = model(trn_x)
    trn_loss = loss_fn(output, trn_labels, reduction='sum')
    trn_loss.backward()
    opt.step()

    model.eval()
    # cumpute accuracy on training batch
    trn_acc = get_acc(output, trn_labels)

    # cumpute accuracy on validation batch
    output = model(val_x)
    val_loss = loss_fn(output, val_labels, reduction='sum')
    val_acc = get_acc(output, val_labels)

    # write result
    result = Result(trn_loss = trn_loss.item()/len(trn_labels),
                     val_loss = val_loss.item()/len(val_labels),
                     trn_acc = trn_acc,
                     val_acc = val_acc)

    return result


def train_single_epoch(args, model, device, data_loader, opt, epoch, loss_fn, scheduler=None):
    n_batches = len(data_loader.train_loader)

    disable_tqdm = args.get('disable_tqdm', False)

    result = Result()

    val_data = next(iter(data_loader.val_loader))

    for batch_idx, trn_data in tqdm(enumerate(data_loader.train_loader),
                                    total=n_batches,
                                    desc=f'epoch {epoch+1}',
                                    disable=disable_tqdm):
        out = train_single_batch(model, trn_data, val_data, opt, device, loss_fn)
        result += out
    result/= n_batches

    if scheduler:
        scheduler.step()

    return result


def log_to_wandb(epoch, result):
    wandb.log({"training loss": result.trn_loss
              ,"validation loss": result.val_loss},
              step=epoch,
              commit=False)
    wandb.log({"training accuracy": result.trn_acc,
               "validation accuracy": result.val_acc},
               step=epoch)

def make_tqdm_postfix(result):
    s= f"tl {result.trn_loss:.4f} vl {result.val_loss:.4f} ta {result.trn_acc:.4f} va {result.val_acc:.4f}"
    return s

def set_tqdm_postfix(t, result):
    t.set_postfix_str(make_tqdm_postfix(result))

def _get_train_args(opt, kwargs):
    args = edict(kwargs)
    args.wandb = args.get('wandb', False)
    args.lr = args.get('lr', opt.param_groups[0]['lr'])
    args.loss_fn = args.get('loss_fn', F.cross_entropy)
    args.save_epochs = args.get('save_epochs', False)
    args.disable_tqdm = args.get('disable_tqdm', False)
    args.scheduler = args.get('scheduler', None)
    return args

def train(model, device, data_loader, opt, **kwargs):
    args = _get_train_args(opt, kwargs)

    start_time=datetime.today()
    dir = createDirs(start_time, model_desc = get_model_desc(model), **args)
    saveParams(start_time, dir, model, opt, **args) # TODO: all like this : args.save_epochs and ?

    train_args = dotdict({'disable_tqdm': True})

    results = Results()

    t=tqdm(range(1, args.epoch_num + 1), disable=args.disable_tqdm)
    for epoch in t:
        out = train_single_epoch(train_args, model, device, data_loader, opt, epoch, loss_fn=args.loss_fn, scheduler=args.scheduler)
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
        # print(f"{lr=:.6f}")

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


def get_predictions_sorted_by_confidence(model, device, test_loader):
    model.eval()
    probs_list = []
    target_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs_list.append(F.softmax(output,dim=1))
            target_list.append(target)

    prob = torch.concat(probs_list)
    target = torch.concat(target_list)
    confidence, pred = torch.max(prob,1)

    sort_idx = torch.argsort(confidence)

    return pred[sort_idx].cpu().numpy(), confidence[sort_idx].cpu().numpy(), target[sort_idx].cpu().numpy()



def get_args():
    args = parser.parse_args()
    args.wandb = args.wandb and wandb != None
    return args

def init_wandb(**kwargs):
    with open("wandb_config.json", "r") as f:
        wandb_config = json.load(f)

    wandb.init(**wandb_config,config=kwargs)
