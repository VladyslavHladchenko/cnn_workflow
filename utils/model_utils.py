import torch

import pickle
import json
from .filesystem_utils import load_pickle
from .dotdict import dotdict

from easydict import EasyDict as edict

def new_network(model_class, device):
    net = model_class()
    net.to(device)
    return net

#TODO: rework
def load_model(model_name, device):
    import networks
    model_class = getattr(networks, model_name)
    model = model_class().to(device)
    return model


def load_state_to_network(device, statefilename, model):
    state = torch.load(open(statefilename, "rb") , map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.note = state['note']


def save_model(model:torch.nn.Module, dir, epoch):
    filename = f'{dir}/models/{get_model_name(model)}_{epoch}.pt'
    torch.save({'model_state_dict': model.state_dict(), 'note':get_model_note(model)}, filename)


def add_model_note(model, note):
    model_note = getattr(model,'note','')
    model.note = model_note + (model_note != '')*' ' + note

def get_model_note(model):
    return getattr(model,'note','')

def get_model_name(model):
    return model.__class__.__name__

def get_model_desc(model):
    return f"{get_model_name(model)} {get_model_note(model)}"


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
