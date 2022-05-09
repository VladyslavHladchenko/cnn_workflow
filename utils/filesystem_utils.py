import pickle
import json
import os
import shutil
from os.path import isdir, join
from easydict import EasyDict as edict


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



def get_subdirs():
    dir = 'runs'
    all_data_folders = [ join(dir, f) for f in os.listdir(dir) if isdir(join(dir, f))]
    return all_data_folders


def filter_subdirs(filters):
    filters = [] if filters is None else filters
    filters = filters if isinstance(filters, list) else [filters]

    all_data_folders = get_subdirs()
    filtered_folders = [df for df in all_data_folders if  all([f in df for f in filters])]

    return filtered_folders

def load_results_filter(filters):
    return load_results_subdirs(filter_subdirs(filters))

def load_results_subdirs(subdirs):
    if not isinstance(subdirs, list):
        return load_results_subdir(subdirs)

    return [load_results_subdir(sd)  for sd in subdirs]

def load_results_subdir(subdir):
    return load_pickle(subdir+'/results.pickle')

def load_params_subdirs(subdirs):
    if not isinstance(subdirs, list):
        return load_params_subdir(subdirs)

    return [load_params_subdir(sd)  for sd in subdirs]

def load_params_subdir(subdir):
    return load_pickle(subdir+'/params.pickle')


def get_model_desc_from_file(data_folder):
    model_params = load_params_subdir(data_folder)
    return f"{model_params.model_name} {model_params.model_note}"

def get_model_desc_fw_time(data_folder):
    model_params_filename = f'{data_folder}/params.pickle'
    model_params = load_pickle(model_params_filename)

    return f"{model_params.start_time} {model_params.model_name}"


def load_pickle(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    return obj

def clean_runs(filters=None):
    data_folders = filter_subdirs(filters)
    for df in data_folders:
        shutil.rmtree(df, ignore_errors=True)

def clean_models(filters = None):
    data_folders = filter_subdirs(filters)
    for df in data_folders:
        shutil.rmtree(df+'/models', ignore_errors=True)