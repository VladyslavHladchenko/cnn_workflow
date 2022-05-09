import os

from matplotlib.pyplot import xlabel
from .Results import Result
from . import display_data
import numpy as np 

def print_pid():
    print(f'[PID {os.getpid()}]')

def lrs_near_given(lr):
    return [ round(lr*c, 10) for c in  [5, 3, 1, 0.75, 0.5]]

def lrs_to_str(lrs):
    return [f'{lr = }' for lr in lrs]


def select_best_lr(results, lrs, based_on, descending, plot=True):
    lr_epoch_param = [(lr, e, p) for result, lr in zip(results, lrs) for e, p in result[based_on].items()]

    lr_epoch_param.sort(key=lambda tup: tup[2], reverse = descending)
    best_lr, best_epoch, best_param = lr_epoch_param[0]

    paran_fullname = Result.pstr(based_on)
    print(f'selected lr {best_lr} with {paran_fullname} {best_param:.5f} at epoch {best_epoch}')
    
    if plot:
        title = f'learning rates and epochs sorted by {paran_fullname}'
        ylabel = paran_fullname
        xlabel = 'lr\nepoch'
        format_data = lambda x: f'{np.format_float_scientific(x[0], precision = 1, exp_digits=1,trim="-")}\n{x[1]}'
        xdata = list(map(format_data, lr_epoch_param))
        ydata = list(map(lambda x: x[2], lr_epoch_param))
        bar_labels = [ ('%.5f' % v).rstrip('0').rstrip('.') for l,e,v in lr_epoch_param]

        display_data.plor_bars(title, ylabel, xlabel, xdata, ydata, bar_labels)

    return best_lr

