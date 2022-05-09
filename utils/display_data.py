import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from . import dataprocessing, Result
import numpy as np


def plot_dict_data(title, xlabel, ylabel,legend, data):
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for d in data:
        data_x = list(d.keys())
        data_y = list(d.values())
        # plt.xticks(data_x[:-1:16] + [data_x[-1]])
        plt.xticks(data_x)
        plt.plot(data_x, data_y)
    plt.legend(legend)


def display_compare(results, paramname, legend):
    data=[]
    plt.figure()
    for r in results:
        attr = getattr(r, paramname)
        data.append(attr)

    ylabel = Result.pstr(paramname)
    plot_dict_data(ylabel + ' comparison', 'epoch', ylabel, legend, data)

def display_compare_test(results, paramname, legend):
    data=[]
    plt.figure()
    for r in results:
        attr = getattr(r, paramname, None)
        attr and data.append(attr)

    if not data:
        return

    fig, ax = plt.subplots()
    ylabel = Result.pstr(paramname)
    ax.set_title(ylabel + ' comparison')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('networks')
    bars = ax.bar(legend, data)
    ax.bar_label(bars)
    plt.grid()



def display_results_compare(results, titles):
    display_compare(results, 'val_acc', titles)
    display_compare(results, 'val_loss', titles)
    display_compare(results, 'trn_acc', titles)
    display_compare(results, 'trn_loss', titles)

    # display_compare_test(results, 'tst_acc', titles)
    # display_compare_test(results, 'tst_loss', titles)


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


def show_misclassified(model, device, loader, mean, std, save_path=''):
    
    targets, confidences, data = dataprocessing.get_misclassified(model, device, loader)
    top3_confidences = dataprocessing.get_top_confidences(confidences)
    target_confidence = dataprocessing.get_target_confidence(confidences, targets)

    top3strs = ["\n".join([f"  • class {target} - {confidence*100:.2f}%" for target, confidence in zip(*t3c)]) for t3c in top3_confidences]
    titles = [ f'target class: {t}, confidence {tc*100:.2f}%\ntop 3 predictions:\n{t3str}' for t, tc, t3str in zip(targets, target_confidence, top3strs)]

    denormalized_images = dataprocessing.denormalize_images(data.cpu(), mean, std)

    show_images_row(denormalized_images, 'misclassified test images (denormalized)', titles, save_path)


def show_images_row(images, title="", axtitles=None, save_path=''):
    fig, axs = plt.subplots(nrows=1, ncols=len(images), figsize=(4*len(images),4))
    axs = axs if isinstance(axs, np.ndarray) else [axs]

    y_offset = 0 if axtitles== None else axtitles[0].count('\n')*0.03 + title.count('\n')*0.05

    fig.suptitle(title, x=0.5, y=0.98 + y_offset)
    for idx, (ax, image) in enumerate(zip(axs,images)):
        im = np.transpose(image, (1,2,0))
        ax.imshow(im)
        if axtitles:
            ax.set_title(axtitles[idx],loc='left', fontsize='smaller')
        ax.axis('off')

    save_path!='' and fig.savefig(save_path+'.png', facecolor='white', dpi=fig.dpi, bbox_inches='tight', pad_inches=1)


def plor_bars(title, ylabel, xlabel, xdata, ydata, bar_labels):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    bars = ax.bar(xdata, ydata)
    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.bar_label(bars, bar_labels, fontsize='small')
    plt.grid()
