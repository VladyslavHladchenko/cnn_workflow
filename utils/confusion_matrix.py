import numpy as np 
import torch
import matplotlib.pyplot as plt


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
