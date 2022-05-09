import torch 
import torch.nn.functional as F 

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

def get_misclassified(model, device, loader):
    """
    get target class, confidences and data of misclassified data from loader
    """
    model.eval()

    confidence_list = []
    target_list = []
    data_list = []

    with torch.no_grad():
        for b_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            wrong_idx = torch.where(predicted != target)

            if len(wrong_idx) == 0 :
                continue

            confidence_list.append(probs[wrong_idx])
            target_list.append(target[wrong_idx])
            data_list.append(data[wrong_idx])

    return torch.concat(target_list), torch.concat(confidence_list), torch.concat(data_list)

def get_top_confidences(confidences, top=3):
    """
    return list of tuples (classes, confidence), data in each tuple is sorted by confidence
    """
    sorted, sort_idx = torch.sort(confidences, descending=True)
    return [(target[:top], confidence[:top]) for target, confidence in zip(sort_idx, sorted)]

def get_target_confidence(confidences, targets):
    tc = confidences.gather(1,targets[None,...].T)
    return tc.squeeze(1)

def mean_std(loader):
    """
    calculate mean and std for each channel across all batches and pixels
    """
    n_data = len(loader.dataset)
    num_pixels = loader.dataset[0][0].shape[1]*loader.dataset[0][0].shape[2]*n_data

    sum = torch.zeros(3)
    for data, _ in loader:
        sum+=data.sum(dim=[0,2,3])
    mean = sum/num_pixels

    sum_of_squared_error = torch.zeros(3)
    for data, _ in loader: 
        sum_of_squared_error += ((data - mean.reshape(3,1,1))**2).sum(dim=[0,2,3])
    std = torch.sqrt(sum_of_squared_error / num_pixels)

    return mean, std

def denormalize_images(images, mean, std):
    if isinstance(images,list):
        return [img*std.reshape([3,1,1]).numpy() + mean.reshape([3,1,1]).numpy() for img in images]
    if isinstance(images,torch.Tensor):
        return images*std.reshape([3,1,1]) + mean.reshape([3,1,1])

def shitfndivide(data):
    """
        output data will be in interval [0..1] per channel for each image in batch
    """
    b = data.shape[0]

    mins = data.min(dim=2)[0].min(dim=2)[0]
    data2 = data - mins.reshape([b,3,1,1])

    maxs = data2.max(dim=2)[0].max(dim=2)[0]
    data3 = data2/maxs.reshape([b,3,1,1])

    return data3

