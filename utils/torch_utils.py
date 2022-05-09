import torch 
from .gpu_utils import get_free_gpus


def get_free_device():
    device = None

    if torch.cuda.is_available():
        free_gpu_ids= get_free_gpus()
        print(f'free GPUs: {free_gpu_ids}')

        if free_gpu_ids:
            device = torch.device(f'cuda:{free_gpu_ids[-1]}')

    if not device:
        device = torch.device('cpu')

    print(f'using device: {device}')
    return device

def get_optimizer(opt_class_name, *opt_args, **opt_kwargs):
    opt_class = getattr(torch.optim, opt_class_name)
    opt = opt_class(*opt_args, **opt_kwargs)

    return opt

# def test_batch_sizes_and_num_workers(td, device):
#     from time import perf_counter
#     bs_times = {}
#     for bs in tqdm([4,8,16,32,64]):
#         times = {}
#         for nw in range(1,20):

#             train_loader_normalized = torch.utils.data.DataLoader(td, batch_size=bs, sampler=torch.utils.data.SubsetRandomSampler(train_indices), num_workers = nw, pin_memory=True)
#             val_loader_normalized = torch.utils.data.DataLoader(td, batch_size=bs, sampler=torch.utils.data.SubsetRandomSampler(val_indices),num_workers = nw, pin_memory=True) # pinned memory 
#             data_loader2 = edict({'train_loader': train_loader_normalized, 'val_loader':val_loader_normalized})

#             model  = get_vgg('vgg11', True).to(device)
#             opt = optim.Adadelta(model.parameters(), lr=0.1)
#             t1 = perf_counter()
#             cnn_workflow.train(model, device, data_loader2, opt, epoch_num=1, save_epochs=False, loss_fn=F.cross_entropy, disable_tqdm=True)
#             t2 = perf_counter()
#             times[nw] = t2-t1
#         bs_times[bs] = times


# def plot_batch_sizes_and_num_workers(data):
#     for k in data:
#         xs = list(data[k].keys())
#         ys = list(data[k].values())
#         plt.plot(xs,ys)
#         plt.xticks(xs)
#     plt.legend(list(data.keys()))