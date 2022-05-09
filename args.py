import argparse

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

parser = argparse.ArgumentParser()
parser.add_argument("-b","--batch_size", type=int, default=32, help="training batch size")
parser.add_argument("--optimizer", "--opt", type=str, default="Adadelta", help="optimizer")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("-m", "--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--epoch_num", "-e" , type=int, default=1, help="training epochs")
# parser.add_argument("--progress", "-p", type=str, default='tqdm', help="tqdm|log")
# parser.add_argument("--log_interval", '-li', type=str, default='epoch', help="epoch|num")
parser.add_argument("--wandb", action='store_true', help="log to wandb")
parser.add_argument("--model_name", type=str, default='Net', help="model class name")
parser.add_argument("--save_epochs", action='store_true', help="save each epoch")
parser.add_argument("--loss_fn",  type=str, help="loss funciton")


def print_args(args):
    print('[args: ' + ' '.join(f'{k}={v}' for k, v in vars(args).items())+']')

def get_args():
    args = parser.parse_args()
    args.wandb = args.wandb and wandb != None
    return args