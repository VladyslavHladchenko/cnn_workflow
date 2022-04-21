from cnn_workflow import *
from utils import *
from DataLoader import DataLoader
from args import print_args

def main(args):
    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu') #TODO: separate function
    print('Using device:', device)

    data_loader = DataLoader(batch_size = args.batch_size, val_fraction = 0.1)
    model = load_model(args.model_name, device)
    opt = get_optimizer(args.optimizer, model.parameters(), lr=args.lr)

    train(model, device, data_loader, opt, **vars(args))

    l, acc = test(model, device, data_loader.test_loader)

    # if args.wandb:
    #     wandb.log({"test loss": l, "test accuracy": acc}, step=1)


if __name__ == "__main__":
    args = get_args()
    print_pid()
    print_args(args)

    if args.wandb:
        init_wandb(**vars(args))

    main(args)
