import json 

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

def init_wandb(reinit=False, **kwargs):
    with open("wandb_config.json", "r") as f:
        wandb_config = json.load(f)

    return wandb.init(**wandb_config,config=kwargs, reinit=reinit)

def log_to_wandb(epoch, result):
    wandb.log({"training loss": result.trn_loss
              ,"validation loss": result.val_loss},
              step=epoch,
              commit=False)
    wandb.log({"training accuracy": result.trn_acc,
               "validation accuracy": result.val_acc},
               step=epoch)