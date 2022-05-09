def make_tqdm_postfix(result):
    s= f"tl {result.trn_loss:.4f} vl {result.val_loss:.4f} ta {result.trn_acc:.4f} va {result.val_acc:.4f}"
    return s

def set_tqdm_postfix(t, result):
    t.set_postfix_str(make_tqdm_postfix(result))