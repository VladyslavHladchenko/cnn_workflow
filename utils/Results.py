from collections import OrderedDict
from . import dotdict
from dataclasses import dataclass

class Results(dotdict):
    def __init__(self):
        self.trn_loss = OrderedDict()
        self.val_loss = OrderedDict()
        self.trn_acc = OrderedDict()
        self.val_acc = OrderedDict()

    def save_epoch(self, epoch, result):
        self.trn_loss[epoch] = result.trn_loss
        self.val_loss[epoch] = result.val_loss
        self.trn_acc[epoch] = result.trn_acc
        self.val_acc[epoch] = result.val_acc

    def at(self, epoch):
        return Result(trn_loss = self.trn_loss[epoch],
                      val_loss = self.val_loss[epoch],
                      trn_acc = self.trn_acc[epoch],
                      val_acc = self.val_acc[epoch])

    def __iter__(self):
        self.iter_position = 0
        return self

    def __next__(self):
        if self.iter_position < len(self.trn_loss):
            self.iter_position += 1
            return self.at(self.iter_position)
        else:
            raise StopIteration
    

@dataclass(frozen=True, order=True)
class Result:
    trn_loss: int = 0
    val_loss: int = 0
    trn_acc: int = 0
    val_acc: int = 0

    def __add__(self, o):
        assert type(o) == type(self)
        return Result(**{k: self.__dict__[k] + o.__dict__[k] for k in self.__dict__ })
    
    def __truediv__(self,o):
        return Result(**{k: self.__dict__[k]/o for k in self.__dict__ })

    @staticmethod
    def pstr(param:str):
        '''
        short param name to full name
        '''
        return ('val' in param)*'validation' + \
               ('trn' in param)*'training' + ' ' + \
               ('loss' in param)*'loss' + \
               ('acc' in param)*'accuracy'

