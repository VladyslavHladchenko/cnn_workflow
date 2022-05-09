class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    """ For a more elaborate solution take a look at the EasyDict package https://pypi.org/project/easydict/ """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # these are needed for deepcopy / pickle
    def __getstate__(self): return self.__dict__

    def __setstate__(self, d): self.__dict__.update(d)