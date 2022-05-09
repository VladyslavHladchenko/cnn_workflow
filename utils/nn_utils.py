import torch.nn as nn 

def get_conv_out_size_params(W,K,P,S):
    return ((W-K+2*P)/S)+1

def get_conv_out_size_seq(in_dim, sequential):
    for m in sequential:
        if type(m) == nn.Conv2d:
            in_dim, last_conv_out = get_conv_out_size_layer(in_dim, m)
        elif type(m) == nn.MaxPool2d:
            in_dim, _ = get_conv_out_size_layer(in_dim, m)
        elif type(m) == nn.Sequential:
            in_dim, last_conv_out = get_conv_out_size_seq(in_dim, m)
    return in_dim, last_conv_out


def get_conv_out_size_layer(in_dim, l):
    K = l.kernel_size if type(l.kernel_size) == int else l.kernel_size[0]
    P = l.padding if type(l.padding) == int else l.padding[0]
    S = l.stride if type(l.stride) == int else l.stride[0]

    out_ch = getattr(l, 'out_channels', None)
    return get_conv_out_size_params(in_dim, K, P, S), out_ch


def get_linear_in_size(in_dim, layers):
    featres_out_dim,last_kernel_depth = get_conv_out_size_seq(in_dim, layers)
    linear_in = int(featres_out_dim*featres_out_dim*last_kernel_depth)

    return linear_in
