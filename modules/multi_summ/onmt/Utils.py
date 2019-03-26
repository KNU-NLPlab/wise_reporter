import torch
from torch.autograd import Variable

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)

def pad(tensor, length, pad_index=1):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat([var,
                Variable(pad_index * torch.ones(length - var.size(0), *var.size()[1:])).cuda().type_as(var)])
        else:
            return var
    else:
        if length > tensor.size(0):
            return torch.cat([tensor,
                                  pad_index * torch.ones(length - tensor.size(0), *tensor.size()[1:]).cuda()])
        else:
            return tensor      