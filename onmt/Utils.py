import torch
from torch.autograd import Variable
import numpy as np
import time

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


def formalize(batch, batch_length, batch_first=False):
    """formalize a batch to sort the batch according to its length

    Args:
        batch: batch
        batch_length: batch length list
    Returns:
        formalized batch
    """
    sorted_lengths, _ = torch.sort(batch_length, descending=True)
    batch_length = batch_length.view(-1).tolist()
    index_length = [(i, l) for i, l in enumerate(batch_length)]
    ordered_index = sorted(index_length, key=lambda e: e[1], reverse=True)

    origin_new = dict([(v[0], k) for k, v in enumerate(ordered_index)])

    sorted_batch = Variable(batch.data.new(batch.size()))
    for k, v in origin_new.items():
        if batch_first:
            sorted_batch[v] = batch[k]
        else:
            sorted_batch[:, v] = batch[:, k]
    return sorted_batch, sorted_lengths, origin_new


def deformalize(batch, origin_new):
    """reform batch in the origin order, batch is the second dimension.

    Args:
        batch: encoded batch, length*batch_size*dim
        origin_new: origin->new index dict
    Returns:
        reformed batch
    """
    desorted_batch = Variable(batch.data.new(batch.size()))
    for k, v in origin_new.items():
        desorted_batch[:, k] = batch[:, v]
    return desorted_batch


def reorder(batch, origin_new):
    """reform batch in the origin order, batch is the second dimension.

    Args:
        batch: encoded batch, length*batch_size*dim
        origin_new: origin->new index dict
    Returns:
        reformed batch
    """
    sorted_batch = Variable(batch.data.new(batch.size()))
    for k, v in origin_new.items():
        sorted_batch[:, v] = batch[:, k]
    return sorted_batch


def cat_history(src_context, src_lengths, rsp_context, rsp_lengths):
    ret = Variable(torch.zeros(src_context.size(0)+rsp_context.size(0), src_context.size(1), src_context.size(2)))
    ret = ret.cuda() if src_context.is_cuda else ret
    for i in range(src_context.size(1)):
        ret[:src_lengths[i], i] = src_context[:src_lengths[i], i]
        ret[src_lengths[i]: src_lengths[i]+rsp_lengths[i], i] = rsp_context[:rsp_lengths[i], i]
    lengths = src_lengths + rsp_lengths
    return ret[:torch.max(lengths)], lengths


def get_length(x):
    if len(x.size()) == 3:
        x = x.squeeze(2)
    x = x.t()
    mask = x.ne(1).long()
    return torch.sum(mask, dim=1).data


def truncated_gumbel(logits, truncation):
    gumbel = sample_gumbel(logits) + logits
    trunc_g = - torch.log(torch.exp(-gumbel) + torch.exp(-truncation))
    return trunc_g


def topdown(logits, k):
    """
    :param logits: batch_size * vocab_size
    :param k: batch_size
    :return:
    """
    topgumbel = sample_gumbel(logits[:, :, :1])
    gumbels = truncated_gumbel(logits, topgumbel)
    gumbels.scatter_(2, k.unsqueeze(2), topgumbel)
    gumbels -= logits
    return gumbels


def posterior_sample(logits, obs):
    """
    :param logits: batch_size * vocab_size
    :param obs: batch_size
    :return:
    """
    u = topdown(logits, obs)
    return u


def sample_gumbel(logits, eps=1e-10):
    u = Variable(logits.data.new(*logits.size()).uniform_())
    gumbel = -torch.log(-torch.log(u + eps) + eps)
    return gumbel
