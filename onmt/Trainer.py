from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import glob
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.modules

from onmt.Utils import posterior_sample
from torch.autograd import Variable
import torch.nn.functional as F


def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        yield lazy_dataset_loader(pt, corpus_type)


def make_dataset_iter(datasets, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_size_fn = None
    if is_train and opt.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            return sofar + max(len(new.tgt), len(new.src)) + 1

    device = opt.gpuid[0] if opt.gpuid else -1

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


class DatasetLazyIter(object):
    """ An Ordered Dataset Iterator, supporting multiple datasets,
        and lazy loading.

    Args:
        datsets (list): a list of datasets, which are lazily loaded.
        fields (dict): fields dict for the datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: the GPU device.
        is_train (bool): train or valid?
    """
    def __init__(self, datasets, fields, batch_size, batch_size_fn,
                 device, is_train):
        self.datasets = datasets
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

        self.cur_iter = self._next_dataset_iterator(datasets)
        # We have at least one dataset.
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def __len__(self):
        # We return the len of cur_dataset, otherwise we need to load
        # all datasets to determine the real len, which loses the benefit
        # of lazy loading.
        assert self.cur_iter is not None
        return len(self.cur_iter)

    def get_cur_dataset(self):
        return self.cur_dataset

    def _next_dataset_iterator(self, dataset_iter):
        try:
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        # We clear `fields` when saving, restore when loading.
        self.cur_dataset.fields = self.fields

        # Sort batch by decreasing lengths of sentence required by pytorch.
        # sort=False means "Use dataset's sortkey instead of iterator's".
        return onmt.io.OrderedIterator(
                dataset=self.cur_dataset, batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device, train=self.is_train,
                sort=False, sort_within_batch=True,
                repeat=False)


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=1e-10, n_correct=1e-10, d1=0, d2=0, step_type='teacher_force'):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()
        self.d1 = d1
        self.d2 = d2
        self.step_type = step_type
        self.num_batchs = 0

    def update(self, stat, mode='train'):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.step_type = stat.step_type

        if mode == 'valid':
            self.d1 += math.exp(-stat.d1)
            self.d2 += math.exp(-stat.d2)
            self.num_batchs += 1
        else:
            self.d1 = math.exp(-stat.d1)
            self.d2 = math.exp(-stat.d2)
            self.num_batchs = 1

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def d_acc(self):
        return (self.d1 / self.num_batchs + self.d2 / self.num_batchs) / 2

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        if self.step_type == 'teacher_force' or self.step_type == 'self_sample':
            print(("Epoch %2d, %s %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
                   "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                  (epoch, self.step_type, batch,  n_batches,
                   self.accuracy(),
                   self.ppl(),
                   self.n_src_words / (t + 1e-5),
                   self.n_words / (t + 1e-5),
                   time.time() - start))
        elif self.step_type == 'd_step':
            print("Epoch %2d, %s %5d/%5d; d: %.5f; %6.0f s elapsed" %
                  (epoch, self.step_type, batch, n_batches, self.d_acc(), time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """

    def __init__(self, opt, fields, model, disc, env,
                 train_loss, valid_loss, g_optim, d_optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 normalization="sents", accum_count=1):
        # Basic attributes.
        self.opt = opt
        self.fields = fields
        self.model = model
        self.disc = disc
        self.env = env
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.accum_count = accum_count
        self.padding_idx = self.train_loss.padding_idx
        self.eos_idx = self.train_loss.tgt_vocab.stoi[onmt.io.EOS_WORD]
        self.bos_idx = self.train_loss.tgt_vocab.stoi[onmt.io.BOS_WORD]
        self.normalization = normalization
        assert(accum_count > 0)
        if accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def load_data(self):
        train_datasets = lazily_load_dataset("train", self.opt)
        train_iter = make_dataset_iter(train_datasets, self.fields, self.opt)
        return train_iter

    def train(self, epoch, report_func=None, training_mode='adv'):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging
            training_mode: training mode

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        report_freq = 5
        idx = 0
        truebatch = []
        accum = 0
        normalization = 0
        g_iter = self.load_data()
        d_iter = iter(self.load_data())

        try:
            add_on = 0
            if len(g_iter) % self.accum_count > 0:
                add_on += 1
            num_batches = len(g_iter) / self.accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        def gradient_accumulation(truebatch_, total_stats_,
                                  report_stats_, nt_, step_type_):
            if self.accum_count > 1:
                self.model.zero_grad()
                self.disc.zero_grad()

            for batch in truebatch_:
                target_size = batch.tgt.size(0)
                # Truncated BPTT
                if self.trunc_size:
                    trunc_size = self.trunc_size
                else:
                    trunc_size = target_size

                dec_state = None
                src = onmt.io.make_features(batch, 'src', self.data_type)
                if self.data_type == 'text':
                    _, src_lengths = batch.src
                    report_stats.n_src_words += src_lengths.sum()
                else:
                    src_lengths = None

                tgt_outer = onmt.io.make_features(batch, 'tgt')

                for j in range(0, target_size-1, trunc_size):
                    # 1. Create truncated target.
                    tgt = tgt_outer[j: j + trunc_size]
                    tgt_lengths = self.get_length(tgt)

                    # 2. F-prop all but generator.
                    if self.accum_count == 1:
                        self.model.zero_grad()
                        self.disc.zero_grad()

                    outputs, attns, fak_tok, fak_outputs, d1, d2 = None, None, None, None, None, None

                    if step_type_ == 'self_sample':
                        obs_outputs, _, _, _ = self.env(src, tgt, src_lengths)
                        assert obs_outputs.size(0) == tgt.size(0) - 1
                        obs_logits = self.env.generator[1](self.env.generator[0](obs_outputs) / 0.5)
                        u = posterior_sample(obs_logits, tgt.squeeze(2)[1:])

                        enc_hidden, context = self.model.encoder(src, src_lengths)
                        enc_state = self.model.decoder.init_decoder_state(src, context, enc_hidden)

                        fak_tok, fak_outputs = self.model.reason(tgt.size(0)-1, enc_state,
                                                                 context, src_lengths, self.bos_idx, u)
                        fak_tok_lengths = self.get_length(fak_tok)
                        fak_tok = fak_tok[: torch.max(fak_tok_lengths)]  # remove extra padding symbols in tail
                        fak_outputs, _, _, _ = self.model(src, fak_tok, src_lengths)

                        d2 = self.disc(fak_tok, fak_tok_lengths, src, src_lengths, step_type='self_sample')[:, :, :1]

                    if step_type_ == 'teacher_force':
                        enc_hidden, context = self.model.encoder(src, src_lengths)
                        enc_state = self.model.decoder.init_decoder_state(src, context, enc_hidden)

                        outputs, dec_state, attns = self.model.decoder(tgt[:-1], context,
                                                                       enc_state if dec_state is None
                                                                       else dec_state,
                                                                       context_lengths=src_lengths)

                        d1 = Variable(torch.ones(tgt[1:].size()), requires_grad=False)
                        if tgt.is_cuda:
                            d1 = d1.cuda()

                    if step_type_ == 'd_step':
                        enc_hidden, context = self.model.encoder(src, src_lengths)
                        enc_state = self.model.decoder.init_decoder_state(src, context, enc_hidden)

                        fak_tok, _ = self.model.infer(tgt.size(0)-1, enc_state,
                                                      context, src_lengths, self.bos_idx, sample=True)
                        fak_tok_lengths = self.get_length(fak_tok)  # exclude start symbol
                        fak_tok = fak_tok[: torch.max(fak_tok_lengths)]  # remove extra padding symbols in tail

                        d2 = self.disc(fak_tok, fak_tok_lengths, src, src_lengths)
                        d1 = self.disc(tgt, tgt_lengths, src, src_lengths)

                    # 3. Compute loss in shards for memory efficiency.
                    batch_stats = self.train_loss.sharded_compute_loss(
                        batch, fak_tok, outputs, fak_outputs,
                        d1, d2, attns, j, trunc_size, self.shard_size, nt_, step_type_)

                    # 4. Update the parameters and statistics.
                    if self.accum_count == 1:
                        if step_type_ == 'd_step':
                            self.d_optim.step()
                        else:
                            self.g_optim.step()
                    total_stats_.update(batch_stats)
                    report_stats_.update(batch_stats)

                    # If truncated, don't backprop fully.
                    if dec_state is not None:
                        dec_state.detach()

            if self.accum_count > 1:
                if step_type_ == 'd_step':
                    self.d_optim.step()
                else:
                    self.g_optim.step()

        for i, batch_ in enumerate(g_iter):
            cur_dataset = g_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            truebatch.append(batch_)
            accum += 1
            if self.normalization is "tokens":
                normalization += batch_.tgt[1:].data.view(-1) \
                                       .ne(self.padding_idx)
            else:
                normalization += batch_.batch_size

            # accumulate d batch from d_iter
            D_SIZE = 5
            d_batch = []
            while len(d_batch) < D_SIZE and training_mode == 'adv':
                try:
                    d_batch.append(next(d_iter))
                except StopIteration:
                    d_iter = iter(self.load_data())

            if accum == self.accum_count:
                if training_mode == 'pre_d':
                    gradient_accumulation(truebatch, total_stats, report_stats, None, 'd_step')
                elif training_mode == 'pre_g':
                    gradient_accumulation(truebatch, total_stats, report_stats, normalization, 'teacher_force')
                elif training_mode == 'adv':
                    gradient_accumulation(truebatch, total_stats, report_stats, normalization, 'self_sample')
                    gradient_accumulation(d_batch, total_stats, report_stats, None, 'd_step')
                else:
                    raise NameError

                if report_func is not None:
                    if i % (2 * report_freq - 1) == 0:
                        report_flag = True
                    else:
                        report_flag = False

                    report_stats = report_func(
                        epoch, idx, num_batches,
                        total_stats.start_time, self.g_optim.lr,
                        report_stats, report_flag)

                truebatch = []
                d_batch = []
                accum = 0
                normalization = 0
                idx += 1

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        num_total = 0.0
        num_correct = 0.0

        pos_num_total = 0.0
        pos_num_correct = 0.0

        neg_num_total = 0.0
        neg_num_correct = 0.0

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')
            tgt_lengths = self.get_length(tgt)

            # F-prop through the model.
            outputs, attns, _, _ = self.model(src, tgt, src_lengths)

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

            enc_hidden, context = self.model.encoder(src, src_lengths)
            enc_state = self.model.decoder.init_decoder_state(src, context, enc_hidden)

            fak_tok, _ = self.model.infer(tgt.size(0) - 1, enc_state, context,
                                        src_lengths, self.bos_idx, sample=True)
            fak_tok_lengths = self.get_length(fak_tok)  # exclude start symbol
            fak_tok = fak_tok[: torch.max(fak_tok_lengths)]  # remove extra padding symbols in tail

            d2 = self.disc(fak_tok, fak_tok_lengths, src, src_lengths)
            d2_pred = d2
            for score in d2_pred:
                if score[0].data[0] < 0.5:
                    num_correct += 1
                    neg_num_correct += 1
                num_total += 1
                neg_num_total += 1

            d1 = self.disc(tgt, tgt_lengths, src, src_lengths)
            d1_pred = d1
            for score in d1_pred:
                if score[0].data[0] >= 0.5:
                    num_correct += 1
                    pos_num_correct += 1
                num_total += 1
                pos_num_total += 1

        print('acc: {}, number correct: {}, number total: {}'.format(num_correct/num_total, num_correct, num_total))
        print('pos acc: {}, number correct: {}, number total: {}'.format(pos_num_correct / pos_num_total, pos_num_correct, pos_num_total))
        print('neg acc: {}, number correct: {}, number total: {}'.format(neg_num_correct / neg_num_total, neg_num_correct, neg_num_total))
        # Set model back to training mode.
        self.model.train()

        return stats, num_correct/num_total

    def epoch_step(self, ppl, epoch, training_mode='adv'):
        if (training_mode == 'pre_g' and epoch < 7) \
                or (training_mode == 'pre_d' and epoch < 25)\
                or (training_mode == 'adv' and epoch < 23):
            return

        self.d_optim.update_learning_rate(ppl, epoch)
        return self.g_optim.update_learning_rate(ppl, epoch)

    def sync(self):
        model_state_dict = self.model.state_dict()
        self.env.load_state_dict(model_state_dict)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats, d_acc):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()

        env_state_dict = self.env.state_dict()
        env_state_dict = {k: v for k, v in env_state_dict.items()
                          if 'generator' not in k}
        env_generator_state_dict = self.env.generator.state_dict()
        disc_state_dict = self.disc.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'env': env_state_dict,
            'env_generator': env_generator_state_dict,
            'disc': disc_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'g_optim': self.g_optim,
            'd_optim': self.d_optim
        }
        torch.save(checkpoint,
                   '%s_adv_ppl_%.2f_d_%4f_e%d.pt'
                   % (opt.save_model, valid_stats.ppl(), d_acc, epoch))

    def get_length(self, x):
        x = x.squeeze(2).t()
        mask = x.ne(self.padding_idx).long()
        return torch.sum(mask, dim=1).data
