# original version
import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from util import get_buckets, DataIterator
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F

from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, AdamW

while (True):
    try:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        break
    except:
        print('reload tokenizer.')

import apex
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel as DDP

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# device = torch.device('cpu')

class ParaModel(nn.Module):
    def __init__(self, roberta):
        super().__init__()
        hidden = roberta.config.hidden_size
        self.roberta = roberta
        self.layer = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.pool = nn.AdaptiveMaxPool2d((1, hidden))

    def forward(self, input_tensor):
        bert_mask = torch.where(input_tensor == 1, torch.full_like(input_tensor, 0),
                                torch.full_like(input_tensor, 1)).half()
        last_layer = self.roberta(input_tensor, attention_mask=bert_mask)[0]
        tmp_a = self.pool(last_layer[:, 1:]).squeeze(1)
        return self.layer(torch.cat([tmp_a, last_layer[:, 0]], -1))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


nll = nn.CrossEntropyLoss()


def train_para(config):
    # parallel training
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    # random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # backup and logging
    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=[os.path.basename(__file__), 'main.py', 'util.py'])

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    logging("Building model...")
    train_buckets = get_buckets(config.train_record_file, config.local_rank, ddp=distributed)
    dev_buckets = get_buckets(config.dev_record_file, config.local_rank)

    def build_train_iterator():
        return DataIterator(train_buckets, config.batch_size, True, mode='train')

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, False, mode='eval')

    print('loading roberta-large...')
    roberta = RobertaModel.from_pretrained("roberta-large")
    para_model = ParaModel(roberta)
    para_model.to(device)

    logging('nparams {}'.format(sum([p.nelement() for p in roberta.parameters() if p.requires_grad])))

    optimizer = AdamW(para_model.parameters(), lr=config.init_lr, betas=(0.9, 0.999), weight_decay=0.01)
    para_model, optimizer = amp.initialize(para_model, optimizer, opt_level='O2')

    if distributed:
        para_model = DDP(para_model, delay_allreduce=True)

    print(optimizer)

    total_loss = 0
    global_step = 0
    best_dev_ACC = None
    stop_train = False
    lr = config.init_lr
    start_time = time.time()
    eval_start_time = time.time()
    roberta.train()

    for epoch in range(10000):
        for data in build_train_iterator():
            train_para = data['train_para'].to(device)
            target_para = data['target_para'].to(device)

            predict_paras = para_model(train_para)
            loss = nll(predict_paras, target_para)

            optimizer.zero_grad()

            warm_up_step = config.warm_up_step
            decay_lr = config.decay_lr
            eval_last_rate = config.eval_last_rate
            # learning rate warm up
            if global_step < warm_up_step:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.1 * lr + (0.9 * lr / warm_up_step) * global_step
            elif global_step == warm_up_step:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.init_lr
            # learning rate decay
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
            if global_step > warm_up_step:
                cur_lr -= decay_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr
            if cur_lr <= decay_lr and global_step > warm_up_step:
                stop_train = True
                break

            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(para_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epoch {:3d} | step {:6d} | lr {:05.10f} | ms/batch {:5.2f} | train loss {:8.3f}' \
                        .format(epoch, global_step, cur_lr, elapsed * 1000 / config.period, cur_loss))
                total_loss = 0
                start_time = time.time()

            if cur_lr < config.init_lr * eval_last_rate and global_step % config.checkpoint == 0:
                para_model.eval()
                metrics = evaluate_batch(build_dev_iterator(), para_model, 0, dev_eval_file, config)
                para_model.train()

                logging('-' * 89)
                logging('| eval {:6d} in epoch {:3d} | time: {:5.2f}s | acc {:.4f}' \
                        .format(global_step // config.checkpoint, epoch, time.time() - eval_start_time, metrics['acc']))
                logging('-' * 89)

                eval_start_time = time.time()

                dev_ACC = metrics['acc']
                if best_dev_ACC is None or dev_ACC > best_dev_ACC:
                    best_dev_ACC = dev_ACC
                    torch.save(para_model.state_dict(),
                               os.path.join(config.save, 'para_model' + str(config.local_rank) + '.pt'))
        if stop_train: break
    logging('best_dev_acc {}'.format(best_dev_ACC))


def evaluate_batch(data_source, para_model, max_batches, eval_file, config):
    right, count = 0, 0
    for step, data in enumerate(data_source):
        if step >= max_batches and max_batches > 0: break

        eval_para = data['eval_para'].to(device)
        eval_para_target = data['eval_para_target']
        para_num_record = data['para_num_eval']

        with torch.no_grad():
            predict_paras = para_model(eval_para)
            predict_paras = F.softmax(predict_paras, dim=-1)[:, 1]

        # count_ans accuracy
        tmp = 0
        for ori_b in range(len(para_num_record)):
            tmp_target = eval_para_target[tmp:para_num_record[ori_b]]
            tmp_predict = predict_paras[tmp:para_num_record[ori_b]]
            _, predict_idx = tmp_predict.topk(2)
            if tmp_target[predict_idx[0]] == 1:
                right += 1
            count += 1
            if tmp_target[predict_idx[1]] == 1:
                right += 1
            count += 1

    metrics = {}
    metrics['acc'] = right / count

    return metrics


def test_para(config):
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    test_buckets = get_buckets(config.dev_record_file)

    def build_test_iterator():
        return DataIterator(test_buckets, config.batch_size, False, mode='eval')

    print('loading roberta-large...')
    roberta = RobertaModel.from_pretrained("roberta-large")
    para_model = ParaModel(roberta)
    para_model.cuda()

    para_model.load_state_dict({k.replace('module.', ''): v for k, v in
                                torch.load(os.path.join(config.save, 'para_model2.pt'), map_location='cuda:0').items()})
    para_model.eval()

    precision, recall, em = predict(build_test_iterator(), para_model, dev_eval_file, config)
    logging('predict_precision {}'.format(precision))
    logging('predict_recall {}'.format(recall))
    logging('exact_match {}'.format(em))


def predict(data_source, para_model, eval_file, config):
    def logging_prediction(s):
        with open('config.save/predict_para.txt', 'a+') as f_log:
            # with open(os.path.join(config.save, 'predict_para.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

    prec, count_prec, recall, count_recall, em = 0, 0, 0, 0, 0
    for step, data in enumerate(tqdm(data_source)):
        ids = data['ids']
        eval_para = data['eval_para'].to(device)
        eval_para_target = data['eval_para_target']
        para_num_record = data['para_num_eval']

        with torch.no_grad():
            predict_paras = para_model(eval_para)
            predict_paras = F.softmax(predict_paras, dim=-1)[:, 1]

        # count_ans precision and recall
        tmp = 0
        for ori_b in range(len(para_num_record)):
            tmp_target = eval_para_target[tmp:para_num_record[ori_b]]
            tmp_predict = predict_paras[tmp:para_num_record[ori_b]]
            try:
                _, predict_idx = tmp_predict.topk(3)
                logging_prediction(
                    ids[ori_b] + ' ' + str(predict_idx[0].item()) + ' ' + str(predict_idx[1].item()) + ' ' + str(
                        predict_idx[2].item()))
            except RuntimeError:
                _, predict_idx = tmp_predict.topk(2)
                logging_prediction(ids[ori_b] + ' ' + str(predict_idx[0].item()) + ' ' + str(predict_idx[1].item()))
            each_prec = 0
            for pred in predict_idx:

                if tmp_target[pred] == 1:
                    prec += 1
                    each_prec += 1
                if each_prec == 2:
                    em += 1
                count_prec += 1
            for i, tgt in enumerate(tmp_target):
                if tgt == 0: continue
                if i in predict_idx:
                    recall += 1
                count_recall += 1
        print("em = ",em)
    return prec / count_prec, recall / count_recall, em / 7405
