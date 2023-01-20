# Multi hop attention +
# span对比学习 有效
# sup 对比学习
import ujson as json
import numpy as np
from tqdm import tqdm
import os
from torch import optim, nn
from util import convert_tokens, evaluate
from util import get_buckets, DataIterator, IGNORE_INDEX
import time
import shutil
import random
import torch
from torch.autograd import Variable
import sys
from torch.nn import functional as F
import math
from torch.utils.checkpoint import checkpoint_sequential
from torch.utils.data.distributed import DistributedSampler as DS
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, AdamW
from hotpot_evaluate_v1 import eval as eval_fun

torch.set_printoptions(threshold=np.inf)
while (True):
    try:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        break
    except:
        print('reload tokenizer.')
        continue

import apex
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel as DDP

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class GAT_layer(nn.Module):
    def __init__(self, hidden, nheads, alpha, num_hop):
        super(GAT_layer, self).__init__()
        self.hidden = hidden
        self.nheads = nheads
        self.alpha = alpha
        self.num_hop = num_hop
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.LN = nn.LayerNorm(hidden)

        # self.emb = nn.Embedding(11, hidden)

    def forward(self, input, adj, node_to_up_mask=None, weight=None):

        q = self.query(input)
        k = self.key(input)
        v = self.value(input)
        h_prime = torch.Tensor().type_as(v).to(device)

        # multi-head attention
        u = self.hidden // self.nheads
        # for each head
        for m in range(self.nheads):  # m=0
            ## NAACL paper version:
            # attention for weights
            weights = torch.bmm(q[:, :, m * u:m * u + u], k[:, :, m * u:m * u + u].transpose(-1, -2))  # bsz * l * l

            # combine the pre-defined weights
            if weight is not None:
                weights += torch.log(torch.sigmoid(weight))

            ## extension version:
            # weights = mybmm(q[:, :, m*u:m*u+u], k[:, :, m*u:m*u+u], self.emb(weight)[:, :, :, m*u:m*u+u])

            weights = weights.masked_fill(mask=(1 - adj).bool(), value=torch.tensor(-16384))

            pros = torch.softmax(weights, dim=-1)

            # h_prime = torch.cat([h_prime, torch.matmul(pros, v[:, :, m * u:m * u + u])], -1)

            # # 2.multihop embedding(近似算法)
            z_init = v[:, :, m * u:m * u + u]
            z_iter = z_init
            for _ in range(self.num_hop):
                z_iter = (1 - self.alpha) * torch.matmul(pros, z_iter) + self.alpha * z_init
            h_prime = torch.cat([h_prime, z_iter], -1)

        # only some nodes are allowed to be updated
        if node_to_up_mask is not None:
            tmp_mask = node_to_up_mask.unsqueeze(-1).repeat(1, 1, input.size(-1))  # 把要更新的行的每个元素置1
            h_prime = h_prime.masked_fill(mask=(1 - tmp_mask).bool(), value=torch.tensor(0))

        # residual connection + layer normalization
        return self.LN(input + h_prime)


class BiAttention(nn.Module):
    def __init__(self, hidden):
        super(BiAttention, self).__init__()
        self.input_linear_1 = nn.Linear(hidden, 1, bias=False)
        self.memory_linear_1 = nn.Linear(hidden, 1, bias=False)

        self.input_linear_2 = nn.Linear(hidden, hidden, bias=True)
        self.memory_linear_2 = nn.Linear(hidden, hidden, bias=True)

        self.dot_scale = np.sqrt(hidden)
        self.output = nn.Linear(hidden * 4, hidden)

    def forward(self, input, memory):
        """
        :param input: context_encoding N * Ld * d
        :param memory: query_encoding N * Lm * d
        :param mask: query_mask N * Lm
        :return:
        """
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)
        input_dot = self.input_linear_1(input)  # N x Ld x 1
        memory_dot = self.memory_linear_1(memory).view(bsz, 1, memory_len)  # N x 1 x Lm
        # N * Ld * Lm
        cross_dot = torch.bmm(input, memory.permute(0, 2, 1).contiguous()) / self.dot_scale
        # [metric, f2]^T [w1, w2] + <metric * w3, f2>
        # (N * Ld * 1) + (N * 1 * Lm) + (N * Ld * Lm)
        att = input_dot + memory_dot + cross_dot  # N x Ld x Lm
        # N * Ld * Lm

        input = self.input_linear_2(input)
        memory = self.memory_linear_2(memory)

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return self.output(torch.cat([input, output_one, input * output_one, output_two * output_one], dim=-1))


class model(nn.Module):
    def __init__(self, roberta, config):
        super().__init__()
        hidden = roberta.config.hidden_size
        self.hidden = hidden
        self.roberta = roberta
        self.biatten = BiAttention(hidden)

        self.predict_span = nn.Sequential(
            nn.Linear(hidden, hidden),
            # nn.Dropout(0.5),
            nn.Tanh(),
            # nn.Dropout(0.5),
            nn.Linear(hidden, 2),
        )

        # type预测层
        self.typelayer1a = nn.Linear(hidden, hidden)
        self.typelayer1b = nn.Linear(hidden, hidden)
        self.typelayer2 = nn.Linear(hidden * 2, 3)

        expand_hidden = hidden * 1

        self.expand_hidden = expand_hidden
        self.expand_sent = nn.Linear(hidden, expand_hidden)
        self.word_weight = nn.Linear(expand_hidden, 1)

        self.GAT = [GAT_layer(expand_hidden, nheads=1, alpha=config.alpha, num_hop=config.num_hop) for _ in range(4)]
        for i, gat_layer in enumerate(self.GAT):
            self.add_module('GAT_{}'.format(i), gat_layer)

        self.sp_attn = GAT_layer(expand_hidden, nheads=1, alpha=config.alpha, num_hop=config.num_hop)
        self.sp1 = nn.Linear(expand_hidden, expand_hidden)
        self.sp2 = nn.Linear(expand_hidden, 2)

        # para打分
        self.para_score1 = nn.Linear(expand_hidden, expand_hidden)
        self.para_score2 = nn.Linear(expand_hidden, 1)
        self.sent_score = nn.Sequential(
            nn.Linear(expand_hidden, expand_hidden),
            nn.Tanh(),
            nn.Linear(expand_hidden, 1),
        )

        # 池化
        self.type_pool = nn.AdaptiveMaxPool2d((1, expand_hidden))
        self.para_pool = nn.AdaptiveMaxPool2d((1, expand_hidden))

        self.LN = nn.LayerNorm(expand_hidden)

        # self.sigma = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1, 0]))
        self.prj1 = nn.Sequential(nn.Linear(hidden, hidden),
                                  nn.Tanh(),
                                  nn.Linear(hidden, hidden))
        self.prj2 = nn.Sequential(nn.Linear(hidden, hidden),
                                  nn.Tanh(),
                                  nn.Linear(hidden, hidden))

    def forward(self, input_tensor, mask, sent_mapping, sp_mask, para_mapping, dyn_mask, node_to_up_mask, graph_weight):
        # roebrta
        bert_mask = torch.where(input_tensor == 1, torch.full_like(input_tensor, 0),
                                torch.full_like(input_tensor, 1)).half()
        last_layer_ = self.roberta(input_tensor, attention_mask=bert_mask)[0]

        # bi-attention
        last_layer = torch.zeros_like(last_layer_).type_as(last_layer_).to(device)
        for b in range(last_layer_.size(0)):
            q_end = np.where(input_tensor.cpu()[b] == 2)[0][0] if 2 in input_tensor[b] else 500
            if q_end > 500: q_end = 500
            para_end = np.where(input_tensor.cpu()[b] == 1)[0][0] - 1 if 1 in input_tensor[b] else 512
            last_layer[b, q_end + 2:para_end] = self.biatten(last_layer_[b, q_end + 2:para_end].unsqueeze(0),
                                                             last_layer_[b, 1:q_end].unsqueeze(0)).squeeze(0)
            last_layer[b, :q_end + 2] = last_layer_[b, :q_end + 2]
            last_layer[b, para_end:] = last_layer_[b, para_end:]

        # predict span (original start&end score)
        span_out = self.predict_span(last_layer)
        predict_y1 = span_out[:, :, 0]
        predict_y2 = span_out[:, :, 1]

        # get sentence embedding using self-alignment method
        expanded = self.expand_sent(last_layer)
        weight = self.word_weight(expanded)
        repeat_weight = weight.squeeze(2).unsqueeze(1).repeat(1, sent_mapping.size(1), 1)
        repeat_weight = repeat_weight.masked_fill(mask=(1 - sent_mapping).bool(), value=torch.tensor(-16384))
        repeat_weight = torch.softmax(repeat_weight, dim=2)
        sent_embedding = self.LN(torch.bmm(repeat_weight, expanded))

        # dynamic graph
        for layer, GAT_layer in enumerate(self.GAT):
            sent_embedding = GAT_layer(sent_embedding, dyn_mask[:, layer].transpose(-1, -2), node_to_up_mask[:, layer],
                                       graph_weight.transpose(-1, -2))

        # self-attention between sentences without question
        sp_mask_2d = torch.bmm(sp_mask.unsqueeze(2), sp_mask.unsqueeze(1)).int()
        predict_sp = self.sp2(torch.tanh(self.sp1(sent_embedding[:, 1:])))

        # predict type (i.e., yes/no/span)
        type_a = self.type_pool(self.typelayer1a(sent_embedding)).squeeze(1)
        type_b = self.typelayer1b(last_layer[:, 0])
        type_out = self.typelayer2(torch.tanh(torch.cat([type_a, type_b], -1)))

        # paragrph score
        paras = torch.tanh(self.para_score1(sent_embedding[:, 1:]))
        repeat_paras = paras.unsqueeze(1).repeat(1, para_mapping.size(1), 1, 1)  # [batch, para_num, sent_num, hidden]
        tmp_para_mask = (1 - para_mapping).unsqueeze(3).repeat(1, 1, 1, repeat_paras.size(3)).bool()
        repeat_paras = repeat_paras.masked_fill(mask=tmp_para_mask, value=torch.tensor(-16384))
        score_paras = self.para_score2(self.para_pool(repeat_paras)).squeeze(3).squeeze(2).unsqueeze(
            1)  # [B, 1, para_num]
        mapped_score_paras = torch.bmm(score_paras, para_mapping.type_as(score_paras)).squeeze(1)  # [batch, sent_num]

        # sentence score
        score_sents = self.sent_score(sent_embedding[:, 1:]).squeeze(2)
        score_sents += mapped_score_paras
        mapped_score_sents = torch.bmm(score_sents.unsqueeze(1), sent_mapping[:, 1:].type_as(score_sents)).squeeze(1)

        predict_y1 += mapped_score_sents
        predict_y2 += mapped_score_sents

        predict_y1 = predict_y1.masked_fill(mask=(1 - mask).bool(), value=torch.tensor(-16384))
        predict_y2 = predict_y2.masked_fill(mask=(1 - mask).bool(), value=torch.tensor(-16384))
        sent = self.prj1(sent_embedding)
        token = self.prj2(last_layer)
        return type_out, predict_y1, predict_y2, predict_sp, sent, token

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight", "LN.weight"]
        grouped_params = [
            {
                "params": [p for n, p in self.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.002,
            },
            {
                "params": [p for n, p in self.roberta.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                'params': [p for n, p in self.named_parameters() if
                           'roberta' not in n and not any(nd in n for nd in no_decay)],
                "weight_decay": 0.002,
            },
            {
                'params': [p for n, p in self.named_parameters() if
                           'roberta' not in n and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        return grouped_params


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


nll_type = nn.CrossEntropyLoss()
nll_span = nn.CrossEntropyLoss(ignore_index=-100)
# 证据句正类少
# w = torch.tensor([1.0,4]).to(device)
# weight=w,
nll_sp = nn.CrossEntropyLoss(ignore_index=-100)


class contrastloss(nn.Module):
    def __init__(self, temperature=0.1, scale_by_temperature=True):
        super(contrastloss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, h, target):
        celoss = nn.CrossEntropyLoss(ignore_index=-100)
        simlarity = torch.div(F.cosine_similarity(h[:, 0].unsqueeze(1), h[:, 1:], dim=-1),
                              self.temperature)  # 计算问题与句子的点乘相似度 [N,p+n]
        loss = celoss(simlarity, target)
        return loss


CLLoss = contrastloss()


# 答案边界位置的对比损失
def span_contrast(y1, y2, score_matrix, token, sent, record_qp1p2):
    k = 1  # 邻居范围
    # 将下三角和段落以外的部分，mask为0
    mask_tensor = torch.triu(torch.ones(score_matrix.size(), dtype=torch.int)).to(device)
    for b in range(len(y1)):
        mask_tensor[b, :record_qp1p2[b][0]] = 0
        mask_tensor[b, :, :record_qp1p2[b][0]] = 0
        mask_tensor[b, record_qp1p2[b][3]:] = 0
        mask_tensor[b, :, record_qp1p2[b][3]:] = 0

    score_matrix = score_matrix.masked_fill(mask=(1 - mask_tensor).bool(), value=torch.tensor(-1e9))
    nei_tensor_y1 = torch.zeros_like(mask_tensor).type_as(mask_tensor).to(device)
    nei_tensor_y2 = torch.zeros_like(mask_tensor).type_as(mask_tensor).to(device)
    row = score_matrix.size(-1)
    pool = nn.AdaptiveMaxPool2d((1, token.size(-1)))
    target = torch.zeros_like(y1).type_as(y1).to(device)
    h = torch.FloatTensor().to(device)
    neg_num = 10  # 负样本总个数
    ans_tensor = torch.FloatTensor().to(device)
    for b in range(len(y1)):
        feature = torch.zeros([neg_num + 2, token.size(-1)], dtype=torch.float32).to(device)
        # print('feature',type(feature))
        feature[0] = sent[b]  # 问题
        if y1[b] == -100 or y2[b] == -100:  # yes/no正负样本随便取一个
            feature[1:] = sent[b]  # 正样本
            ans_tensor = torch.cat([ans_tensor, feature[1].unsqueeze(0)], dim=0)
            target[b] = -100
        else:
            feature[1] = pool(token[b, y1[b]:y2[b] + 1].unsqueeze(0)).squeeze(0).squeeze(0)  # 正样本
            ans_tensor = torch.cat([ans_tensor, feature[1].unsqueeze(0)], dim=0)
            # 正确答案附近,mask_tensor y1 y2附近不为0的位置
            nei_tensor_y1[b, y1[b] - k:y1[b] + k + 1, :] = 1
            nei_tensor_y2[b, :, y2[b] - k:y2[b] + k + 1] = 1
            neg_mask = mask_tensor[b] & nei_tensor_y1[b] & nei_tensor_y2[b]
            # print('neg_mask:',neg_mask)
            neg_y1, neg_y2 = torch.where(neg_mask == 1)
            neg_y1 = neg_y1.cpu().numpy().tolist()
            neg_y2 = neg_y2.cpu().numpy().tolist()
            neg_list = []
            # print(neg_y1, neg_y2)
            for i in range(len(neg_y1)):
                neg_list.append([neg_y1[i], neg_y2[i]])

            # topk
            j = 2
            _, pos = torch.topk(score_matrix[b].view(-1), 15, dim=0)
            i = 0
            while neg_num != len(neg_list) - 1:
                y1_ = torch.div(pos[i], row, rounding_mode='floor')  # //地板除的安全方法
                y2_ = pos[i] % row
                if [y1_, y2_] not in neg_list:  # topk个与正确答案位置不同的作为负样本
                    neg_list.append([y1_.cpu().numpy().tolist(), y2_.cpu().numpy().tolist()])
                i += 1
            neg_list.remove([y1[b], y2[b]])  # 去掉正样本位置
            # print('neg_list:', neg_list)
            for i in range(len(neg_list)):
                feature[i + 2] = pool(token[b, neg_list[i][0]:neg_list[i][1] + 1].unsqueeze(0)).squeeze(0).squeeze(0)

        h = torch.cat([h, feature.unsqueeze(0)], dim=0)
    return CLLoss(h, target), ans_tensor


# 证据句与答案的对比损失
class predContrast(nn.Module):
    def __init__(self, temperature=0.1):
        super(predContrast, self).__init__()
        self.temperature = temperature

    def forward(self, ans, sent, label,qtype):
        # 判断是否为squad数据
        loss = torch.zeros(1).to(device)
        if label[0, 0] == -100:
            return loss
        # mask the position where is padded and 1
        pad_mask = (torch.ones_like(label) * (-100) & label).to(device)
        positive_mask = (torch.ones_like(label) & label).to(device)
        span = torch.where(qtype != -100)[0]
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        sim = torch.div(cos(ans.unsqueeze(1), sent), self.temperature)  # 计算问题与句子的点乘相似度
        sim = sim.masked_fill(mask=pad_mask.bool(), value=torch.tensor(-16384))
        sim = -F.log_softmax(sim, -1) * positive_mask
        if len(span)==len(qtype):
            return sim.sum(-1).mean()
        elif len(span) == 0:
            return loss
        else:
            for index in span:
                loss += sim[index]
            return loss/len(span)

pred_CL = predContrast()

def train(config):
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=[os.path.basename(__file__), 'main.py', 'util2.py'])

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            # pass
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    logging('Config')
    for k, v in config.__dict__.items():
        logging('    - {} : {}'.format(k, v))

    logging("Building model...")
    train_buckets = get_buckets(config.train_record_file, config.local_rank, ddp=distributed)
    dev_buckets = get_buckets(config.dev_record_file, config.local_rank)

    def build_train_iterator():
        return DataIterator(train_buckets, config.batch_size, True, config.sent_limit, config.local_rank, mode='train')

    def build_dev_iterator():
        return DataIterator(dev_buckets, config.batch_size, False, config.sent_limit, config.local_rank, mode='eval')

    lr = config.init_lr

    print('loading roberta-large...')
    roberta = RobertaModel.from_pretrained("roberta-large")
    # tokenizer.add_tokens(['<t>', '</t>'])
    roberta.resize_token_embeddings(50266)  # 0305

    span_model = model(roberta, config)
    span_model.to(device)
    logging('nparams {}'.format(sum([p.nelement() for p in span_model.parameters() if p.requires_grad])))
    # optimizer = AdamW(span_model.get_optimizer(), lr=config.init_lr, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer = optimizers.FusedAdam(span_model.get_optimizer(), adam_w_mode=True, lr=lr, bias_correction=False)
    span_model, optimizer = amp.initialize(span_model, optimizer, opt_level='O2')

    if distributed:
        span_model = DDP(span_model, delay_allreduce=True)

    print(optimizer)

    total_loss = 0
    global_step = 0
    best_dev_F1 = None
    stop_train = False
    start_time = time.time()
    eval_start_time = time.time()
    span_model.train()
    optimizer.zero_grad()

    for epoch in range(10000):
        for data in build_train_iterator():
            # print(global_step)
            idxs_2p = data['idxs_2p'].to(device)
            mask = data['mask'].to(device)
            is_facts = data['is_facts'].to(device)
            sp_mask = data['sp_mask'].to(device)
            y1 = torch.LongTensor(data['y1']).to(device)
            y2 = torch.LongTensor(data['y2']).to(device)
            q_type = data['q_type'].to(device)
            dyn_mask = data['dyn_mask'].to(device)
            node_to_up_mask = data['node_to_up_mask'].to(device)
            sent_mapping = data['sent_mapping'].to(device)
            para_mapping = data['para_mapping'].to(device)
            ans_mask = data['ans_mask'].bool().to(device)
            graph_weight = data['graph_weight'].to(device)
            record_qp1p2 = data['record_qp1p2']
            predict_type, predict_y1, predict_y2, predict_sp, sent, token = \
                span_model(idxs_2p, mask, sent_mapping, sp_mask, para_mapping, dyn_mask, node_to_up_mask, graph_weight)
            predict_y1 = predict_y1.masked_fill(mask=ans_mask, value=torch.tensor(-16384))
            predict_y2 = predict_y2.masked_fill(mask=ans_mask, value=torch.tensor(-16384))
            score_matrix = predict_y1[:, :, None] + predict_y2[:, None]
            loss_type = nll_type(predict_type, q_type)
            span_cl, ans = span_contrast(y1, y2, score_matrix, token, sent[:, 0], record_qp1p2)
            # sup_cl = supcl(sent, is_facts)

            sup_ans_cl = pred_CL(ans, sent[:, 1:], is_facts, q_type)
            loss_sup = nll_sp(predict_sp.view(-1, 2), is_facts.view(-1))
            loss_span = nll_span(predict_y1, y1) + nll_span(predict_y2, y2)
            loss = loss_type + 5 * loss_sup + loss_span + config.cl1 * span_cl + config.cl3 * sup_ans_cl
            accumulate_step = 1
            loss /= accumulate_step

            frozen_step = config.frozen_step
            warm_up_step = frozen_step + config.warm_up_step
            decay_lr = config.decay_lr
            eval_last_rate = config.eval_last_rate

            # frozing BERT and warm up
            if global_step <= frozen_step:
                optimizer.param_groups[0]['lr'] = 0
                optimizer.param_groups[1]['lr'] = 0
                optimizer.param_groups[2]['lr'] = config.frozen_lr
                optimizer.param_groups[3]['lr'] = config.frozen_lr
            elif frozen_step < global_step < warm_up_step:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.1 * lr + (0.9 * lr / (warm_up_step - frozen_step)) * (
                            global_step - frozen_step)
            elif global_step == warm_up_step:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # decay
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

            if global_step % accumulate_step == 0:
                torch.nn.utils.clip_grad_norm_(span_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            if global_step % config.period == 0:
                cur_loss = total_loss / config.period
                elapsed = time.time() - start_time
                logging('| epo {:3d} | step {:6d} | lr {:05.10f} | ms/b {:5.1f} | train loss {:6.2f} |' \
                        .format(epoch, global_step, cur_lr, elapsed * 1000 / config.period, cur_loss))
                total_loss = 0
                start_time = time.time()

            if (cur_lr < config.init_lr * eval_last_rate and global_step % config.checkpoint == 0) or global_step == 2000:
            # if global_step % config.checkpoint == 0:
                span_model.eval()
                metrics = evaluate_batch(build_dev_iterator(), span_model, 0, dev_eval_file, config)
                span_model.train()

                logging('-' * 89)
                logging('| eval {:6d} in epo {:3d} | time {:5.2f}s | EM {:.4f} | F1 {:.4f}' \
                        .format(global_step // config.checkpoint, epoch, time.time() - eval_start_time, \
                                metrics['exact_match'], metrics['f1']))
                logging('| sp_em {:.4f} | sp_prec {:.4f} | sp_recall {:.4f} | sp_f1 {:.4f}' \
                        .format(metrics['sp_em'], metrics['sp_prec'], metrics['sp_recall'], metrics['sp_f1']))
                logging('| joint_em {:.4f} |  joint_f1 {:.4f}' \
                        .format(metrics['joint_em'], metrics['joint_f1']))
                logging('-' * 89)

                eval_start_time = time.time()

                dev_F1 = metrics['f1']
                if best_dev_F1 is None or dev_F1 > best_dev_F1:
                    best_dev_F1 = dev_F1
                    torch.save(span_model.state_dict(),
                               os.path.join(config.save, 'span_model' + str(config.local_rank) + '.pt'))
                    torch.save(optimizer.state_dict(),
                               os.path.join(config.save, 'optimizer' + str(config.local_rank) + '.pt'))

        if stop_train: break
    logging('best_dev_f1 {}'.format(best_dev_F1))


def evaluate_batch(data_source, span_model, max_batches, eval_file, config):
    answer_dict = {}
    sp_dict = {}
    # max_batches = 10
    for step, data in enumerate(data_source):
        if step >= max_batches and max_batches > 0: break
        idxs_2p = data['idxs_2p'].to(device)
        mask = data['mask'].to(device)
        is_facts = data['is_facts'].to(device)
        sp_mask = data['sp_mask'].to(device)
        y1 = torch.LongTensor(data['y1']).to(device)
        y2 = torch.LongTensor(data['y2']).to(device)
        q_type = data['q_type'].to(device)
        dyn_mask = data['dyn_mask'].to(device)
        node_to_up_mask = data['node_to_up_mask'].to(device)
        sent_mapping = data['sent_mapping'].to(device)
        para_mapping = data['para_mapping'].to(device)
        graph_weight = data['graph_weight'].to(device)
        support_para = data['support_para']
        record_qp1p2 = data['record_qp1p2']

        with torch.no_grad():
            predict_type, predict_y1, predict_y2, predict_sp, _, _ = \
                span_model(idxs_2p, mask, sent_mapping, sp_mask, para_mapping, dyn_mask, node_to_up_mask, graph_weight)

        final_y1 = []
        final_y2 = []
        final_para = []

        # joint score for start&end positions
        score_matrix = predict_y1[:, :, None] + predict_y2[:, None]

        # supporting fact prediction
        pred_sp = F.softmax(predict_sp[:, :, :2], dim=2).data.cpu().numpy()
        pred_sp = pred_sp[:, :, 1]

        # iterate over instances to calculate
        for b in range(len(y1)):
            # for answer span
            scores = []
            for s in range(record_qp1p2[b][0], record_qp1p2[b][3]):
                if s >= score_matrix.size(1): break
                for e in range(s, min(record_qp1p2[b][3], s + 15)):
                    if e >= score_matrix.size(1): break
                    # scores.append((s, e, predict_y1[b,s] * pedict_y2[b,e]))
                    scores.append((s, e, score_matrix[b, s, e].item()))

            scores.sort(key=lambda x: x[2], reverse=True)
            tmp_y1 = scores[0][0]
            tmp_y2 = scores[0][1]
            # predicted answer is in 1st paragraph in BERT
            if record_qp1p2[b][0] <= tmp_y1 < record_qp1p2[b][1] and \
                    record_qp1p2[b][0] <= tmp_y2 < record_qp1p2[b][1]:
                final_y1.append(tmp_y1 - record_qp1p2[b][0])
                final_y2.append(tmp_y2 - record_qp1p2[b][0])
                final_para.append(support_para[b][0])
            # predicted answer is in 2nd paragraph in BERT
            elif record_qp1p2[b][1] <= tmp_y1 < record_qp1p2[b][2] and \
                    record_qp1p2[b][1] <= tmp_y2 < record_qp1p2[b][2]:
                final_y1.append(tmp_y1 - record_qp1p2[b][1])
                final_y2.append(tmp_y2 - record_qp1p2[b][1])
                final_para.append(support_para[b][1])
            # predicted answer is in 3th paragraph in BERT
            elif tmp_y1 >= record_qp1p2[b][2] and tmp_y2 >= record_qp1p2[b][2]:
                final_y1.append(tmp_y1 - record_qp1p2[b][2])
                final_y2.append(tmp_y2 - record_qp1p2[b][2])
                final_para.append(support_para[b][2])
            else:
                print('predicted start & end are not in the same paragraph.')
                final_y1.append(0)
                final_y2.append(0)
                final_para.append(0)

            # for supporting fact
            cur_id = data['ids'][b]
            cur_sp_pred = []
            sent_ptr = 0
            for p in support_para[b]:
                for each in eval_file[cur_id]['sent2title_ids'][p]:
                    if sent_ptr >= pred_sp.shape[1]: break
                    if pred_sp[b, sent_ptr] > config.sp_threshold:
                        cur_sp_pred.append(each)
                    sent_ptr += 1
            sp_dict.update({cur_id: cur_sp_pred})

        answer_dict_ = convert_tokens(eval_file, data['ids'], final_y1, final_y2, final_para,
                                      np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

    metrics = evaluate(eval_file, answer_dict, sp_dict)
    return metrics


def test(config):
    with open(config.test_eval_file, "r") as fh:
        test_eval_file = json.load(fh)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    test_buckets = get_buckets(config.dev_record_file, config.local_rank)

    def build_test_iterator():
        return DataIterator(test_buckets, config.batch_size, False, config.sent_limit, config.local_rank, mode='eval')

    print('loading roberta-large...')
    roberta = RobertaModel.from_pretrained("roberta-large")
    roberta.resize_token_embeddings(50266)
    span_model = model(roberta, config)
    span_model.cuda()

    span_model.load_state_dict({k.replace('module.', ''): v for k, v in
                                torch.load(os.path.join(config.save, 'span_model0.pt'), map_location='cuda:0').items()})
    span_model.eval()

    predict(build_test_iterator(), span_model, test_eval_file, config, config.prediction_file)


def predict(data_source, span_model, eval_file, config, prediction_file):
    answer_dict = {}
    sp_dict = {}
    for step, data in enumerate(tqdm(data_source)):
        # if step > 100: break
        idxs_2p = data['idxs_2p'].to(device)
        mask = data['mask'].to(device)
        is_facts = data['is_facts'].to(device)
        sp_mask = data['sp_mask'].to(device)
        y1 = torch.LongTensor(data['y1']).to(device)
        y2 = torch.LongTensor(data['y2']).to(device)
        q_type = data['q_type'].to(device)
        dyn_mask = data['dyn_mask'].to(device)
        node_to_up_mask = data['node_to_up_mask'].to(device)
        sent_mapping = data['sent_mapping'].to(device)
        para_mapping = data['para_mapping'].to(device)
        graph_weight = data['graph_weight'].to(device)
        support_para = data['support_para']
        record_qp1p2 = data['record_qp1p2']

        with torch.no_grad():
            predict_type, predict_y1, predict_y2, predict_sp, _, _= \
                span_model(idxs_2p, mask, sent_mapping, sp_mask, para_mapping, dyn_mask, node_to_up_mask, graph_weight)
            predict_y1 = F.softmax(predict_y1, dim=-1)
            predict_y2 = F.softmax(predict_y2, dim=-1)

        final_y1 = []
        final_y2 = []
        final_para = []

        # joint score for start&end positions
        score_matrix = predict_y1[:, :, None] + predict_y2[:, None]

        # supporting fact prediction
        pred_sp = F.softmax(predict_sp[:, :, :2], dim=2).data.cpu().numpy()  # predict_sp [bzs,num_sent,2]
        pred_sp = pred_sp[:, :, 1]  # 1为是证据的概率

        # iterate over instances to calculate
        for b in range(len(y1)):
            # for answer span
            scores = []
            for s in range(record_qp1p2[b][0], record_qp1p2[b][3]):  # p1p3的start
                if s >= score_matrix.size(1): break
                for e in range(s, min(record_qp1p2[b][3], s + 15)):  # end限定在start之后的这个区间内
                    if e >= score_matrix.size(1): break
                    scores.append((s, e, score_matrix[b, s, e].item()))
            scores.sort(key=lambda x: x[2], reverse=True)

            cur_win = 0  # begin from the highest score
            while (True):
                tmp_y1 = scores[cur_win][0]
                tmp_y2 = scores[cur_win][1]
                # predicted answer is in 1st paragraph in BERT
                if record_qp1p2[b][0] <= tmp_y1 < record_qp1p2[b][1] and \
                        record_qp1p2[b][0] <= tmp_y2 < record_qp1p2[b][1]:
                    final_y1.append(tmp_y1 - record_qp1p2[b][0])
                    final_y2.append(tmp_y2 - record_qp1p2[b][0])
                    final_para.append(support_para[b][0])
                    break
                # predicted answer is in 2nd paragraph in BERT
                elif record_qp1p2[b][1] <= tmp_y1 < record_qp1p2[b][2] and \
                        record_qp1p2[b][1] <= tmp_y2 < record_qp1p2[b][2]:
                    final_y1.append(tmp_y1 - record_qp1p2[b][1])
                    final_y2.append(tmp_y2 - record_qp1p2[b][1])
                    final_para.append(support_para[b][1])
                    break
                # predicted answer is in 3th paragraph in BERT
                elif tmp_y1 >= record_qp1p2[b][2] and tmp_y2 >= record_qp1p2[b][2]:
                    final_y1.append(tmp_y1 - record_qp1p2[b][2])
                    final_y2.append(tmp_y2 - record_qp1p2[b][2])
                    final_para.append(support_para[b][2])
                    break
                # if (start, end) not in the same paragraph, find the next one
                else:
                    cur_win += 1
                    if cur_win >= 20:
                        print('predicted start & end are not in the same paragraph.')
                        final_y1.append(0)
                        final_y2.append(0)
                        final_para.append(0)
                        break

            # for supporting fact
            cur_id = data['ids'][b]
            cur_sp_pred = []
            sent_ptr = 0
            for p in support_para[b]:  # 段落
                for each in eval_file[cur_id]['sent2title_ids'][p]:  # title和句子及其序号
                    if sent_ptr >= pred_sp.shape[1]: break
                    if pred_sp[b, sent_ptr] > config.sp_threshold:
                        cur_sp_pred.append(each)
                    sent_ptr += 1
            sp_dict.update({cur_id: cur_sp_pred})

        answer_dict_ = convert_tokens(eval_file, data['ids'], final_y1, final_y2, final_para,
                                      np.argmax(predict_type.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

    metrics = evaluate(eval_file, answer_dict, sp_dict)

    def logging(s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(config.save, 'log.txt'), 'a+') as f_log:
                f_log.write(s + '\n')

    for k, v in metrics.items():
        logging('{} : {:.2f}'.format(k, v))
    # print(metrics)
    prediction = {'answer': answer_dict, 'sp': sp_dict}
    with open(config.save + '/pred.json', 'w') as f:
        json.dump(prediction, f)
    logging('*****************\nfinal metrics\n')
    # eval by hotpot_evaluate.py
    final_metrics = eval_fun(config.save + '/pred.json', 'hotpot_dev_distractor_v1.json')
    for k, v in final_metrics.items():
        logging('{} : {:.2f}'.format(k, v))
