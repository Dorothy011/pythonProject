import torch
import numpy as np
import re
from collections import Counter
import string
import pickle
import random
from torch.autograd import Variable
import copy
import ujson as json
import traceback
import spacy
import os

IGNORE_INDEX = -100

# load the result from stage 1
id2para = {}
with open('../stage1/predict_para.txt', 'rb') as file:
    for line in file:
        arr = line.decode("utf8").replace('\r', '').replace('\n', '').split(' ')  # arr: id   p1  p2  p3
        id2para[arr[0]] = [k for k in arr[1:]]  # id2para[id] = [4,6,0]


# if two entities are alike
def ents_alike(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    for punc in string.punctuation:
        str1 = str1.replace(punc, ' ')
        str2 = str2.replace(punc, ' ')
    in12 = True
    in21 = True
    for s in str1.split():
        if s not in str2:
            in12 = False
            break
    for s in str2.split():
        if s not in str1:
            in21 = False
            break
    if in12 or in21:
        return True
    else:
        return False


# connect two sentences with co-occurance entities
def link_sents(list1, list2):
    for ent1 in list1:
        for ent2 in list2:
            if ents_alike(ent1, ent2):
                return True
    return False


# entity recognition for title
def ner_title(sent):
    punc_list = ['(', '（', '[', '【', '']
    i = sent.find('(')
    if i > 0:
        sent = sent[:i]
    for punc in string.punctuation:
        sent = sent.replace(punc, ' ')
    ent = sent.split()
    return [e for e in ent if len(e) >= 2]


class DataIterator(object):
    def __init__(self, buckets, bsz, shuffle, sent_limit, local_rank, mode):
        self.buckets = buckets
        self.bsz = bsz  # 2
        self.mode = mode
        self.shuffle = shuffle
        self.sent_limit = sent_limit
        self.num_buckets = len(self.buckets)
        self.bkt_pool = [i for i in range(self.num_buckets) if len(self.buckets[i]) > 0]
        if shuffle:
            for i in range(self.num_buckets):
                random.shuffle(self.buckets[i])
        self.bkt_ptrs = [0 for i in range(self.num_buckets)]
        self.squad_data = torch.load('../data/squad_record.pkl')
        self.local_rank = local_rank

    def __iter__(self):
        q_type = torch.LongTensor(self.bsz)
        sent_limit = self.sent_limit  # 25
        while True:
            if len(self.bkt_pool) == 0: break
            random.shuffle(self.bkt_pool)
            bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]
            start_id = self.bkt_ptrs[bkt_id]
            cur_bucket = self.buckets[bkt_id]
            cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

            if self.local_rank != 2 or self.mode != 'train':
                cur_batch = cur_bucket[start_id: start_id + cur_bsz]
                squad = False
            else:
                cur_batch = random.sample(self.squad_data, cur_bsz)
                squad = True

            ids = []
            ques_idxs = []
            idxs_2p = []
            y1_batch = []
            y2_batch = []
            is_facts_batch = []
            support_para_batch = []
            matrix_batch = []
            ans_mask = []
            sent_mapping_batch = []
            para_mapping_batch = []
            record_qp1p2 = []
            weight_batch = []
            K1, K2 = 2, -2
            for i in range(len(cur_batch)):
                ids.append(cur_batch[i]['id'])  # unique id for each training instance
                q_idxs = cur_batch[i]['ques_idxs']  # question (encoded with roberta tokenizer)
                len_q = len(q_idxs)
                if squad: len_q += 1
                ques_idxs.append(q_idxs)
                y1y2s = cur_batch[i]['y1y2s']  # span locations list(一个或多个，或者是yes/no): [[1, (15, 18)], [1, (31, 34)]]
                # 可以有多个span，可能是多个不同答案，也可能是不同位置的同一个答案
                # if multiple spans, randomly mask others,只取一个span作为ground truth
                random.shuffle(y1y2s)
                y1y2 = y1y2s[0]  # 取第一个[Para id, (start, end)] 作为label
                y1y2_ = y1y2s[1:]  # to be masked

                if y1y2[1][0] >= 0:  # 第一个段落序号 >=0，说明存在span
                    y1, y2 = y1y2
                    q_type[i] = 0
                elif y1y2[1][0] == -1:  #
                    y1 = y2 = IGNORE_INDEX
                    q_type[i] = 1
                elif y1y2[1][0] == -2:
                    y1 = y2 = IGNORE_INDEX
                    q_type[i] = 2
                elif y1y2[1][0] == -3:
                    y1 = y2 = IGNORE_INDEX
                    q_type[i] = 3
                else:
                    assert False

                # when training, use ground-truth paragraphs; when testing, use predicted paragraphs
                if self.mode == 'train':
                    support_para = copy.deepcopy(cur_batch[i]['sp_para'])  # useful paragraphs
                    try:
                        support_para += random.sample(
                            {k for k in range(len(cur_batch[i]['context_idxs']))} - set(support_para),
                            1)  # 随机取1个neg_para(1/8)
                    except ValueError:
                        pass
                else:
                    support_para = [int(k) for k in id2para[cur_batch[i]['id']]]  # test时使用上一步预测的段落序号

                # 以一定概率把BERT的段落顺序打乱来训练，相当于数据增强
                if self.mode == 'train' and random.random() > 0.5:
                    random.shuffle(support_para)
                support_para_batch.append(support_para)
                # 提取出输入到BERT的段落编号
                # 由于最多只有3段（记作p1,p2,p3），因此代码写死了只有3段
                # 这种写法很粗暴、拓展性差，但没来得及改
                # 后面也有类似的写法
                p1p2p3 = [None, None, None]
                p1p2p3[:len(support_para)] = support_para
                p1, p2, p3 = p1p2p3

                # 建立从token到paragraph/sentence映射矩阵，便于在模型中对其并行处理
                cur_sef = cur_batch[i]['start_end_facts']  # 每个段落中，从第几到第几token是一个句子，以及是否是supporting sentence
                is_facts = []  # not contain question
                sent_mapping = np.zeros((sent_limit, 512), dtype='int')  # sent_limit=25，contain question
                para_mapping = np.zeros((3, sent_limit), dtype='int')  # not contain question
                sent_mapping[0, 1:len_q + 1] = 1
                j = 1
                allow_p2, allow_p3 = True, True
                # p1
                # 手动添加特殊token
                for e in cur_sef[p1]:
                    if e[1] + len_q + 3 <= 512 and j < sent_limit:  # 句子长度+问题长度+3个符号（<s> </s> <s>）
                        start = e[0] + len_q + 3
                        end = e[1] + len_q + 3
                        sent_mapping[j, start:end] = 1  # [[Q],[p1],[p2],[p3],...[p9]],只有预测的三个段落处的token_mapping为1
                        is_facts.append(int(e[2]))
                        j += 1
                    else:
                        allow_p2 = False  # 句子长度或个数超过上限，就不再添加预测的句子
                p1_where = j - 1
                para_mapping[0, :p1_where] = 1  # 指出该段的句子所在位置
                """
                1 1 1 1 0 0 0 0 0 0 0       p1
                0 0 0 0 1 1 1 0 0 0 0       p2
                0 0 0 0 0 0 0 1 1 0 0       p3
                一个数字代表一个句子，一行25个数字，不含问题
                """
                # p2
                if allow_p2 and p2 != None:
                    tmp_len = end  # p2 starts from where
                    for e in cur_sef[p2]:
                        if e[1] + tmp_len <= 512 and j < sent_limit:
                            start = e[0] + tmp_len
                            end = e[1] + tmp_len
                            sent_mapping[j, start:end] = 1
                            is_facts.append(int(e[2]))
                            j += 1
                        else:
                            allow_p3 = False
                    p2_where = j - 1
                    para_mapping[1, p1_where:p2_where] = 1
                # p3
                if allow_p2 and allow_p3 and p3 != None:
                    tmp_len = end  # p3 starts from where
                    for e in cur_sef[p3]:
                        if e[1] + tmp_len <= 512 and j < sent_limit:
                            start = e[0] + tmp_len
                            end = e[1] + tmp_len
                            sent_mapping[j, start:end] = 1
                            is_facts.append(int(e[2]))
                            j += 1
                    para_mapping[2, p2_where:j - 1] = 1
                tmp_sents_num = j
                sent_mapping_batch.append(sent_mapping)
                is_facts_batch.append(is_facts)
                para_mapping_batch.append(para_mapping)
                # 建立GNN的邻接矩阵，计算先验权重
                para_ents = cur_batch[i]['entities']  # 每个段落中，每个句子中，有哪些实体
                ents = [cur_batch[i]['ques_entities']]  # 问题中有哪些实体
                for pk in [p1, p2, p3]:
                    if pk is not None: ents += para_ents[pk]
                # where_split指的是段落之间在哪个句子划分
                where_split1 = len([cur_batch[i]['ques_entities']] + para_ents[p1]) if p2 is not None else 998
                where_split2 = len(
                    [cur_batch[i]['ques_entities']] + para_ents[p1] + para_ents[p2]) if p3 is not None else 999
                tmp_len = min(tmp_sents_num, len(ents))
                matrix = np.zeros((tmp_len, tmp_len), dtype=int)
                weight = np.zeros((tmp_len, tmp_len))
                for aa in range(tmp_len - 1):
                    for bb in range(aa + 1, tmp_len):
                        num_ents = link_sents(ents[aa], ents[bb])
                        if num_ents >= 1:
                            matrix[aa][bb] = 1
                            weight[aa][bb] = num_ents - K1
                            # weight[aa][bb] = min(num_ents, 5)
                        elif (1 <= aa < where_split1 and 1 <= bb < where_split1) \
                                or (where_split1 <= aa < where_split2 and where_split1 <= bb < where_split2) \
                                or (where_split2 <= aa and where_split2 <= bb):
                            matrix[aa][bb] = 1
                            weight[aa][bb] = -K2 + aa - bb
                            # weight[aa][bb] = min(6 + bb - aa, 10)
                # 对称矩阵
                matrix += matrix.T
                weight += weight.T
                matrix_batch.append(matrix)
                weight_batch.append(weight)

                # 把问题和已选段落拼接起来，用于BERT输入（记作2p），并将span对齐到其中
                cur_context = cur_batch[i]['context_idxs']
                if p3 is not None:
                    idxs_2p.append([0] + q_idxs + [2, 2] + cur_context[p1] + cur_context[p2] + cur_context[p3] + [2])
                elif p2 is not None:
                    idxs_2p.append([0] + q_idxs + [2, 2] + cur_context[p1] + cur_context[p2] + [2])
                elif p1 is not None:
                    idxs_2p.append([0, 50265] + q_idxs + [2, 2] + cur_context[p1] + [2])  # SQuAD
                if y1 != IGNORE_INDEX:
                    if y1y2[0] == p1:  # span在p1
                        y1_2p = y1y2[1][0] + len_q + 3
                        y2_2p = y1y2[1][1] + len_q + 3
                    elif y1y2[0] == p2:  # span在p2
                        y1_2p = y1y2[1][0] + len_q + 3 + len(cur_context[p1])
                        y2_2p = y1y2[1][1] + len_q + 3 + len(cur_context[p1])
                    elif y1y2[0] == p3:  # span在p3
                        y1_2p = y1y2[1][0] + len_q + 3 + len(cur_context[p1]) + len(cur_context[p2])
                        y2_2p = y1y2[1][1] + len_q + 3 + len(cur_context[p1]) + len(cur_context[p2])
                    else:
                        y1_2p = IGNORE_INDEX
                        y2_2p = IGNORE_INDEX
                else:
                    y1_2p = IGNORE_INDEX
                    y2_2p = IGNORE_INDEX
                if len_q >= 500: y1_2p, y2_2p = -100, -100
                y1_batch.append(y1_2p)
                y2_batch.append(y2_2p)

                # 记录问题和已选段落的长度，便于在测试时排除跨越不同段落的span
                if p3 is not None:
                    record_qp1p2.append([len_q + 3, len_q + 3 + len(cur_context[p1]),
                                         len_q + 3 + len(cur_context[p1]) + len(cur_context[p2]), \
                                         len_q + 3 + len(cur_context[p1]) + len(cur_context[p2]) + len(
                                             cur_context[p3])])
                elif p2 is not None:
                    record_qp1p2.append([len_q + 3, len_q + 3 + len(cur_context[p1]),
                                         len_q + 3 + len(cur_context[p1]) + len(cur_context[p2]), 999])
                elif p1 is not None:
                    record_qp1p2.append([len_q + 3, len_q + 3 + len(cur_context[p1]), 998, 999])

                # 有时会有多个span都能作为答案，这时随机选一个作为答案，而将其他mask掉
                if p3 is not None:
                    con_ = [0 for _ in
                            range(len_q + len(cur_context[p1]) + len(cur_context[p2]) + len(cur_context[p3]) + 4)]
                elif p2 is not None:
                    con_ = [0 for _ in range(len_q + len(cur_context[p1]) + len(cur_context[p2]) + 4)]
                elif p1 is not None:
                    con_ = [0 for _ in range(len_q + len(cur_context[p1]) + 4)]
                where_split1 = len_q + len(cur_context[p1]) + 3
                where_split2 = len_q + len(cur_context[p1]) + len(cur_context[p2]) + 3 if p2 is not None else 998
                for p_, (y1_, y2_) in y1y2_:
                    if p_ == p1:  # span在p1
                        tmp1_ = len_q + 3 + y1_
                        tmp2_ = len_q + 3 + y2_
                        con_[tmp1_:tmp2_ + 1] = [1 for _ in range(tmp2_ - tmp1_ + 1)]
                    elif p_ == p2:  # span在p2
                        tmp1_ = where_split1 + y1_
                        tmp2_ = where_split1 + y2_
                        con_[tmp1_:tmp2_ + 1] = [1 for _ in range(tmp2_ - tmp1_ + 1)]
                    elif p_ == p3:  # span在p3
                        tmp1_ = where_split2 + y1_
                        tmp2_ = where_split2 + y2_
                        con_[tmp1_:tmp2_ + 1] = [1 for _ in range(tmp2_ - tmp1_ + 1)]
                ans_mask.append(con_)

            # convert lists to tensors, and cut when too long
            max_facts = max(len(x) for x in is_facts_batch)
            facts_tensor = torch.zeros(len(cur_batch), max_facts).long() + IGNORE_INDEX
            max_len_2p = max(len(x) for x in idxs_2p)
            idxs_2p_tensor = torch.ones(len(cur_batch), max_len_2p).long()
            ans_mask_tensor = torch.zeros(len(cur_batch), max_len_2p).int()
            sent_mapping_tensor = torch.zeros(len(cur_batch), max_facts + 1, max_len_2p).int()
            para_mapping_tensor = torch.Tensor(para_mapping_batch).int()
            para_mapping_tensor = para_mapping_tensor[:, :, :max_facts]
            matrix_tensor = torch.zeros(len(cur_batch), max_facts + 1, max_facts + 1).int()
            weight_tensor = torch.zeros(len(cur_batch), max_facts + 1, max_facts + 1)  # .long()
            for i in range(len(cur_batch)):
                facts_tensor[i, :len(is_facts_batch[i])] = torch.LongTensor(is_facts_batch[i])
                idxs_2p_tensor[i, :len(idxs_2p[i])] = torch.LongTensor(idxs_2p[i])
                ans_mask_tensor[i, :len(ans_mask[i])] = torch.LongTensor(ans_mask[i])
                expect_len_2p = min(max_len_2p, 512)
                sent_mapping_tensor[i, :, :expect_len_2p] = \
                    torch.from_numpy(sent_mapping_batch[i][:max_facts + 1, :expect_len_2p])
                matrix_tensor[i, :len(matrix_batch[i]), :len(matrix_batch[i])] = \
                    torch.from_numpy(matrix_batch[i])
                weight_tensor[i, :len(weight_batch[i]), :len(weight_batch[i])] = \
                    torch.from_numpy(weight_batch[i])
                if y1_batch[i] >= 512:
                    y1_batch[i] = IGNORE_INDEX
                if y2_batch[i] >= 512:
                    y2_batch[i] = IGNORE_INDEX
            if max_len_2p > 512:
                idxs_2p_tensor = idxs_2p_tensor[:, :512]
                ans_mask_tensor = ans_mask_tensor[:, :512]
                sent_mapping_tensor = sent_mapping_tensor[:, :, :512]

            # mask
            mask = np.zeros((len(cur_batch), max_len_2p))
            for i in range(len(cur_batch)):
                mask[i, len(ques_idxs[i]) + 3: len(idxs_2p[i]) - 1] = 1
            mask = torch.from_numpy(mask)
            if mask.size(1) > 512:
                mask = mask[:, :512]
            sp_mask = torch.where(facts_tensor == 3, torch.full_like(facts_tensor, 0),
                                  torch.full_like(facts_tensor, 1)).half()

            self.bkt_ptrs[bkt_id] += cur_bsz
            if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
                self.bkt_pool.remove(bkt_id)

            # get dynamic mask
            # 1表示该连边允许通过，0代表不允许通过
            # mask1代表邻居连边（静态）
            # mask2代表当前活跃节点（动态），仅允许当前节点更新其他节点
            # mask3记录已走过的连边（动态），已走过的置0
            dyn_mask = torch.IntTensor()
            node_to_up_mask = torch.IntTensor()
            for i in range(len(cur_batch)):
                mask1 = matrix_tensor[i].numpy()
                kk = [k for k in range(len(mask1))]
                mask3 = np.ones_like(mask1, dtype='int')
                cur_node = [0]
                pre_node = None
                dyn_mask_ = torch.IntTensor()
                node_to_up_mask_ = torch.IntTensor()
                for j in range(4):  # 最大层数
                    mask2 = np.zeros_like(mask1, dtype='int')
                    for n in cur_node:
                        mask2[n] = 1
                    final_mask = mask1 & mask2 & mask3
                    node_to_update = np.where(final_mask[cur_node] == 1)[1]
                    final_mask[kk, kk] = 1

                    dyn_mask_ = torch.cat([dyn_mask_, torch.from_numpy(final_mask).int().unsqueeze(0)], 0)
                    tmp_mask = np.zeros(len(mask1), dtype='int')
                    tmp_mask[node_to_update] = 1
                    node_to_up_mask_ = torch.cat([node_to_up_mask_, torch.from_numpy(tmp_mask).int().unsqueeze(0)], 0)

                    pre_node = cur_node
                    final_mask[kk, kk] = 0
                    cur_node = np.where(final_mask[cur_node] == 1)[1]
                    for pn in pre_node:
                        # mask3[pn, cur_node] = 0
                        # mask3[cur_node, pn] = 0
                        mask3[pn] = 0
                        mask3[:, pn] = 0
                dyn_mask = torch.cat([dyn_mask, dyn_mask_.unsqueeze(0)], 0)
                node_to_up_mask = torch.cat([node_to_up_mask, node_to_up_mask_.unsqueeze(0)], 0)


            yield {
                'ids': ids,
                'idxs_2p': idxs_2p_tensor,
                'mask': mask,
                'is_facts': facts_tensor,
                'sp_mask': sp_mask,
                'y1': y1_batch,
                'y2': y2_batch,
                'q_type': q_type[:cur_bsz],
                'ans_mask': ans_mask_tensor,
                'dyn_mask': dyn_mask,
                'node_to_up_mask': node_to_up_mask,
                'sent_mapping': sent_mapping_tensor,
                'para_mapping': para_mapping_tensor,
                'graph_weight': weight_tensor,
                'support_para': support_para_batch,
                'record_qp1p2': record_qp1p2
            }


def get_buckets(record_file, local_rank, ddp=False):
    datapoints = torch.load(record_file)
    total_rank = torch.cuda.device_count() - 1  # for SQuAD auxiliary training
    len_data = len(datapoints)
    print('total len_data:', len_data)
    if ddp and local_rank != 2:
        split = len_data // total_rank
        datapoints = datapoints[split * local_rank:split * (local_rank + 1)]
    if ddp:
        print('len_data per in gpu', local_rank, ':', len(datapoints))
    return [datapoints]


def convert_tokens(eval_file, qa_id, pp1, pp2, para, p_type):
    answer_dict = {}
    for qid, p1, p2, p, type in zip(qa_id, pp1, pp2, para, p_type):
        if type == 0:
            context = eval_file[str(qid)]["context"][p]
            spans = eval_file[str(qid)]["spans"][p]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx: end_idx]
        elif type == 1:
            answer_dict[str(qid)] = 'yes'
        elif type == 2:
            answer_dict[str(qid)] = 'no'
        elif type == 3:
            answer_dict[str(qid)] = 'noanswer'
    return answer_dict


def evaluate(eval_file, answer_dict, sp_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answer"]
        prediction = value
        assert len(ground_truths) == 1
        cur_EM = exact_match_score(prediction, ground_truths[0])
        cur_f1, _, _ = f1_score(prediction, ground_truths[0])
        exact_match += cur_EM
        f1 += cur_f1
    sp_em = sp_prec = sp_recall = sp_f1 = 0
    for key, pred_sps in sp_dict.items():
        tp, fp, fn = 0, 0, 0
        ground_truths = eval_file[key]["supporting_facts"]
        for e in pred_sps:
            if e in ground_truths:
                tp += 1
            else:
                fp += 1
        for e in ground_truths:
            if e not in pred_sps:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f_1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        sp_em += em
        sp_prec += prec
        sp_recall += recall
        sp_f1 += f_1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    sp_em = 100.0 * sp_em / total
    sp_prec = 100.0 * sp_prec / total
    sp_recall = 100.0 * sp_recall / total
    sp_f1 = 100.0 * sp_f1 / total

    return {'exact_match': exact_match, 'f1': f1, 'sp_em': sp_em, 'sp_prec': sp_prec, 'sp_recall': sp_recall,
            'sp_f1': sp_f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
