import torch
# from torch import nn
# from torch.nn import functional as F
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
import os


class DataIterator(object):
    def __init__(self, buckets, bsz, shuffle, mode):
        self.buckets = buckets
        self.bsz = bsz
        self.mode = mode
        self.shuffle = shuffle
        self.num_buckets = len(self.buckets)
        self.bkt_pool = [i for i in range(self.num_buckets) if len(self.buckets[i]) > 0]
        if shuffle:
            for i in range(self.num_buckets):
                random.shuffle(self.buckets[i])  # 将列表元素打乱
        self.bkt_ptrs = [0 for i in range(self.num_buckets)]

    def __iter__(self):
        while True:
            if len(self.bkt_pool) == 0: break
            random.shuffle(self.bkt_pool)
            bkt_id = random.choice(self.bkt_pool) if self.shuffle else self.bkt_pool[0]  # random.choice()从序列中获取一个随机元素
            start_id = self.bkt_ptrs[bkt_id]
            cur_bucket = self.buckets[bkt_id]
            cur_bsz = min(self.bsz, len(cur_bucket) - start_id)

            cur_batch = cur_bucket[start_id: start_id + cur_bsz]
            max_sent_cnt = 0

            ids = []
            ques_idxs = []
            idxs_2p = []
            target_para = []
            para_num_eval = []
            train_para = []
            eval_para = []
            eval_para_target = []
            for i in range(len(cur_batch)):
                ids.append(cur_batch[i]['id'])  # unique id for each training instance
                q_idxs = cur_batch[i]['ques_idxs']  # question (encoded with roberta tokenizer)
                ques_idxs.append(q_idxs)
                support_para = cur_batch[i]['sp_para']  # which paragraphs are useful
                cur_context = cur_batch[i]['context_idxs']  # all given paragraphs (encoded with roberta tokenizer)
                # ans_para = cur_batch[i]['y1y2s'][0][0]
                # if ans_para!=support_para[0]:  # 交换包含答案的段落到前面
                #     support_para[1]=support_para[0]
                #     support_para[0]=ans_para

                # for run_para training
                # sample paragraphs as negative instances
                neg_limit = 2
                neg_para = list(set([w for w in range(len(cur_context))]) - set(support_para))
                cur_neg_num = min(neg_limit, len(neg_para))
                # concatenate the positives and negatives for training
                train_para_id = support_para + random.sample(neg_para, cur_neg_num)
                for p in train_para_id:
                    train_para.append(q_idxs + cur_context[p])
                # labels, 2 paragraphs are positive, others are negative
                target_para.extend([1] * 2 + [0] * cur_neg_num)
                # print(target_para)
                # for run_para eval/testing
                for p in range(len(cur_context)):
                    eval_para.append(q_idxs + cur_context[p])
                    eval_para_target.extend([int(p in support_para)])
                para_num_eval.append(len(cur_context))

            # convert lists to tensors, and cut when too long
            max_len_4p = max(len(x) for x in train_para)
            train_para_tensor = torch.ones(len(train_para), max_len_4p).long()
            for i in range(len(train_para)):
                train_para_tensor[i, :len(train_para[i])] = torch.LongTensor(train_para[i])
            if max_len_4p > 400:
                train_para_tensor = train_para_tensor[:, :400]
            max_len_10p = max(len(x) for x in eval_para)
            eval_para_tensor = torch.ones(len(eval_para), max_len_10p).long()
            for i in range(len(eval_para)):
                eval_para_tensor[i, :len(eval_para[i])] = torch.LongTensor(eval_para[i])
            if max_len_10p > 512:
                eval_para_tensor = eval_para_tensor[:, :512]

            self.bkt_ptrs[bkt_id] += cur_bsz
            if self.bkt_ptrs[bkt_id] >= len(cur_bucket):
                self.bkt_pool.remove(bkt_id)

            yield {
                'ids': ids,
                'train_para': train_para_tensor,
                'target_para': torch.LongTensor(target_para),
                'eval_para': eval_para_tensor,
                'eval_para_target': eval_para_target,
                'para_num_eval': para_num_eval,
            }



def get_buckets(record_file, local_rank=None, ddp=False):
    datapoints = torch.load(record_file)
    total_rank = torch.cuda.device_count()
    len_data = len(datapoints)
    print('total len_data:', len_data)
    if ddp:
        split = len_data // total_rank
        datapoints = datapoints[split * local_rank:split * (local_rank + 1)]
        print('len_data per in gpu', local_rank, ':', len(datapoints))
    return [datapoints]

