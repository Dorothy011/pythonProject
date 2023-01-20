import os
from run_para import train_para, test_para
import argparse


parser = argparse.ArgumentParser()

word_emb_file = "word_emb.json"
char_emb_file = "char_emb.json"
train_eval = "../data/train_eval.json"
dev_eval = "../data/dev_eval.json"
test_eval = "../data/dev_eval.json"
train_record_file = '../data/train_record.pkl'
dev_record_file = '../data/dev_record.pkl'
test_record_file = '../data/dev_record.pkl'

parser.add_argument('--mode', type=str, default='train_para', help='train_para or test_para')
parser.add_argument('--save', type=str, default='HOTPOT', help='save folder name')

parser.add_argument('--train_eval_file', type=str, default=train_eval)
parser.add_argument('--dev_eval_file', type=str, default=dev_eval)
parser.add_argument('--test_eval_file', type=str, default=test_eval)
parser.add_argument('--train_record_file', type=str, default=train_record_file)
parser.add_argument('--dev_record_file', type=str, default=dev_record_file)
parser.add_argument('--test_record_file', type=str, default=test_record_file)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--checkpoint', type=int, default=1000, help='steps to save checkpoint')
parser.add_argument('--period', type=int, default=100, help='steps to print')
parser.add_argument('--init_lr', type=float, default=1e-5)
parser.add_argument('--hidden', type=int, default=1024)
parser.add_argument('--seed', type=int, default=13, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='for apex parallel')

parser.add_argument('--warm_up_step', type=int, default=1000, help='lr warm up steps')
parser.add_argument('--decay_lr', type=float, default=0.82e-10, help='lr decay at each training step')
parser.add_argument('--eval_last_rate', type=float, default=0.3, help='only ev print("predict_paras:", predict_paras)al during last xx% training steps')

config = parser.parse_args()

if config.mode == 'train_para':
    train_para(config)
elif config.mode == 'test_para':
    test_para(config)
