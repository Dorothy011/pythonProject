import os
from run_span import train, test
import argparse

parser = argparse.ArgumentParser()

train_eval = "../data/train_eval.json"
dev_eval = "../data/dev_eval.json"
test_eval = "../data/dev_eval.json"
train_record_file = '../data/train_record.pkl'  # for training
dev_record_file = '../data/dev_record.pkl'  # for evaluating during training
test_record_file = '../data/dev_record.pkl'  # for evaluating after training

parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--save', type=str, default='HOTPOT', help='save folder name')

parser.add_argument('--train_eval_file', type=str, default=train_eval)
parser.add_argument('--dev_eval_file', type=str, default=dev_eval)
parser.add_argument('--test_eval_file', type=str, default=test_eval)
parser.add_argument('--train_record_file', type=str, default=train_record_file)
parser.add_argument('--dev_record_file', type=str, default=dev_record_file)
parser.add_argument('--test_record_file', type=str, default=test_record_file)
parser.add_argument('--prediction_file', type=str, default='pred.json', help='final prediction file')

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--checkpoint', type=int, default=1000, help='steps to save checkpoint')
parser.add_argument('--period', type=int, default=100, help='steps to print')
parser.add_argument('--init_lr', type=float, default=0.5)
parser.add_argument('--hidden', type=int, default=80)
parser.add_argument('--seed', type=int, default=13, help='random seed')
parser.add_argument('--sent_limit', type=int, default=25, help='max number of sentences')
parser.add_argument('--sp_threshold', type=float, default=0.43, help='threshold for supporting fact prediction')
parser.add_argument('--local_rank', type=int, default=0, help='for apex parallel')
parser.add_argument('--frozen_step', type=int, default=1000, help='frozen BERT layers at the begining steps')
parser.add_argument('--frozen_lr', type=float, default=5e-5, help='lr for other layers when frozing BERT layers')
parser.add_argument('--warm_up_step', type=int, default=1000, help='lr warm up steps')
parser.add_argument('--decay_lr', type=float, default=1.1e-10, help='lr decay at each training step')
parser.add_argument('--eval_last_rate', type=float, default=0.3, help='only eval after these steps')

parser.add_argument('--alpha', type=float, default=0.7, help='probability of transcend')
parser.add_argument('--num_hop', type=int, default=2, help='multi hop')
parser.add_argument('--num_heads', type=int, default=1, help='num of attention head')
parser.add_argument('--cl1', type=float, default=0.0, help='weight for span contrast loss')
parser.add_argument('--cl2', type=float, default=0.0, help='weight for sup contrast loss')
parser.add_argument('--cl3', type=float, default=0.0, help='weight for sup contrast loss')

config = parser.parse_args()

if config.mode == 'train':
    train(config)
elif config.mode == 'test':
    test(config)
