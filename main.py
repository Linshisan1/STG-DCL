import json
from transformers import RobertaTokenizer
from transformers import BertTokenizer
import numpy as np
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm import trange
import datetime
import os, random

from utils.common_utils import Logging

from utils.data_utils import load_data_instances
from data.data_preparing import DataIterator

from modules.models.roberta import Model
from modules.f_loss import FocalLoss
from tools.trainer import Trainer


if __name__ == '__main__':
    # 如果有多块 GPU，可设置 device id
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    # ======== 1) Base model & data settings ===========
    parser.add_argument('--model_name_or_path', type=str, default='./modules/models/roberta-base',
                        choices=["roberta-base","bert-base-uncased"],
                        help='Path to pretrained model or shortcut name')
    parser.add_argument('--model_cache_dir', type=str, default='./modules/models/',
                        help='Directory to cache pretrained models')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of the tagging matrix')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained BERT/Roberta embedding')
    parser.add_argument('--class_num', type=int, default=5,
                        help='e.g. N,CTD,POS,NEU,NEG => 5 classes')
    parser.add_argument('--task', type=str, default="triplet",choices=["pair","triplet"],
                        help='downstream task name, for logs usage')

    # ======== 2) Data & logging settings ==============
    parser.add_argument('--prefix', type=str, default="./data/",
                        help='dataset path prefix')
    parser.add_argument('--data_version', type=str, default="D2",choices=["D1","D2"],
                        help='which version: D1 or D2')
    parser.add_argument('--dataset', type=str, default="lap14",
                        choices=["res14","lap14","res15","res16"],
                        help='dataset name')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training & evaluating')
    parser.add_argument('--epochs', type=int, default=800,
                        help='training epoch number')
    parser.add_argument('--log_path', type=str, default=None,
                        help='output log file path. if None => auto-generate')

    # ======== 3) Contrastive hyperparams =============
    parser.add_argument('--lambda_scl', type=float, default=1e-4,
                        help='scale for supervised contrast loss (SCL)')
    parser.add_argument('--lambda_syntax', type=float, default=1e-7,
                        help='scale for syntax adjacency InfoNCE')

    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for contrastive')

    # ======== 4) Misc & device =======================
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu device to use')
    parser.add_argument('--model_save_dir', type=str,
                        default="./modules/models/saved_models/",
                        help='where to save best checkpoint')
    parser.add_argument('--sentiment2id', type=dict,
                        default={'negative':2,'neutral':3,'positive':4},
                        help='mapping sentiments to ids')
    parser.add_argument('--num_hard_positive', type=int, default=3, help='Number of hard positives per anchor for SCL')
    parser.add_argument('--num_hard_negative', type=int, default=10, help='Number of hard negatives per anchor for SCL')
    parser.add_argument('--alpha_dep', type=float, default=1, help='Alpha for dependency distance weighting')
    parser.add_argument('--dep_threshold', type=int, default=3, help='Dependency distance threshold')

    parser.add_argument('--use_scl', type=bool, default=False)
    parser.add_argument('--use_syntaxcl', type=bool, default=False)

    args = parser.parse_known_args()[0]

    # 如果未指定日志文件 => 自动生成
    if args.log_path is None:
        args.log_path = 'log_{}_{}_{}.log'.format(
            args.data_version,
            args.dataset,
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        )

    # 初始化 tokenizer
    # tokenizer = RobertaTokenizer.from_pretrained(
    #     args.model_name_or_path,
    #     cache_dir=args.model_cache_dir,
    #     local_files_only=True,
    #     force_download=False,
    #     lowercase=True
    # )

    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.model_cache_dir,
        local_files_only=True,
        force_download=False,
        lowercase=True
    )
    # 初始化日志
    logging = Logging(file_name=args.log_path).logging

    # 固定随机种子 => 确保可复现
    def seed_torch(seed):

        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        random.seed(seed)


    seed1 = random.randint(0, 2**32 - 1)
    seed2 = 999
    print(f"🌱 当前自动生成的随机种子是: {seed1}")
    seed_torch(1200931)


    # ======= 5) Load Dataset ==========================
    train_sentence_packs = json.load(open(
        os.path.join(args.prefix, args.data_version, args.dataset, 'train.json'),
        encoding="utf-8"
    ))
    dev_sentence_packs = json.load(open(
        os.path.join(args.prefix, args.data_version, args.dataset, 'dev.json'),
        encoding="utf-8"
    ))
    test_sentence_packs= json.load(open(
        os.path.join(args.prefix, args.data_version, args.dataset, 'test.json'),
        encoding="utf-8"
    ))

    # parse to instance
    from utils.data_utils import load_data_instances
    train_instances = load_data_instances(tokenizer, train_sentence_packs, args)
    dev_instances   = load_data_instances(tokenizer, dev_sentence_packs, args)
    test_instances  = load_data_instances(tokenizer, test_sentence_packs, args)

    trainset = DataIterator(train_instances, args)
    devset   = DataIterator(dev_instances, args)
    testset  = DataIterator(test_instances, args)

    # ======= 6) Initialize Model & Optimizer =========
    from modules.models.roberta import Model
    model = Model(args).to(args.device)

    # FocalLoss => multi-class + binary
    from modules.f_loss import FocalLoss
    weight_mc = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0]).float().to(args.device)
    f_loss_mc = FocalLoss(weight_mc, ignore_index=-1)

    weight_bin= torch.tensor([1.0,2.0]).float().to(args.device)
    f_loss_bin= FocalLoss(weight_bin, ignore_index=-1)

    # 两个Loss => (f_loss, f_loss1)
    criterion = (f_loss_mc, f_loss_bin)

    # Adam =>
    #   group0 => model.bert => lr=1e-5
    #   group1 => new layers => lr=1e-4
    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(),        'lr':1e-5},
        {'params': model.linear1.parameters(),     'lr':1e-4},
        {'params': model.cls_linear.parameters(),  'lr':1e-4},
        {'params': model.cls_linear1.parameters(), 'lr':1e-4},
    ], lr=1e-4)

    # 学习率调度 => milestones => [10,20,30,40], gamma=0.5 ...
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[150, 250, 350, 450, 500],
        gamma=0.5
    )

    # ======= 7) Setup trainer & run ==================
    from tools.trainer import Trainer
    # beta_1 => scale for second head
    # beta_2 => can be 0 if not used
    beta_1 = 1.0
    beta_2 = 0.0
    bear_max=10
    last=10

    trainer = Trainer(
        model, trainset, devset, testset,
        optimizer, criterion, lr_scheduler,
        args, logging,
        beta_1, beta_2,
        bear_max, last,
        plot=False
    )
    logging(f"""
    ========= - * - =========
    date: {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    seed:{1200931}
    dataset: {args.data_version}/{args.dataset}
    beta_1: {beta_1}
    beta_2: {beta_2}
    lambda_scl：: {args.lambda_scl}
    lambda_syntax: {args.lambda_syntax}
    lr_scheduler: {lr_scheduler}
    lr_scheduler milestones: {lr_scheduler.milestones}
    lr_scheduler gamma: {lr_scheduler.gamma}
    lr_bert: {optimizer.param_groups[0]['lr']}
    lr_linear1: {optimizer.param_groups[1]['lr']}
    lr_cls_linear: {optimizer.param_groups[2]['lr']}
    lr_cls_linear1: {optimizer.param_groups[3]['lr']}
    ========= - * - =========
    """)
    trainer.train()
