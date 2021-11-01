# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from model import Graphormer, SelfSupervisedGraphormer
from data import GraphDataModule, get_dataset, AugmentedDataModule
from monitors import LogAUCMonitor, LossMonitor, PPVMonitor

from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from clearml import Task


def cli_main(logger):
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Graphormer.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    # print(f'args.max_steps:{args.max_steps}')
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)
    augmented_dataset = AugmentedDataModule.from_argparse_args(args)
    # ------------
    # model
    # ------------
    print(f'=========================')
    print(f'checkpoint_path:{args.checkpoint_path}')
    if args.checkpoint_path != '':
        print('loading pretrain model')
        model = Graphormer.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )

        # pretrain_model = SelfSupervisedGraphormer(
        #     args.checkpoint_path,
        #     strict = False,
        #     n_layers=args.n_layers,
        #     num_heads=args.num_heads,
        #     hidden_dim=args.hidden_dim,
        #     attention_dropout_rate=args.attention_dropout_rate,
        #     dropout_rate=args.dropout_rate,
        #     intput_dropout_rate=args.intput_dropout_rate,
        #     weight_decay=args.weight_decay,
        #     ffn_dim=args.ffn_dim,
        #     dataset_name=dm.dataset_name,
        #     warmup_updates=args.warmup_updates,
        #     tot_updates=args.tot_updates,
        #     peak_lr=args.peak_lr,
        #     end_lr=args.end_lr,
        #     edge_type=args.edge_type,
        #     multi_hop_max_dist=args.multi_hop_max_dist,
        #     flag=args.flag,
        #     flag_m=args.flag_m,
        #     flag_step_size=args.flag_step_size,
        # )
    else:
        print(f'option2')
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    pretrain_model = SelfSupervisedGraphormer(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        attention_dropout_rate=args.attention_dropout_rate,
        dropout_rate=args.dropout_rate,
        intput_dropout_rate=args.intput_dropout_rate,
        weight_decay=args.weight_decay,
        ffn_dim=args.ffn_dim,
        dataset_name=dm.dataset_name,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        peak_lr=args.peak_lr,
        end_lr=args.end_lr,
        edge_type=args.edge_type,
        multi_hop_max_dist=args.multi_hop_max_dist,
        flag=args.flag,
        flag_m=args.flag_m,
        flag_step_size=args.flag_step_size,
    )
    if not args.test and not args.validate:
        print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # ------------
    # training
    # ------------
    metric = 'valid_' + get_dataset(dm.dataset_name)['metric']
    dirpath = args.default_root_dir + f'/lightning_logs/checkpoints'

    pretrain_checkpoint_callback = ModelCheckpoint(
        # monitor=metric,
        dirpath=dirpath+'/pretrain',
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        # save_top_k=100,
        # mode=get_dataset(dm.dataset_name)['metric_mode'],
        save_last=True,
    )

    checkpoint_callback = ModelCheckpoint(
        # monitor=metric,
        dirpath=dirpath,
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        # save_top_k=100,
        # mode=get_dataset(dm.dataset_name)['metric_mode'],
        save_last=True,
    )
    # if os.path.exists(dirpath + '/pretrain/last.ckpt'):
    #     print('pretraining checkpoint exists, resuming checkpoint')
    #     args.resume_from_checkpoint = dirpath + '/pretrain/last.ckpt'
        # print('pretraining args.resume_from_checkpoint', args.resume_from_checkpoint)


    self_supervised_trainer = pl.Trainer.from_argparse_args(args)
    self_supervised_trainer.callbacks.append(pretrain_checkpoint_callback)
    self_supervised_trainer.callbacks.append(LossMonitor(stage='train', logger=logger, logging_interval='step', title='pretrain_'))
    self_supervised_trainer.callbacks.append(LossMonitor(stage='train', logger=logger, logging_interval='epoch', title='pretrain_'))
    self_supervised_trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))
    # self_supervised_trainer.fit(pretrain_model, augmented_dataset)



    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt'):
        print('actual checkpoint exists')
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('actual training args.resume_from_checkpoint', args.resume_from_checkpoint)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LossMonitor(stage='train', logger=logger, logging_interval='step'))
    trainer.callbacks.append(LossMonitor(stage='train', logger=logger, logging_interval='epoch'))
    trainer.callbacks.append(LogAUCMonitor(stage='train', logger=logger, logging_interval='epoch'))
    trainer.callbacks.append(PPVMonitor(stage='train', logger=logger, logging_interval='epoch'))
    trainer.callbacks.append(LogAUCMonitor(stage='valid',logger=logger, logging_interval='epoch'))
    trainer.callbacks.append(PPVMonitor(stage='valid',logger=logger, logging_interval='epoch'))
    trainer.callbacks.append(LossMonitor(stage='valid', logger=logger, logging_interval='step'))
    trainer.callbacks.append(LossMonitor(stage='valid', logger=logger, logging_interval='epoch'))
    # trainer.callbacks.append(LogAUCMonitor(stage='train', logger=logger, logging_interval='step'))
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))


    if args.test:
        print(f'testing$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.validate:
        print(f'validating$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:
        print(f'training$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        trainer.fit(model, datamodule=dm)


task = Task.init(project_name="Tests/Graphormer", task_name="pretrain test", tags=["graphormer", "debug", "qsar","pretrain"])

if __name__ == '__main__':
    logger = task.get_logger()
    # logger.report_scalar(title='just a test', series='test', value=1, iteration=epoch)
    cli_main(logger)
