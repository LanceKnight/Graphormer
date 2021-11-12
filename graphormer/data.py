# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collator import collator
from wrapper import MyGraphPropPredDataset, MyPygPCQM4MDataset, MyZINCDataset, MyQSARDataset, AugmentedDataset

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
# from  torch_geometric.data import DataLoader as PyGDataLoader
import ogb
import ogb.lsc
import ogb.graphproppred
from functools import partial

aug_num = 5
dataset = None


def get_dataset(dataset_name='abaaba'):
    global dataset
    if dataset is not None:
        return dataset

    # max_node is set to max(max(num_val_graph_nodes), max(num_test_graph_nodes))
    # if dataset_name == 'ogbg-molpcba':
    #     dataset = {
    #         'num_class': 128,
    #         'loss_fn': F.binary_cross_entropy_with_logits,
    #         'metric': 'ap',
    #         'metric_mode': 'max',
    #         'evaluator': ogb.graphproppred.Evaluator('ogbg-molpcba'),
    #         'dataset': MyGraphPropPredDataset('ogbg-molpcba', root='../../dataset'),
    #         'max_node': 128,
    #     }
    # elif dataset_name == 'ogbg-molhiv':
    #     dataset = {
    #         'num_class': 1,
    #         'loss_fn': F.binary_cross_entropy_with_logits,
    #         'metric': 'rocauc',
    #         'metric_mode': 'max',
    #         'evaluator': ogb.graphproppred.Evaluator('ogbg-molhiv'),
    #         'dataset': MyGraphPropPredDataset('ogbg-molhiv', root='../../dataset'),
    #         'max_node': 128,
    #     }
    # elif dataset_name == 'PCQM4M-LSC':
    #     dataset = {
    #         'num_class': 1,
    #         'loss_fn': F.l1_loss,
    #         'metric': 'mae',
    #         'metric_mode': 'min',
    #         'evaluator': ogb.lsc.PCQM4MEvaluator(),
    #         'dataset': MyPygPCQM4MDataset(root='../../dataset')[:1000],
    #         'max_node': 128,
    #     }
    # elif dataset_name == 'ZINC':
    #     dataset = {
    #         'num_class': 1,
    #         'loss_fn': F.l1_loss,
    #         'metric': 'mae',
    #         'metric_mode': 'min',
    #         'evaluator': ogb.lsc.PCQM4MEvaluator(),  # same objective function, so reuse it
    #         'train_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='train'),
    #         'valid_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='val'),
    #         'test_dataset': MyZINCDataset(subset=True, root='../../dataset/pyg_zinc', split='test'),
    #         'max_node': 128,
    #     }
    if dataset_name in ['435008', '1798', '435034']:
        dataset = {
            'num_class': 1,
            'loss_fn': BCEWithLogitsLoss(),
            'metric': 'LogAUC',
            # 'metric_mode': 'min',
            # 'evaluator': ogb.lsc.PCQM4MEvaluator(),  # same objective function, so reuse it
            'dataset': MyQSARDataset(root='../../dataset/qsar', dataset=dataset_name),
            'num_samples': len(MyQSARDataset(root='../../dataset/qsar', dataset=dataset_name)),
            'max_node':512
        }

    else:
        print(f'dataset_name:{dataset_name}')
        raise NotImplementedError

    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset

# graph dataset
class GraphDataModule(LightningDataModule):
    name = "OGB-GRAPH"

    def __init__(
        self,
        dataset_name: str = 'ogbg-molpcba',
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        spatial_pos_max: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def setup(self, stage: str = None):
        if self.dataset_name == 'ZINC':
            self.dataset_train = self.dataset['train_dataset']
            self.dataset_val = self.dataset['valid_dataset']
            self.dataset_test = self.dataset['test_dataset']
        else:
            split_idx = self.dataset['dataset'].get_idx_split()
            self.dataset_train = self.dataset['dataset'][split_idx["train"]]
            print(f'training len:{len(self.dataset_train)})')
            self.dataset_val = self.dataset['dataset'][split_idx["valid"]]
            print(f'validation len:{len(self.dataset_val)})')
            self.dataset_test = self.dataset['dataset'][split_idx["test"]]
            print(f'testing len:{len(self.dataset_test)})')

    def train_dataloader(self):
        num_train_active = len(torch.nonzero(torch.tensor([data.y for data in self.dataset_train])))
        num_train_inactive = len(self.dataset_train) - num_train_active
        print(f'training size: {len(self.dataset_train)}, actives: {num_train_active}')

        train_sampler_weight = torch.tensor([(1. / num_train_inactive) if data.y == 0 else (1. / num_train_active) for data in self.dataset_train])

        train_sampler = WeightedRandomSampler(train_sampler_weight, len(train_sampler_weight))


        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            # sampler= train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(train_dataloader)', len(loader))
        counter = 0
        for batch in loader:
            print(batch.y)
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(val_dataloader)', len(val_loader))
        train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            # sampler= train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        return val_loader,self.train_dataloader()

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(test_dataloader)', len(loader))
        for batch in loader:
            print(batch.y)
        return loader

class AugmentedDataModule(LightningDataModule):
    def __init__(self,
            dataset_name: str = 'qsar',
             num_workers: int = 0,
             batch_size: int = 256,
             seed: int = 42,
             multi_hop_max_dist: int = 5,
             spatial_pos_max: int = 1024,
             generate_num = aug_num,
             *args,
             **kwargs,
            ):
        super(AugmentedDataModule, self).__init__()
        self.batch_size = batch_size
        self.generate_num = generate_num
        self.metric = 'loss'
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max


    def setup(self, stage= None):
        self.train_set = AugmentedDataset(root = '../../dataset/connect_aug', generate_num=self.generate_num)
        self.val_set = self.train_set[:2]
        print(f'len(trainset):{len(self.train_set)} len(val_set):{len(self.val_set)}')


    def train_dataloader(self):
        print('train loader here')
        loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=partial(collator, max_node=38, multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),)
        # for batch in loader:
        #     print(batch)
        return loader

    def val_dataloader(self):
        print('pretrain validation loader here')
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=partial(collator, max_node=38, multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),)
        return val_loader

        # # test set
        # smi = 'C1(=CC=CC(=C1)C(CC)C)O'
        # data1 = smiles2graph(2, smi)
        # smi = 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'
        # data2 = smiles2graph(2, smi)
        # smi = 'C1=CC=CC(=C1)C(CC)C'
        # data3 = smiles2graph(2, smi)
        # dataset = [data1, data2, data3]
        #
        # self.test_set = dataset
