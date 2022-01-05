import os
import torch
import numpy as np
import torch_geometric.datasets
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
import pandas as pd
import os.path as osp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from ogb.utils import smiles2graph as ogb_smiles2graph
from ogb.utils.features import atom_to_feature_vector
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from random import randint

# pyximport.install(setup_args={'include_dirs': np.get_include()})
# import algos

pattern_dict = {'[NH-]': '[N-]'}
add_atom_num = 5
num_reference = 10000  # number of reference molecules for augmentation


def smiles_cleaner(smiles):
    '''
    this function is to clean smiles for some known issues that makes
    rdkit:Chem.MolFromSmiles not working
    '''
    print('fixing smiles for rdkit...')
    new_smiles = smiles
    for pattern, replace_value in pattern_dict.items():
        if pattern in smiles:
            print('found pattern and fixed the smiles!')
            new_smiles = smiles.replace(pattern, replace_value)
    return new_smiles


def generate_element_rep_list(elements):
    print('calculating rdkit element representation lookup table')
    elem_rep_lookup = []
    for elem in elements:
        pt = Chem.GetPeriodicTable()

        if isinstance(elem, int):
            num = elem
            sym = pt.GetElementSymbol(num)
        else:
            num = pt.GetAtomicNumber(elem)
            sym = elem
        w = pt.GetAtomicWeight(elem)

        Rvdw = pt.GetRvdw(elem)
        #     Rcoval = pt.GetRCovalent(elem)
        valence = pt.GetDefaultValence(elem)
        outer_elec = pt.GetNOuterElecs(elem)

        elem_rep = [num, w, Rvdw, valence, outer_elec]
        #             print(elem_rep)

        elem_rep_lookup.append(elem_rep)
    elem_lst = elem_rep_lookup.copy()
    return elem_rep_lookup


max_elem_num = 118
element_nums = [x + 1 for x in range(max_elem_num)]
elem_lst = generate_element_rep_list(element_nums)


def get_atom_rep(atomic_num):
    '''use rdkit to generate atom representation
    '''
    global elem_lst

    result = 0
    try:
        result = elem_lst[atomic_num - 1]
    except:
        print(f'error: atomic_num {atomic_num} does not exist')

    return result


def smiles2graph(D, smiles):
    if D == None:
        raise Exception(
            'smiles2grpah() needs to input D to specifiy 2D or 3D graph '
            'generation.')
    # print(f'smiles:{smiles}')
    # Default RDKit behavior is to reject hypervalent P, so you need to set
    # sanitize=False. Search keyword = 'Explicit Valence Error - Partial
    # Sanitization' on https://www.rdkit.org/docs/Cookbook.html for more info
    smiles = smiles.replace(r'/=', '=')
    smiles = smiles.replace(r'\=', '=')
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as e:
        print(f'Cannot generate mol, error:{e}, smiles:{smiles}')

    if mol is None:
        smiles = smiles_cleaner(smiles)
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except Exception as e:
            print(f'Generated mol is None, error:{e}, smiles:{smiles}')
            return None
    try:
        # mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f'{e}, smiles:{smiles}')

    if D == 2:
        Chem.rdDepictor.Compute2DCoords(mol)
    if D == 3:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    atom_pos = []
    atom_attr = []

    # get atom attributes and positions
    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        h = get_atom_rep(atomic_num)

        if D == 2:
            atom_pos.append(
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y])
        elif D == 3:
            atom_pos.append([conf.GetAtomPosition(
                i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
        atom_attr.append(h)

    # get bond attributes
    edge_list = []
    edge_attr_list = []
    for idx, edge in enumerate(mol.GetBonds()):
        i = edge.GetBeginAtomIdx()
        j = edge.GetEndAtomIdx()

        bond_attr = None
        bond_type = edge.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_attr = [1]
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_attr = [2]
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_attr = [3]
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_attr = [4]

        edge_list.append((i, j))
        edge_attr_list.append(bond_attr)
        #         print(f'i:{i} j:{j} bond_attr:{bond_attr}')

        edge_list.append((j, i))
        edge_attr_list.append(bond_attr)
    #         print(f'j:{j} j:{i} bond_attr:{bond_attr}')

    x = torch.tensor(atom_attr)
    p = torch.tensor(atom_pos)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list)

    # graphormer-specific features
    # adj = torch.zeros([N, N], dtype=torch.bool)
    # adj[edge_index[0, :], edge_index[1, :]] = True
    # attn_bias = torch.zeros(
    #     [N + 1, N + 1], dtype=torch.float)  # with graph token
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)],
    # dtype=torch.long)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]
    #                ] = convert_to_single_emb(edge_attr) + 1

    data = Data(x=x, p=p, edge_index=edge_index,
                edge_attr=edge_attr)  # , adj=adj, attn_bias=attn_bias,
    # attn_edge_type=attn_edge_type)
    # data = preprocess_item(data)
    return data


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset,
                                  dtype=torch.long)
    x = x + feature_offset
    return x




class QSARDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 D=3,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='435008',
                 empty=False):

        self.dataset = dataset
        self.root = root
        self.D = D
        super(QSARDataset, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, \
                                                              pre_transform,\
                                                              pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    # def get(self, idx):
    #     data = Data()
    #     for key in self.data.keys:
    #         item, slices = self.data[key], self.slices[key]
    #         s = list(repeat(slice(None), item.dim()))
    #         s[data.__cat_dim__(key, item)] = slice(slices[idx],
    #                                                 slices[idx + 1])
    #         data[key] = item[s]
    #     return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return f'{self.dataset}-{self.D}D.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    # def __getitem__(self, index):
    #     return self.get(index)

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset not in ['435008', '1798', '435034']:
            # print(f'dataset:{self.dataset}')
            raise ValueError('Invalid dataset name')

        for file, label in [(f'{self.dataset}_actives.smi', 1),
                            (f'{self.dataset}_inactives.smi', 0)]:
            smiles_path = os.path.join(self.root, 'raw', file)
            smiles_list = pd.read_csv(
                smiles_path, sep='\t', header=None)[0]

            # Only get first N data, just for debugging
            smiles_list = smiles_list[0:4000]

            for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
                # for i in tqdm(range(1)):
                smi = smiles_list[i]

                data = smiles2graph(self.D, smi)
                if data is None:
                    continue

                # # If use ogb_smiles2graph()
                # try:
                #     graph = ogb_smiles2graph(smi)
                # except:
                #     print('cannot convert smiles to graph')
                #     pass

                # data = Data()
                # data.__num_nodes__ = int(graph['num_nodes'])
                # data.edge_index = torch.from_numpy(graph['edge_index']).to(
                #     torch.int64)
                # data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                #     torch.int64)
                # data.x = torch.from_numpy(graph['node_feat']).to(torch.float32)

                data.idx = i
                data.y = torch.tensor([label], dtype=torch.int)
                data.smiles = smi

                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print('doing pre_transforming...')
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(
            self.processed_dir, f'{self.dataset}-smiles.csv'), index=False,
            header=False)

        # print(f'data length:{len(data_list)}')
        # for data in data_list:
        #     print(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = {}
        # # Total 362 actives. Split: train-290, 36, 36
        # split_dict['train'] = [torch.tensor(x) for x in range(0, 326)] + [
        #     torch.tensor(x) for x in range(1000, 10674)]  # 10K Training
        # split_dict['valid'] = [torch.tensor(x) for x in range(326, 362)] + [
        #     torch.tensor(x) for x in range(20000, 29964)]  # 10K val
        # split_dict['test'] = [torch.tensor(x) for x in range(326, 362)] + [
        #     torch.tensor(x) for x in range(3000, 9066)]

        # Super small dataset for processing debugging.
        # Total 362 actives. Split: 290, 36, 36
        split_dict['train'] = [torch.tensor(x) for x in range(0, 326)] + [
            torch.tensor(x) for x in range(400, 1074)]  # 1K Training
        split_dict['valid'] = [torch.tensor(x) for x in range(326, 362)] + [
            torch.tensor(x) for x in range(1100, 2064)]  # 1K val
        split_dict['test'] = [torch.tensor(x) for x in range(326, 362)] + [
            torch.tensor(x) for x in range(3000, 4000)]

        return split_dict

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return item
        else:
            return self.index_select(idx)



if __name__ == "__main__":
    pass
    # dataset = MyQSARDataset(root='../../dataset/connect_aug/',
    #                            generate_num=5)
