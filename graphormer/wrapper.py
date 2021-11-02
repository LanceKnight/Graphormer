# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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

pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos

pattern_dict = {'[NH-]': '[N-]'}
add_atom_num = 5
num_reference = 10000 # number of reference molecules for augmentation


def smiles_cleaner(smiles):
    '''
    this function is to clean smiles for some known issues that makes rdkit:Chem.MolFromSmiles not working
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
            'smiles2grpah() needs to input D to specifiy 2D or 3D graph generation.')
    # print(f'smiles:{smiles}')
    # default RDKit behavior is to reject hypervalent P, so you need to set sanitize=False. Search keyword = 'Explicit Valence Error - Partial Sanitization' on https://www.rdkit.org/docs/Cookbook.html for more info
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
            atom_pos.append([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y])
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
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]
    #                ] = convert_to_single_emb(edge_attr) + 1

    data = Data(x=x, p=p, edge_index=edge_index, edge_attr=edge_attr)  # , adj=adj, attn_bias=attn_bias, attn_edge_type=attn_edge_type)
    # data = preprocess_item(data)
    return data


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    return item


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    @ property
    def processed_file_names(self):
        return f'{self.dataset}-{self.D}D.pt'

    def process(self):
        # super(MyPygPCQM4MDataset, self).process()
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz')).head(100)
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)

            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        dict_train = split_dict['train']
        dict_val = split_dict['valid']
        dict_test = split_dict['test']
        print(f'train len:{len(dict_train)}')
        print(f'val len:{len(dict_val)}')
        print(f'test len:{len(dict_test)}')
        # for i in dict_test[:20]:
        #     print(f'split:dict:{i}')

        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['test']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)

    def get_idx_split(self):
        split_dict = {}
        split_dict['train'] = [torch.tensor(x) for x in range(0, 80)]
        split_dict['valid'] = [torch.tensor(x) for x in range(80, 90)]
        split_dict['test'] = [torch.tensor(x) for x in range(90, 100)]
        # split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict


class MyZINCDataset(torch_geometric.datasets.ZINC):
    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyQSARDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 D=2,
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
        super(MyQSARDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

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

    @ property
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

            # only get first 100 data
            smiles_list = smiles_list

            for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
                # for i in tqdm(range(1)):
                smi = smiles_list[i]

                # data = smiles2graph(self.D, smi)
                # if data is None:
                #     continue

                # if use ogb_smiles2graph()
                try:
                    graph = ogb_smiles2graph(smi)
                except:
                    print('cannot convert smiles to graph')
                    pass

                data = Data()
                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)


                data.idx = i
                data.y = torch.tensor([label], dtype = torch.float32)
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
            self.processed_dir, f'{self.dataset}-smiles.csv'), index=False, header=False)

        # print(f'data length:{len(data_list)}')
        # for data in data_list:
        #     print(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = {}
        # total 362 actives. split: train-290, 36, 36
        split_dict['train'] = [torch.tensor(x) for x in range(0, 326)] + [torch.tensor(x) for x in range(1000, 10674)] #  training
        # split_dict['valid'] = [torch.tensor(x) for x in range(0, 290)] + [torch.tensor(x) for x in range(1000, 1510)] # 800 training
        split_dict['valid'] = [torch.tensor(x) for x in range(326, 362)] + [torch.tensor(x) for x in range(20000, 29964)] # 100 valid
        split_dict['test'] = [torch.tensor(x) for x in range(326, 362 )]+ [torch.tensor(x) for x in range(3000, 9066)] # 100 test
        # split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class AugmentedDataset(InMemoryDataset):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None, generate_num=None, empty=False    ):
        self.root = root
        self.generate_num = generate_num
        self.train_set =[]
        super(AugmentedDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def raw_file_names(self):

        return None

    @property
    def processed_file_names(self):
        return f'connect_aug.pt'

    def randomly_add_atom(self, mol, num_atom=add_atom_num, added_atomic_num = 6):
        '''randomly add x number of atoms to the molecule
        mol: a rdkit mol object
        num_atom: number of atoms to be added
        added_atomic_num: the atomic number for the new atom. Default is to add carbon, hence 6 for the atomic number
        '''

        new_mol = mol

        for i in range(num_atom):
            new_mol = Chem.RWMol(new_mol)
            #         new_atom = Chem.Atom('Cl')
            #         new_mol.AddAtom(new_atom)
            num_atom = new_mol.GetNumAtoms()
            new_atom_id = num_atom - 1

            #         print(f'total num:{num_atom}')
            invalid = True
            #         random_atom_id = 0
            #         explicit_Hs = 0
            #         implicit_Hs = 0
            #         total_Hs = 0
            while (invalid == True):
                random_atom_id = randint(0, num_atom - 1)
                # print(f'random_atom_id:{random_atom_id}')
                atom = new_mol.GetAtomWithIdx(random_atom_id)
                total_Hs = atom.GetTotalNumHs()

                #             print(f'id = {random_atom_id } symbol:{atom.GetSymbol()} total_Hs:{total_Hs} implicit_Hs:{implicit_Hs} explicit_Hs:{explicit_Hs} total_valence:{total_valence} implicit valence:{implicit_valence} explicit_val:{explicit_valence}')
                if (total_Hs > 0):
                    invalid = False
            bt = Chem.BondType.SINGLE
            #         print(f'explicit:{explicit_Hs} random_atom_id:{random_atom_id}')

            new_mol.UpdatePropertyCache()
            new_mol = Chem.AddHs(new_mol)
            atom = new_mol.GetAtomWithIdx(random_atom_id)
            for nbr in atom.GetNeighbors():
                # print(f'nbr:{nbr.GetAtomicNum()}')
                if nbr.GetAtomicNum() == 1:
                    # print('replced')
                    nbr.SetAtomicNum(added_atomic_num)
                    break
            new_mol = Chem.RemoveAllHs(new_mol)
            #             for bond in atom.GetBonds():
            #                 print(f'begin:{bond.GetBeginAtom().GetAtomicNum()} end:{bond.GetEndAtom().GetAtomicNum()}')
            #                 if (bond.GetBeginAtom().GetAtomicNum() ==1) or (bond.GetEndAtom().GetAtomicNum() ==1):
            #                     print('replaced')
            #                     nbr.SetAtomicNum(19)

            #         elif implicit_Hs>0:
            #             print(f'***num_atom-1:{new_atom_id} a2:{random_atom_id }')

            #             new_mol.AddBond(new_atom_id, random_atom_id, bt)

            try:
                Chem.SanitizeMol(new_mol)
            except Exception as e:
                print(f'generated molecule didn\'t pass sanitization test!atom_id:{new_atom_id}-{e}')
        return new_mol

    def generate_2D_molecule_from_reference(self, smiles, num):
        '''generate molecules with similar connectivity with the reference molecule, input smiles, output mol
        smiles: input molecule
        num: number of augmented molecules to generate
        '''
        mol = Chem.MolFromSmiles(smiles)


        output_list = []
        for i in range(num):
            new_mol = self.randomly_add_atom(mol, 2, 6)

                # graph_dict = ogb_smiles2graph(smiles)
            output_list.append(new_mol)
        return output_list


    def process(self):
        file ='../../dataset/connect_aug/raw/smiles.csv'
        raw_list = pd.read_csv(file, header=None)[0].tolist()[:num_reference]

        # raw_list = ['C1(=CC=CC(=C1)C(CC)C)O', 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5']
        data_list = []
        for idx, smi in tqdm(enumerate(raw_list)):
            # print(smi)
            try:
                graph = ogb_smiles2graph(smi)

                data = Data()
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)

                data.labels = idx
                data.y = torch.tensor([idx])
                data.root_smiles = smi
                data.is_root = True
                data_list.append(data)
                # print(f'setup data:{data}')
                augmented_list = [ogb_smiles2graph(Chem.MolToSmiles(mol)) for mol in self.generate_2D_molecule_from_reference(smi, self.generate_num)]

                for aug_graph in augmented_list:
                    data = Data()
                    data.x = torch.from_numpy(aug_graph['node_feat']).to(torch.int64)
                    data.edge_index = torch.from_numpy(aug_graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(aug_graph['edge_feat']).to(torch.int64)
                    data.labels = idx
                    data.y = torch.tensor([idx])
                    # if smi == 'C1(=CC=CC(=C1)C(CC)C)O':
                    #     data.root_smiles = 'short'
                    # else:
                    #     data.root_smiles = 'long'
                    data.is_root = False
                    data_list.append(data)
            except:
                pass

        # print('data_list')
        # for item in data_list:
        #     print(f'{item}')

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            # print('triggered')
            return preprocess_item(item)
        else:
            return self.index_select(idx)

if __name__ == "__main__":
    dataset = AugmentedDataset(root = '../../dataset/connect_aug/', generate_num=5)





