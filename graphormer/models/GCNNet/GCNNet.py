import torch
from torch.optim import Adam
from torch_geometric.nn import GCNConv, global_mean_pool
import pytorch_lightning as pl
from lr import PolynomialDecayLR
import pytorch_warmup as warmup


class GCNNet(torch.nn.Module):
    """
    A dummy GCNNet used for testing the general training framework

    It consist of 3 GCNConv layers and a mean pooling layer
    It outputs a graph embedding
    """
    def __init__(self, input_dim, hidden_dim):
        super(GCNNet, self).__init__()
        self.gcn_conv1 = GCNConv(input_dim, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn_conv3 = GCNConv(hidden_dim, hidden_dim)


    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        h = self.gcn_conv1(x, edge_index)
        h = self.gcn_conv2(h, edge_index)
        h = self.gcn_conv3(h, edge_index)
        graph_embedding = global_mean_pool(h, batch)

        return graph_embedding


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model arguments to the parent parser
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("GCNNet")

        # Add specific model arguments below
        # E.g., parser.add_argument('--GCN_arguments', type=int,
        # default=12)

        return parent_parser


    def configure_optimizers(self, warmup_iterations, tot_iterations,
                             peak_lr, end_lr):
        """
        Returns an optimizer and scheduler suitable for GCNNet
        :return: optimizer, scheduler
        """
        optimizer = Adam(self.parameters())
        # scheduler = warmup.
        scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_iterations=warmup_iterations,
                tot_iterations=tot_iterations,
                lr=peak_lr,
                end_lr=end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return optimizer, scheduler


