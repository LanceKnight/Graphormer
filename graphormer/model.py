from models.GCNNet.GCNNet import GCNNet
from evaluation import calculate_logAUC, calculate_ppv

# Public libraries
import pytorch_lightning as pl
from torch.nn import Linear, Sigmoid, BCEWithLogitsLoss
from torch_geometric.data import Data


class GNNModel(pl.LightningModule):
    """
    A wrapper for different GNN models

    It uses a GNN model to output a graph embedding, and use some prediction
    method to output a final prediction

    Here a linear layer with a sigmoid function is used
    """

    def __init__(self,
                 gnn_type,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 warmup_iterations,
                 tot_iterations,
                 peak_lr,
                 end_lr,
                 loss_func=BCEWithLogitsLoss()):
        super(GNNModel, self).__init__()
        if gnn_type == 'gcn':
            self.gnn_model = GCNNet(input_dim, hidden_dim)
        else:
            raise ValueError("model.py::GNNModel: GNN model type is not "
                             "defined.")

        self.linear = Linear(hidden_dim, output_dim)
        self.activate_func = Sigmoid()  # Not used
        self.warmup_iterations = warmup_iterations
        self.tot_iterations = tot_iterations
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.loss_func = loss_func

    def forward(self, data):
        graph_embedding = self.gnn_model(data)
        prediction = self.linear(graph_embedding)

        return prediction, graph_embedding

    @staticmethod
    def add_model_args(gnn_type, parent_parser):
        """
        Add model arguments to the parent parser
        :param gnn_type: a lowercase string specifying GNN type
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("GNN_Model")

        # Add general model arguments below
        # E.g., parser.add_argument('--general_model_args', type=int,
        # default=12)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--input_dim', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=32)
        parser.add_argument('--output_dim', type=int, default=32)
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--warmup_iterations', type=int, default=60000)
        parser.add_argument('--tot_iterations', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)

        if gnn_type == 'gcn':
            GCNNet.add_model_specific_args(parent_parser)
        return parent_parser

    def training_step(self, batch_data, batch_idx):
        # Get prediction and ground truth
        prediction, _ = self(batch_data)
        prediction = prediction.view(-1)
        y = batch_data.y.view(-1)

        # Get metrics
        loss = self.loss_func(prediction, y.float())
        numpy_prediction = prediction.detach().cpu().numpy()
        numpy_y = y.cpu().numpy()
        logAUC = calculate_logAUC(numpy_y, numpy_prediction)
        ppv = calculate_ppv(numpy_y, numpy_prediction)

        return {"loss": loss, "logAUC": logAUC, "ppv":ppv}

    def training_epoch_end(self, train_step_outputs):
        train_epoch_outputs={}
        for key in train_step_outputs[0].keys():
            mean_output = sum(output[key] for output in train_step_outputs) \
                          / len(train_step_outputs)
            train_epoch_outputs[key] = mean_output

        self.train_epoch_outputs =train_epoch_outputs

    def configure_optimizers(self):
        optimizer, scheduler = self.gnn_model.configure_optimizers(
            self.warmup_iterations, self.tot_iterations, self.peak_lr,
            self.end_lr)
        return [optimizer], [scheduler]
