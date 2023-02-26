import collections
from math import ceil
from collections import OrderedDict
import functools
import paddle
from paddle import nn
from gn_block import Processor
import pgl
import time


def knn_interpolate(features, coarse_nodes, fine_nodes):
    coarse_nodes_input = paddle.repeat_interleave(coarse_nodes.unsqueeze(0), fine_nodes.shape[0], 0)  # [6684,352,2]
    fine_nodes_input = paddle.repeat_interleave(fine_nodes.unsqueeze(1), coarse_nodes.shape[0], 1)  # [6684,352,2]
    dist_w = 1.0 / (paddle.norm(x=coarse_nodes_input - fine_nodes_input, p=2, axis=-1) + 1e-9)  # [6684,352]
    knn_value, knn_index = paddle.topk(dist_w, k=3, largest=True)  # [6684,3],[6684,3]
    weight = knn_value.unsqueeze(-2)
    features_input = features[knn_index]
    output = paddle.bmm(weight, features_input).squeeze(-2) / paddle.sum(knn_value, axis=-1, keepdim=True)
    return output


def MyCopy(graph):
    data = pgl.Graph(num_nodes=graph.num_nodes,
                        edges=graph.edges,)
    data.x = graph.x
    data.y = graph.y
    data.pos = graph.pos
    data.edge_index = graph.edge_index
    data.edge_attr = graph.edge_attr
    return data

def Myadd(g1, g2):
    g1.x = paddle.concat([g1.x, g2.x], axis=0)
    g1.y = paddle.concat([g1.y, g2.y], axis=0)
    g1.edge_index = paddle.concat([g1.edge_index,g2.edge_index], axis=1) 
    g1.edge_attr =  paddle.concat([g1.edge_attr, g2.edge_attr], axis=0)
    g1.pos = paddle.concat([g1.pos, g2.pos], axis=0)
    return g1

class LazyMLP(nn.Layer):
    def __init__(self, layer, input_size):
        super(LazyMLP, self).__init__()
        num_layers = len(layer) 
        self._layers_ordered_dict = OrderedDict()
        self.in_size = input_size 
        for index, output_size in enumerate(layer):
            self._layers_ordered_dict["linear_" + str(index)] = nn.Linear(self.in_size, output_size)
            if index < (num_layers - 1): 
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
            self.in_size = output_size

        self.layers = nn.LayerDict(self._layers_ordered_dict)

    def forward(self, input):
        for k in self.layers:
            l = self.layers[k] 
            output = l(input)
            input = output
        return input

class Encoder(nn.Layer):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size, mode):
        super(Encoder, self).__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size
        if mode == "airfoil": 
            self.node_model = self._make_mlp(latent_size, input_size = 5)
        else:
            self.node_model = self._make_mlp(latent_size, input_size = 4)

        self.mesh_edge_model = self._make_mlp(latent_size, input_size = 1)
        '''
        for _ in graph.edge_sets:
          edge_model = make_mlp(latent_size)
          self.edge_models.append(edge_model)
        '''

    def forward(self, graph):
        node_latents = self.node_model(graph.x)
        edge_latent = self.mesh_edge_model(graph.edge_attr)
        graph.x=node_latents
        graph.edge_attr=edge_latent
        return graph

class Decoder(nn.Layer):
    """Decodes node features from graph."""

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, output_size):
        super(Decoder, self).__init__()
        self.model = make_mlp(output_size, 128)
    def forward(self,node_features):
        return self.model(node_features)

class EncodeProcessDecode(nn.Layer):
    """Encode-Process-Decode GraphNet model."""
    def __init__(self,
                 output_size, 
                 latent_size, 
                 num_layers, 
                 message_passing_aggregator, 
                 message_passing_steps, 
                 mode, 
                 nodes = 6684 
                 ):
        super(EncodeProcessDecode, self).__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self.min_nodes=nodes
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator  
        self.mode = mode   
        self.encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size, mode = self.mode)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator,
                                   stochastic_message_passing_used=False)
        self.post_processor=self._make_mlp(self._latent_size, 128)                           
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)                

    def _make_mlp(self, output_size, input_size = 5, layer_norm=True): 
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size] 
        network = LazyMLP(widths, input_size)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def spa_compute(self,x,p):
        j=len(x)-1
        node_features=x[j].x
        for k in range(1,j+1):
            pos=p[-k]
            fine_nodes = x[-(k+1)].pos
            feature=knn_interpolate(node_features, pos, fine_nodes)  
            node_features=x[-(k+1)].x+feature                  
            node_features=self.post_processor(node_features)       
        return node_features   


    def forward(self, graphs):
        batch = MyCopy(graphs[0])
        for index, graph in enumerate(graphs):
            if index > 0:
                batch = Myadd(batch, graph)

        latent_graph = self.encoder(batch)
        x,p= self.processor(latent_graph)
        node_features=self.spa_compute(x,p)
        pred_field = self.decoder(node_features)

        return pred_field
