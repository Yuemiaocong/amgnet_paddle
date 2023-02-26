import paddle
from paddle import nn
import os
import sys
import collections
from collections import OrderedDict
from utils import getcorsenode, graph_connectivity
from paddle.nn import LayerNorm
import numpy as np
import pgl
import time

class GraphNetBlock(nn.Layer):
    """Multi-Edge Interaction Network with residual connections."""
    def __init__(self, model_fn, output_size, message_passing_aggregator, attention=False):
        super().__init__()
        self.edge_model = model_fn(output_size, 384)
        self.node_model = model_fn(output_size, 256)
        self.message_passing_aggregator = message_passing_aggregator


    def _update_edge_features(self,graph):
        """Aggregrates node features, and applies edge function."""
        senders = graph.edge_index[0]
        receivers = graph.edge_index[1]
        sender_features = paddle.index_select(x=graph.x,index=senders, axis=0)
        receiver_features = paddle.index_select(x=graph.x,index=receivers, axis=0)
        features = [sender_features, receiver_features,graph.edge_attr]
        features = paddle.concat(features, axis=-1)
        return self.edge_model(features)


    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
        assert data.shape[0] == segment_ids.shape[0], "data.shape and segment_ids.shape should be equal"
        shape = [num_segments] + list(data.shape[1:])
        result_shape = paddle.zeros(shape)
        if operation == 'sum':
            result = paddle.scatter(result_shape, segment_ids, data, overwrite=False)
        return result
    

    def _update_node_features(self, node_features, edge_attr,edge_index):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0] 
        features = [node_features]
        features.append(
                    self.unsorted_segment_operation(edge_attr,edge_index[1], num_nodes,
                                                    operation=self.message_passing_aggregator))
        features = paddle.concat(features, axis=-1)
        return self.node_model(features)

    def forward(self, graph):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        new_edge_features = self._update_edge_features(graph)
        new_node_features = self._update_node_features(graph.x,graph.edge_attr,graph.edge_index)

        new_node_features += graph.x
        new_edge_features+=graph.edge_attr 
        latent_graph = pgl.Graph(num_nodes=new_node_features.shape[0],
                        edges=graph.edge_index)
        latent_graph.x = new_node_features
        latent_graph.edge_attr = new_edge_features
        latent_graph.pos = graph.pos
        latent_graph.edge_index = graph.edge_index
        return latent_graph

min_nodes=2000  #Each mesh can be coarsened to have no fewer points than this value

class Processor(nn.Layer):
    '''
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection(features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent graph will be feed into original processor

    Option: choose whether to normalize the high rank node connection
    '''

    def __init__(self, make_mlp, output_size, message_passing_steps, message_passing_aggregator,attention=False,
                 stochastic_message_passing_used=False):
        super().__init__()
        self.stochastic_message_passing_used = stochastic_message_passing_used
        self.graphnet_blocks = nn.LayerList()
        self.cofe_edge_blocks = nn.LayerList()
        self.pool_blocks =nn.LayerList()
        self.latent_size=output_size
        self.normalization=LayerNorm(128)
        for index in range(message_passing_steps):
            self.graphnet_blocks.append(GraphNetBlock(model_fn=make_mlp, output_size=output_size,
                                                      message_passing_aggregator=message_passing_aggregator,
                                                      attention=attention))

            self.pool_blocks.append(GraphNetBlock(model_fn=make_mlp, output_size=output_size,
                                                      message_passing_aggregator=message_passing_aggregator,
                                                      attention=attention))                                         
    def forward(self, latent_graph, normalized_adj_mat=None):
            x=[]
            pos=[]
            new=[]
            for (graphnet_block,pool) in zip(self.graphnet_blocks,self.pool_blocks):
                if latent_graph.x.shape[0]>min_nodes:
                    pre_matrix=graphnet_block(latent_graph)
                    x.append(pre_matrix)
                    cofe_graph=pool(pre_matrix) 
                    coarsenodes=getcorsenode(pre_matrix)
                    nodesfeatures=cofe_graph.x[coarsenodes]
                    subedge_index, edge_weight,subpos=graph_connectivity(perm=coarsenodes,
                    edge_index=cofe_graph.edge_index,
                    edge_weight=cofe_graph.edge_attr,score=cofe_graph.edge_attr[:,0]
                    ,pos=cofe_graph.pos, N=cofe_graph.x.shape[0],nor=self.normalization)  

                    edge_weight=self.normalization(edge_weight)
                    pos.append(subpos)
                    latent_graph = pgl.Graph(num_nodes=nodesfeatures.shape[0],
                        edges=subedge_index)
                    latent_graph.x = nodesfeatures
                    latent_graph.edge_attr = edge_weight
                    latent_graph.pos = subpos
                    latent_graph.edge_index = subedge_index
                else: 
                      latent_graph=graphnet_block(latent_graph)
                      new.append(latent_graph)
            if len(new):
                x.append(new[-1])           
            return x,pos