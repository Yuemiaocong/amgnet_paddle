import paddle
import os
import pickle
from pathlib import Path 
import numpy as np
import math,copy
from os import PathLike
from typing import Sequence, Dict, Union, Tuple, List
import collections.abc as container_abcs
import pgl
from pgl.utils.data.dataloader import Dataloader, Dataset

SU2_SHAPE_IDS = {
    'line': 3,
    'triangle': 5,
    'quad': 9,
}

def get_mesh_graph(mesh_filename: Union[str, PathLike],
                   dtype: np.dtype = np.float32
                   ) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], Dict[str, List[List[int]]]]:

    def get_rhs(s: str) -> str:
        return s.split('=')[-1]

    marker_dict = {}
    with open(mesh_filename) as f:
        for line in f:
            if line.startswith('NPOIN'):
                num_points = int(get_rhs(line))
                mesh_points = [[float(p) for p in f.readline().split()[:2]]
                               for _ in range(num_points)]
                nodes = np.array(mesh_points, dtype=dtype)
            if line.startswith('NMARK'):
                num_markers = int(get_rhs(line))
                for _ in range(num_markers):
                    line = f.readline()
                    assert line.startswith('MARKER_TAG')
                    marker_tag = get_rhs(line).strip()
                    num_elems = int(get_rhs(f.readline()))
                    marker_elems = [[int(e) for e in f.readline().split()[-2:]]
                                    for _ in range(num_elems)]
                    marker_dict[marker_tag] = marker_elems
            if line.startswith('NELEM'):
                edges = []
                triangles = []
                quads = []
                num_edges = int(get_rhs(line))
                for _ in range(num_edges):
                    elem = [int(p) for p in f.readline().split()]
                    if elem[0] == SU2_SHAPE_IDS['triangle']:
                        n = 3
                        triangles.append(elem[1:1+n])
                    elif elem[0] == SU2_SHAPE_IDS['quad']:
                        n = 4
                        quads.append(elem[1:1+n])
                    else:
                        raise NotImplementedError
                    elem = elem[1:1+n]
                    edges += [[elem[i], elem[(i+1) % n]] for i in range(n)]
                edges = np.array(edges, dtype=np.compat.long).transpose() 
                elems = [triangles, quads]      
    return nodes, edges, elems, marker_dict

class MeshcylinderDataset(Dataset):
    def __init__(self, root, mode='train'):
        self.mode = mode
        self.data_dir = Path(root) / (mode)
        self.file_list = os.listdir(self.data_dir)
        self.len = len(self.file_list)
        self.MyPath = Path(root) 

        self.mesh_graph = get_mesh_graph(Path(root) / 'cylinder.su2')

        self.normalization_factors = paddle.to_tensor([[978.6001,  48.9258,  24.8404],
        [-692.3159,   -6.9950,  -24.8572]])
        self.nodes = paddle.to_tensor(self.mesh_graph[0])
        self.meshnodes=self.mesh_graph[0]
        self.edges = paddle.to_tensor(self.mesh_graph[1])
        self.elems_list = self.mesh_graph[2]
        self.marker_dict = self.mesh_graph[3]
        self.bounder=[]
        self.node_markers = paddle.full([self.nodes.shape[0], 1], fill_value=-1)
        for i, (marker_tag, marker_elems) in enumerate(self.marker_dict.items()):
            for elem in marker_elems:
                self.node_markers[elem[0]] = i
                self.node_markers[elem[1]] = i

        
    def __len__(self):
        return self.len

    def get(self, idx):
        with open(self.data_dir/self.file_list[idx],'r') as f:
            field=[]
            pos=[]
            i=1
            for lines in f.readlines():
                if not i: 
                        lines=lines.rstrip('\n')
                        lines_pos=lines.split(',')[1:3]
                        lines_field=lines.split(',')[3:]
                        numbers_float =list(eval(i) for i in lines_pos)
                        array=np.array(numbers_float,np.float32)
                        a=paddle.to_tensor(array) 
                        pos.append(a)
                        numbers_float =list(eval(i) for i in lines_field)
                        array=np.array(numbers_float,np.float32)
                        a=paddle.to_tensor(array) 
                        field.append(a)
                i=0
        field=paddle.stack(field,axis=0)
        pos= paddle.stack(pos,axis=0)       
        indexlist=[]
        f=open(self.MyPath/"2.txt","w")
        for i in range(self.meshnodes.shape[0]):
            b=paddle.to_tensor(self.meshnodes[i:(i+1)])
            b=paddle.squeeze(b) 
            index=paddle.nonzero(paddle.sum((pos==b), axis=1,dtype='float32')==pos.shape[1])
            f.write(f'{index}\n')
            indexlist.append(index) 
        f.close()    
        indexlist=paddle.stack(indexlist, axis=0)
        indexlist=paddle.squeeze(indexlist)
        fields=field[indexlist]
        velocity= self.get_params_from_name(self.file_list[idx])
        aoa = paddle.to_tensor(velocity) 

        norm_aoa = paddle.to_tensor(aoa/40)
        # add physics parameters to graph
        nodes = np.concatenate([
            self.nodes,
            np.repeat(a=norm_aoa, repeats=self.nodes.shape[0])[:, np.newaxis], # np.repeat用于重复repeats次，[:, np.newaxis]给列增加维度，变为1
            self.node_markers
        ], axis=-1).astype(np.float32)
        nodes = paddle.to_tensor(nodes)

        data = pgl.Graph(num_nodes=nodes.shape[0],
                        edges=self.edges,
                        )
        data.x = nodes
        data.y = fields
        data.pos = self.nodes
        data.edge_index = self.edges
        data.velocity = aoa

        sender=data.x[data.edge_index[0]]
        receiver=data.x[data.edge_index[1]]
        relation_pos=sender[:,0:2]-receiver[:,0:2]
        post=paddle.linalg.norm(relation_pos,p=2,axis=1,keepdim=True)
        data.edge_attr=post
        std_epsilon=paddle.to_tensor([1e-8])
        a=paddle.mean(data.edge_attr,axis=0)
        b=data.edge_attr.std(axis=0)
        b=paddle.maximum(b,std_epsilon)
        data.edge_attr=(data.edge_attr-a)/b
        a=paddle.mean(data.y,axis=0)
        b=data.y.std(axis=0)
        b=paddle.maximum(b,std_epsilon)
        data.y=(data.y-a)/b
        data.norm_max = a
        data.norm_min = b
        
        "find the face of the boundery,our cylinder dataset come from fluent solver"
        with open(self.MyPath/'bounder','r') as f:
            field=[]
            pos=[]
            i=1
            for lines in f.readlines():
                if not i: 
                        lines=lines.rstrip('\n')
                        lines_pos=lines.split(',')[1:3]
                        lines_field=lines.split(',')[3:]
                        numbers_float =list(eval(i) for i in lines_pos)
                        array=np.array(numbers_float,np.float32)
                        a=paddle.to_tensor(array) 
                        pos.append(a)
                        numbers_float =list(eval(i) for i in lines_field)
                        array=np.array(numbers_float,np.float32)
                        a=paddle.to_tensor(array) 
                        field.append(a)
                i=0
        field=paddle.stack(field,axis=0)
        pos= paddle.stack(pos,axis=0)       
        indexlist=[]
        for i in range(pos.shape[0]):
            b=pos[i:(i+1)]
            b=paddle.squeeze(b)
            index=paddle.nonzero(paddle.sum((self.nodes==b), axis=1,dtype='float32')==self.nodes.shape[1])
            indexlist.append(index) 
        indexlist=paddle.stack(indexlist,axis=0)
        indexlist=paddle.squeeze(indexlist)
        self.bounder=indexlist
        return data

    @staticmethod
    def get_params_from_name(filename):
        s = filename.rsplit('.', 1)[0]
        reynolds = np.array(s[13:])[np.newaxis].astype(np.float32)
        return reynolds