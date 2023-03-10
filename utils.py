import paddle
import numpy as np
from scipy.sparse import coo_matrix
from pyamg.classical.split import RS
from paddle.sparse import coalesce
import pathlib
import warnings
import math
from PIL import Image
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from paddle.vision.transforms import ToTensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def getcorsenode(latent_graph):
    row = latent_graph.edge_index[0].numpy()
    col = latent_graph.edge_index[1].numpy()
    data = paddle.ones(shape=[row.size]).numpy()
    A = coo_matrix((data, (row, col))).tocsr()
    splitting=RS(A)
    index=np.array(np.nonzero(splitting))
    b = paddle.to_tensor(index)
    b = paddle.squeeze(b)
    return b


def StAS(index_A, value_A, index_S, value_S,N, kN,nor):
    r"""come from Ranjan, E., Sanyal, S., Talukdar, P. (2020, April). Asap: Adaptive structure aware pooling
        for learning hierarchical graph representations. AAAI(2020)"""

    sp_x = paddle.sparse.sparse_coo_tensor(index_A, value_A)
    sp_x = paddle.sparse.coalesce(sp_x)
    index_A = sp_x.indices()
    value_A = sp_x.values() 

    sp_s = paddle.sparse.sparse_coo_tensor(index_S, value_S)
    sp_s = paddle.sparse.coalesce(sp_s)
    index_S = sp_s.indices()
    value_S = sp_s.values()

    indices_A = index_A.numpy()
    values_A = value_A.numpy()
    coo_A = coo_matrix((values_A, (indices_A[0],indices_A[1])), shape=(N,N))

    indices_S = index_S.numpy()
    values_S = value_S.numpy()
    coo_S = coo_matrix((values_S, (indices_S[0],indices_S[1])), shape=(N,kN))


    ans = coo_A.dot(coo_S).tocoo()
    row = paddle.to_tensor(ans.row)
    col = paddle.to_tensor(ans.col)
    index_B = paddle.stack([row, col], axis=0)
    value_B = paddle.to_tensor(ans.data)

    indices_A = index_S
    values_A = value_S
    coo_A = paddle.sparse.sparse_coo_tensor(indices_A, values_A)
    out = paddle.sparse.transpose(coo_A, [1, 0])
    index_St = out.indices()
    value_St = out.values()

    sp_x = paddle.sparse.sparse_coo_tensor(index_B, value_B)
    sp_x = paddle.sparse.coalesce(sp_x)
    index_B = sp_x.indices()
    value_B = sp_x.values()


    indices_A = index_St.numpy()
    values_A = value_St.numpy()
    coo_A = coo_matrix((values_A, (indices_A[0],indices_A[1])), shape=(kN,N))

    indices_S = index_B.numpy()
    values_S = value_B.numpy()
    coo_S = coo_matrix((values_S, (indices_S[0],indices_S[1])), shape=(N,kN))

    ans = coo_A.dot(coo_S).tocoo()
    row = paddle.to_tensor(ans.row)
    col = paddle.to_tensor(ans.col)
    index_E = paddle.stack([row, col], axis=0)
    value_E = paddle.to_tensor(ans.data)

    return index_E, value_E




# @brief:??????????????????????????????
def remove_self_loops(edge_index: paddle.Tensor,
                      edge_attr: Optional[paddle.Tensor] = None) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    mask = edge_index[0] != edge_index[1]
    mask = mask.tolist()
    # edge_index = edge_index[:, mask]
    edge_index = edge_index.t()
    edge_index = edge_index[mask]
    edge_index = edge_index.t()
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]

def graph_connectivity(perm, edge_index, edge_weight, score, pos, N, nor):
    """come from Ranjan, E., Sanyal, S., Talukdar, P. (2020, April). Asap: Adaptive structure aware pooling
    for learning hierarchical graph representations. AAAI(2020)"""

    kN = perm.shape[0]
    perm2 = perm.reshape((-1, 1))
    mask = (edge_index[0] == perm2).sum(axis=0).astype('bool')

    S0 = edge_index[1][mask].reshape((1, -1))
    S1 = edge_index[0][mask].reshape((1, -1))
    index_S = paddle.concat([S0, S1], axis=0)
    value_S = score[mask].detach().squeeze()
    n_idx = paddle.zeros([N], dtype=paddle.int64)
    n_idx[perm] = paddle.arange(perm.shape[0])
    index_S = index_S.astype('int64')
    index_S[1] = n_idx[index_S[1]]
    subgraphnode_pos = pos[perm]
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].shape[0])
    else:
        value_A = edge_weight.clone()
    value_A = paddle.squeeze(value_A)
    attrlist = []
    for i in range(128):
        val_A = paddle.where(value_A[:,i] == 0, paddle.to_tensor(0.001), value_A[:,i])
        index_E, value_E = StAS(index_A, val_A, index_S, value_S, N, kN, nor)
        index_E, value_E = remove_self_loops(edge_index=index_E, edge_attr=value_E)
        attrlist.append(value_E)
    edge_weight = paddle.stack(attrlist, axis=1)      

    return index_E, edge_weight, subgraphnode_pos



@paddle.no_grad()
def make_grid(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
              nrow: int=8,
              padding: int=2,
              normalize: bool=False,
              value_range: Optional[Tuple[int, int]]=None,
              scale_each: bool=False,
              pad_value: int=0,
              **kwargs) -> paddle.Tensor:
    if not (isinstance(tensor, paddle.Tensor) or
            (isinstance(tensor, list) and all(
                isinstance(t, paddle.Tensor) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img - low
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding,
                        width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(
                x + 1) * width] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(tensor: Union[paddle.Tensor, List[paddle.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str]=None,
               **kwargs) -> None:
    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose(
        [1, 2, 0]).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def log_images(nodes, pred, true, elems_list, mode, index, flag, aoa, mach, iterate=0, file='field.png'):
        for field in range(pred.shape[1]):
            true_img = plot_field(nodes, elems_list, true[:, field],
                                  title='true')
            true_img = ToTensor()(true_img)
            min_max = (true[:, field].min().item(), true[:, field].max().item())

            pred_img = plot_field(nodes, elems_list, pred[:, field],
                                  title='pred',clim=min_max) 
            pred_img=ToTensor()(pred_img)
            imgs=[pred_img,true_img] 
            grid = make_grid(paddle.stack(imgs), padding=0)
            out_file=file+f'{field}'
            if flag == "airfoil":
                if ((aoa == 8.0) and (mach == 0.65)):
                    save_image(grid,'./result/image/'+str(index)+out_file+'_field.png')
                save_image(grid,'./result/image/airfoil/'+str(index)+out_file+'_field.png')
            else:
                save_image(grid,'./result/image/cylinder/'+str(index)+out_file+'_field.png')


def plot_field(nodes, elems_list, field, contour=False, clim=None, zoom=True,
               get_array=True, out_file=None, show=False, title=''):
    elems_list = sum(elems_list, [])
    tris, _ = quad2tri(elems_list)
    tris = np.array(tris)
    x, y = nodes[:, :2].t().detach().numpy()
    field = field.detach().numpy()
    fig = plt.figure(dpi=800)
    if contour:
        plt.tricontourf(x, y, tris, field)
    else:
        plt.tripcolor(x, y, tris, field)
    if clim:
        plt.clim(*clim)
    plt.colorbar()
    if zoom:
        plt.xlim(left=-0.5, right=1.5)
        plt.ylim(bottom=-1, top=1.0)

    if title:
        plt.title(title)

    if out_file is not None:
        plt.savefig(out_file)
        plt.close()

    if show:
        plt.show()
        raise NotImplementedError

    if get_array:
        fig.canvas.draw()
        a = np.fromstring(fig.canvas.tostring_rgb(),
                          dtype=np.uint8, sep='')
        a = a.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        fig.clf()
        fig.clear()
        plt.close()
        return a
def quad2tri(elems):
    new_elems = []
    new_edges = []
    for e in elems:
        if len(e) <= 3:
            new_elems.append(e)
        else:
            new_elems.append([e[0], e[1], e[2]])
            new_elems.append([e[0], e[2], e[3]])
            new_edges.append(paddle.to_tensor(([[e[0]], [e[2]]]), dtype=paddle.int64))
    new_edges = paddle.concat(new_edges, axis=1) if new_edges else paddle.to_tensor([], dtype=paddle.int64)
    return new_elems, new_edges
