"""
"""

import numpy as np
import networkx as nx
from numpy.random import default_rng
import warnings


def circuit_generator(
    n_rec=5, 
    n_inp=2, 
    
    n_rinh=2,
    n_iinh=0,
    
    n_rconn=5, 
    n_iconn=4, 
    
    post_min=1, 
    pre_min=1, 
    
    rwmin=0.2, 
    rwmax=0.8,
    
    iwmin=1, 
    iwmax=1,
    
    seed=None, 
    normalize=True, 
    norm_with_input=False
):
    """
    Generate a randomly connected circuit
    """
    rng = default_rng(seed)
    
    Wi = np.zeros((n_inp, n_rec))
    Wr = np.zeros((n_rec, n_rec))
    
    rinh_idx = rng.choice(n_rec, n_rinh, replace=False)
    iinh_idx = rng.choice(n_inp, n_iinh, replace=False)
    
    rw = rng.random(n_rconn) * (rwmax - rwmin) + rwmin
    iw = rng.random(n_iconn) * (iwmax - iwmin) + iwmin
    
    Wi[np.arange(n_inp), rng.integers(n_rec, size=n_inp)] = iw[:n_inp]
    rest = n_iconn - n_inp
    if rest:
        idcs = np.stack(np.nonzero(Wi == 0), axis=1)
        idcs = rng.choice(idcs, size=rest, replace=False)
        Wi[(idcs[:, 0], idcs[:, 1])] = iw[n_inp:]
        
    Wi[iinh_idx] *= -1
    
    def get_pre_and_post_cond(Wr, Wi):
        pre = (Wr != 0).sum(axis=1) >= pre_min
        post = ((Wi != 0).sum(axis=0) + (Wr != 0).sum(axis=0)) >= post_min
        return pre, post
    
    pre, post = get_pre_and_post_cond(Wr, Wi)
    conds = pre & post
    wr_idx = 0
    while not conds.all() or (wr_idx < n_rconn):
        pre_idx, post_idx = np.nonzero(~pre)[0], np.nonzero(~post)[0] 
        # print(pre_idx, post_idx)
        if len(pre_idx) and len(post_idx):
            idcs = np.stack(np.meshgrid(pre_idx, post_idx)).reshape(2, -1)
        elif len(pre_idx):
            idcs = np.stack(np.meshgrid(pre_idx, np.arange(n_rec))).reshape(2, -1)
        elif len(post_idx):
            idcs = np.stack(np.meshgrid(np.arange(n_rec), post_idx)).reshape(2, -1)
        else:
            idcs = np.stack(np.meshgrid(np.arange(n_rec), np.arange(n_rec))).reshape(2, -1)

        idcs = idcs[:, idcs[0] != idcs[1]]  # not self
        # zeros
        idcs_ = np.stack(np.nonzero(Wr == 0), axis=0)
        idcs = idcs[:, (idcs[..., None] == idcs_[:, None]).all(axis=0).any(axis=-1)]

        idx = rng.choice(idcs.T)
        Wr[idx[0], idx[1]] = rw[wr_idx]

        wr_idx += 1
        pre, post = get_pre_and_post_cond(Wr, Wi)
        conds = pre & post
        
    if not conds.all():
        warnings.warn("Conditions not met")

    if wr_idx < n_rconn:
        rest = n_rconn - wr_idx
        idcs = np.stack(np.nonzero(Wr == 0), axis=1)
        idcs = rng.choice(idcs, size=rest, replace=False)
        Wi[(idcs[:, 0], idcs[:, 1])] = iw[wr_idx:]

    Wr[rinh_idx] *= -1
    
    if normalize:
        if norm_with_input:
            n = np.abs(Wi).sum(0)+np.abs(Wr).sum(0)
            Wi /= n
            Wr /= n
        else:
            n = np.abs(Wr).sum(0)
            Wr[:, n != 0] /= n[n!=0]

    return Wi, Wr


def stack_ws(Wi, Wr):
    W = np.vstack([Wi, Wr])
    
    rest = np.zeros((len(W), Wi.shape[0]))
    return np.hstack([rest, W])


def create_nx_plot(
    Wi, Wr, 
    ax=None,
    rexc_color='limegreen', 
    rinh_color='tomato', 
    iexc_color='darkgreen', 
    iinh_color='firebrick', 
    edge_scale=1, 
    connectionstyle='arc3,rad=0.2', 
    node_scale=1, 
    arrow_scale=1, 
    inp_pos=4, 
    rec_pos=2,
    highlight_idcs=[], 
    highlight_color='black', 
    annot=False
):
    W = stack_ws(Wi, Wr)
    
    G = nx.from_numpy_matrix(
        W, create_using=nx.MultiDiGraph
    )
    
    inhidx = np.nonzero((W < 0).any(axis=1))[0]
    inpidx = np.arange(Wi.shape[0])
    recidx = np.arange(Wi.shape[0], W.shape[0])
    
    pos = {}
    for idx in inpidx:
        pos[idx] = np.array([np.cos(idx * 2 * np.pi / len(inpidx)), np.sin(idx * 2 * np.pi / len(inpidx))]) * inp_pos

    for idx in recidx:
        pos[idx] = np.array([np.cos(idx * 2 * np.pi / len(recidx)), np.sin(idx * 2 * np.pi / len(recidx))]) * rec_pos
        
    for idx in inpidx:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[idx], 
            node_color=(
                iinh_color if idx in inhidx else iexc_color
            ),
            node_size=700*node_scale, 
            ax=ax, 
            edgecolors=(
                highlight_color if idx in highlight_idcs
                else None
            )
        )
    
    for idx in recidx:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[idx], 
            node_color=(
                rinh_color if idx in inhidx else rexc_color
            ), 
            node_size=700*node_scale, 
            ax=ax,
            edgecolors=(
                highlight_color if idx in highlight_idcs
                else None
            )
        )

    for (u, v, d) in G.edges(data=True):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], 
            width=5 * np.abs(d['weight']) * edge_scale, 
            edge_color=(
                iinh_color if u in inpidx and u in inhidx 
                else
                iexc_color if u in inpidx
                else
                rinh_color if u in inhidx
                else
                rexc_color
            ), 
            arrowsize=20*arrow_scale, 
            connectionstyle=connectionstyle, 
            ax=ax
        )
    if annot:
        labels = {
            idx: str(idx)
            for idx in inpidx.tolist() + recidx.tolist()
        }
        nx.draw_networkx_labels(G, pos, labels, font_size=16*node_scale)
    return G

