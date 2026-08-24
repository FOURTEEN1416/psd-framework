"""ST-GCN+BC K9Graph 测试.

TDD 前置：确认 24 节点拓扑结构。
Owner: W11 窗口
"""
import pytest
import numpy as np


def test_k9_graph_num_nodes():
    from psd.models.stgcn_bc_k9_graph import K9Graph
    g = K9Graph()
    assert g.num_nodes == 24


def test_k9_graph_root_is_withers():
    from psd.models.stgcn_bc_k9_graph import K9Graph
    g = K9Graph()
    assert g.root == 22  # withers


def test_k9_graph_node_names_count():
    from psd.models.stgcn_bc_k9_graph import K9Graph
    g = K9Graph()
    assert len(g.NODE_NAMES) == 24
    assert g.NODE_NAMES[22] == "withers"
    assert g.NODE_NAMES[0] == "front_left_paw"


def test_k9_graph_edges():
    from psd.models.stgcn_bc_k9_graph import K9Graph
    g = K9Graph()
    assert len(g.outward) > 0
    assert len(g.inward) == len(g.outward)
    # inward is reversed outward
    for (c, p) in g.inward:
        assert (p, c) in g.outward


def test_k9_graph_parent_array():
    from psd.models.stgcn_bc_k9_graph import K9Graph
    g = K9Graph()
    assert g.parent.shape == (24,)
    assert g.parent[g.root] == -1  # root has no parent
    # All other nodes should have a parent
    for i in range(24):
        if i != g.root:
            assert g.parent[i] >= 0


def test_k9_graph_adjacency_symmetric():
    from psd.models.stgcn_bc_k9_graph import K9Graph
    g = K9Graph()
    adj = g.adjacency
    assert adj.shape == (24, 24)
    np.testing.assert_array_equal(adj, adj.T)  # symmetric
    assert np.allclose(np.diag(adj), 1)  # self-loops


def test_k9_graph_summary():
    from psd.models.stgcn_bc_k9_graph import K9Graph
    g = K9Graph()
    s = g.summary()
    assert "24-node" in s
    assert "withers" in s.lower()
