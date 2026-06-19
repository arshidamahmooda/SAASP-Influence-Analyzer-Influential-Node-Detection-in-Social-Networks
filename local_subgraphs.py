"""
local_subgraphs.py — Extract L-hop local subgraphs per node.

For each node vi, extract subgraph GL(vi) containing all nodes
reachable within L hops using BFS up to depth L.
"""

import networkx as nx


def extract_local_subgraph(G: nx.Graph, node, L: int = 3) -> nx.Graph:
    """
    Extract the L-hop local subgraph around a given node.
    
    Parameters
    ----------
    G : nx.Graph
        The full graph.
    node : 
        The center node.
    L : int
        Number of hops (default 3).
    
    Returns
    -------
    GL : nx.Graph
        Induced subgraph of nodes within L hops of `node`.
    """
    # BFS to depth L
    nodes_in_ball = set()
    nodes_in_ball.add(node)
    frontier = {node}
    for _ in range(L):
        next_frontier = set()
        for u in frontier:
            for v in G.neighbors(u):
                if v not in nodes_in_ball:
                    nodes_in_ball.add(v)
                    next_frontier.add(v)
        frontier = next_frontier
        if not frontier:
            break
    return G.subgraph(nodes_in_ball).copy()


def extract_local_subgraph_without_node(G: nx.Graph, node, L: int = 3) -> nx.Graph:
    """
    Extract the L-hop local subgraph around `node`, then remove `node`.
    
    Returns
    -------
    GL_minus : nx.Graph
        Local subgraph with the center node removed.
    """
    GL = extract_local_subgraph(G, node, L)
    GL_minus = GL.copy()
    GL_minus.remove_node(node)
    return GL_minus
