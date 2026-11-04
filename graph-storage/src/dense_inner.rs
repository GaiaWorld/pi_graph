//! 稠密存储的配对实现；状态和契约均由父模块真实声明。
use super::*;

pub(super) fn batch_delete_node<N, E>(
    graph: &mut DenseGraph<N, E>,
    id: DenseNodeId,
    ids: &mut crate::common::BatchIds<DenseNodeId, DenseEdgeId>,
) -> Result<bool, GraphError> {
    graph.delete_node_tracked(id, Some(ids))
}

pub(super) fn batch_delete_edge<N, E>(
    graph: &mut DenseGraph<N, E>,
    id: DenseEdgeId,
    ids: &mut crate::common::BatchIds<DenseNodeId, DenseEdgeId>,
) -> Result<bool, GraphError> {
    if !graph.valid_edge(id) {
        return Ok(false);
    }
    let moved = DenseEdgeId {
        index: graph.edges.len() - 1,
        generation: graph.edges.last().expect("nonempty edges").generation,
    };
    let result = graph.delete_edge(id)?;
    ids.remove_edge(id);
    if moved != id {
        ids.relocate_edge(
            moved,
            DenseEdgeId {
                index: id.index,
                generation: moved.generation,
            },
        );
    }
    Ok(result)
}

pub(super) fn delete_node_tracked<N, E>(
    graph: &mut DenseGraph<N, E>,
    id: DenseNodeId,
    mut ids: Option<&mut crate::common::BatchIds<DenseNodeId, DenseEdgeId>>,
) -> Result<bool, GraphError> {
    if !graph.valid_node(id) {
        return Ok(false);
    }
    let version = graph.next_version()?;
    let mut removed_edges = Vec::with_capacity(
        graph.nodes[id.index].outgoing.len() + graph.nodes[id.index].incoming.len(),
    );
    while let Some(edge) = graph.nodes[id.index]
        .outgoing
        .last()
        .copied()
        .or_else(|| graph.nodes[id.index].incoming.last().copied())
    {
        let moved = DenseEdgeId {
            index: graph.edges.len() - 1,
            generation: graph.edges.last().expect("nonempty edges").generation,
        };
        removed_edges.push(graph.unlink_edge(edge));
        if let Some(ids) = ids.as_deref_mut() {
            ids.remove_edge(edge);
            if moved != edge {
                ids.relocate_edge(
                    moved,
                    DenseEdgeId {
                        index: edge.index,
                        generation: moved.generation,
                    },
                );
            }
        }
    }
    let last = graph.nodes.len() - 1;
    let removed = graph.nodes.swap_remove(id.index);
    if id.index != last {
        let node = &graph.nodes[id.index];
        let from = DenseNodeId {
            index: last,
            generation: node.generation,
        };
        let to = DenseNodeId {
            index: id.index,
            generation: node.generation,
        };
        for edge in &node.outgoing {
            graph.edges[edge.index].source = to;
        }
        for edge in &node.incoming {
            graph.edges[edge.index].target = to;
        }
        if let Some(ids) = ids.as_deref_mut() {
            ids.relocate_node(from, to);
        }
    }
    if let Some(ids) = ids {
        ids.remove_node(id);
    }
    graph.version = version;
    drop(removed_edges);
    drop(removed);
    Ok(true)
}

pub(super) fn delete_node<N, E>(
    graph: &mut DenseGraph<N, E>,
    id: DenseNodeId,
) -> Result<bool, GraphError> {
    graph.delete_node_tracked(id, None)
}

pub(super) fn unlink_edge<N, E>(graph: &mut DenseGraph<N, E>, id: DenseEdgeId) -> Edge<E> {
    let last = graph.edges.len() - 1;
    let removed = graph.edges.swap_remove(id.index);
    graph.nodes[removed.source.index]
        .outgoing
        .retain(|edge| *edge != id);
    graph.nodes[removed.target.index]
        .incoming
        .retain(|edge| *edge != id);
    if id.index != last {
        let moved = &graph.edges[id.index];
        let from = DenseEdgeId {
            index: last,
            generation: moved.generation,
        };
        let to = DenseEdgeId {
            index: id.index,
            generation: moved.generation,
        };
        for edge in &mut graph.nodes[moved.source.index].outgoing {
            if *edge == from {
                *edge = to;
            }
        }
        for edge in &mut graph.nodes[moved.target.index].incoming {
            if *edge == from {
                *edge = to;
            }
        }
    }
    removed
}

pub(super) fn delete_edge<N, E>(
    graph: &mut DenseGraph<N, E>,
    id: DenseEdgeId,
) -> Result<bool, GraphError> {
    if !graph.valid_edge(id) {
        return Ok(false);
    }
    let version = graph.next_version()?;
    let removed = graph.unlink_edge(id);
    graph.version = version;
    drop(removed);
    Ok(true)
}

pub(super) fn add_edge<N, E>(
    graph: &mut DenseGraph<N, E>,
    source: DenseNodeId,
    target: DenseNodeId,
    weight: E,
) -> Result<DenseEdgeId, GraphError> {
    if !graph.valid_node(source) || !graph.valid_node(target) {
        return Err(GraphError::MissingEndpoint);
    }
    let generation = graph.next_version()?;
    let id = DenseEdgeId {
        index: graph.edges.len(),
        generation,
    };
    graph.edges.reserve(1);
    graph.nodes[source.index].outgoing.reserve(1);
    graph.nodes[target.index].incoming.reserve(1);
    graph.edges.push(Edge {
        weight,
        generation,
        source,
        target,
    });
    graph.nodes[source.index].outgoing.push(id);
    graph.nodes[target.index].incoming.push(id);
    graph.version = generation;
    Ok(id)
}

pub(super) fn add_node<N, E>(
    graph: &mut DenseGraph<N, E>,
    weight: N,
) -> Result<DenseNodeId, GraphError> {
    let generation = graph.next_version()?;
    let id = DenseNodeId {
        index: graph.nodes.len(),
        generation,
    };
    graph.nodes.push(Node {
        weight,
        generation,
        outgoing: Vec::new(),
        incoming: Vec::new(),
    });
    graph.version = generation;
    Ok(id)
}

pub(super) fn adjacent<N, E>(
    graph: &DenseGraph<N, E>,
    id: DenseNodeId,
    outgoing: bool,
) -> std::iter::Copied<std::slice::Iter<'_, DenseEdgeId>> {
    let edges = if graph.valid_node(id) {
        let n = &graph.nodes[id.index];
        if outgoing {
            n.outgoing.as_slice()
        } else {
            n.incoming.as_slice()
        }
    } else {
        &[]
    };
    edges.iter().copied()
}

pub(super) fn edge_data_mut<N, E>(
    graph: &mut DenseGraph<N, E>,
    id: DenseEdgeId,
) -> Option<(DenseNodeId, DenseNodeId, &mut E)> {
    if !graph.valid_edge(id) {
        return None;
    }
    let e = &mut graph.edges[id.index];
    Some((e.source, e.target, &mut e.weight))
}

pub(super) fn edge_data<N, E>(
    graph: &DenseGraph<N, E>,
    id: DenseEdgeId,
) -> Option<(DenseNodeId, DenseNodeId, &E)> {
    if !graph.valid_edge(id) {
        return None;
    }
    let e = &graph.edges[id.index];
    Some((e.source, e.target, &e.weight))
}

pub(super) fn node_weight_mut<N, E>(
    graph: &mut DenseGraph<N, E>,
    id: DenseNodeId,
) -> Option<&mut N> {
    if !graph.valid_node(id) {
        return None;
    }
    Some(&mut graph.nodes[id.index].weight)
}

pub(super) fn node_weight<N, E>(graph: &DenseGraph<N, E>, id: DenseNodeId) -> Option<&N> {
    graph.valid_node(id).then(|| &graph.nodes[id.index].weight)
}

pub(super) fn edge_keys<N, E>(graph: &DenseGraph<N, E>) -> DenseEdges<'_, E> {
    DenseEdges {
        edges: graph.edges.iter().enumerate(),
    }
}

pub(super) fn node_keys<N, E>(graph: &DenseGraph<N, E>) -> DenseNodes<'_, N> {
    DenseNodes {
        nodes: graph.nodes.iter().enumerate(),
    }
}

pub(super) fn valid_edge<N, E>(graph: &DenseGraph<N, E>, id: DenseEdgeId) -> bool {
    graph
        .edges
        .get(id.index)
        .is_some_and(|e| e.generation == id.generation)
}

pub(super) fn valid_node<N, E>(graph: &DenseGraph<N, E>, id: DenseNodeId) -> bool {
    graph
        .nodes
        .get(id.index)
        .is_some_and(|n| n.generation == id.generation)
}

pub(super) fn next_version<N, E>(graph: &DenseGraph<N, E>) -> Result<u64, GraphError> {
    graph
        .version
        .checked_add(1)
        .ok_or(GraphError::VersionExhausted)
}

pub(super) fn edge_count<N, E>(graph: &DenseGraph<N, E>) -> usize {
    graph.edges.len()
}

pub(super) fn node_count<N, E>(graph: &DenseGraph<N, E>) -> usize {
    graph.nodes.len()
}

pub(super) fn with_capacity<N, E>(nodes: usize, edges: usize) -> DenseGraph<N, E> {
    DenseGraph {
        nodes: Vec::with_capacity(nodes),
        edges: Vec::with_capacity(edges),
        version: 0,
    }
}

pub(super) fn new<N, E>() -> DenseGraph<N, E> {
    DenseGraph::with_capacity(0, 0)
}

pub(super) fn index_node_at_index<N, E>(
    graph: &DenseGraph<N, E>,
    index: usize,
) -> Option<DenseNodeId> {
    graph.nodes.get(index).map(|node| DenseNodeId {
        index,
        generation: node.generation,
    })
}

pub(super) fn index_to_index<N, E>(graph: &DenseGraph<N, E>, id: DenseNodeId) -> usize {
    assert!(graph.valid_node(id));
    id.index
}

pub(super) fn index_node_bound<N, E>(graph: &DenseGraph<N, E>) -> usize {
    graph.nodes.len()
}

pub(super) fn dense_node_at_dense<N, E>(
    graph: &DenseGraph<N, E>,
    index: usize,
) -> Option<DenseNodeId> {
    graph.node_at_index(index)
}

pub(super) fn dense_to_dense<N, E>(graph: &DenseGraph<N, E>, id: DenseNodeId) -> Option<usize> {
    graph.valid_node(id).then_some(id.index)
}

pub(super) fn dense_node_count<N, E>(graph: &DenseGraph<N, E>) -> usize {
    graph.nodes.len()
}
