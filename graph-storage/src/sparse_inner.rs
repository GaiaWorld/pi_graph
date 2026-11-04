//! 稀疏存储的配对实现；状态和契约均由父模块真实声明。
use super::*;

pub(super) fn batch_delete_node<N, E>(
    graph: &mut SparseGraph<N, E>,
    id: SparseNodeId,
    ids: &mut crate::common::BatchIds<SparseNodeId, SparseEdgeId>,
) -> Result<bool, GraphError> {
    let incident: Vec<_> = graph
        .adjacent(id, true)
        .chain(graph.adjacent(id, false))
        .collect();
    let result = graph.delete_node(id)?;
    if result {
        ids.remove_node(id);
        for edge in incident {
            ids.remove_edge(edge);
        }
    }
    Ok(result)
}

pub(super) fn batch_delete_edge<N, E>(
    graph: &mut SparseGraph<N, E>,
    id: SparseEdgeId,
    ids: &mut crate::common::BatchIds<SparseNodeId, SparseEdgeId>,
) -> Result<bool, GraphError> {
    let result = graph.delete_edge(id)?;
    if result {
        ids.remove_edge(id);
    }
    Ok(result)
}

pub(super) fn delete_node<N, E>(
    graph: &mut SparseGraph<N, E>,
    id: SparseNodeId,
) -> Result<bool, GraphError> {
    let Some(node) = graph.nodes.get(id) else {
        return Ok(false);
    };
    let version = graph.next_version()?;
    let mut removed_edges = Vec::with_capacity(node.outgoing.len() + node.incoming.len());
    loop {
        let node = &graph.nodes[id];
        let edge = node
            .outgoing
            .iter(&graph.links)
            .next()
            .or_else(|| node.incoming.iter(&graph.links).next())
            .copied();
        let Some(edge) = edge else {
            break;
        };
        removed_edges.push(graph.unlink_edge(edge));
    }
    let removed = graph.nodes.remove(id).expect("live node");
    graph.node_ids.swap_remove(removed.dense);
    if let Some(moved) = graph.node_ids.get(removed.dense) {
        graph.nodes[*moved].dense = removed.dense;
    }
    graph.slots[id.data().index() as usize] = None;
    graph.version = version;
    drop(removed_edges);
    drop(removed);
    Ok(true)
}

pub(super) fn delete_edge<N, E>(
    graph: &mut SparseGraph<N, E>,
    id: SparseEdgeId,
) -> Result<bool, GraphError> {
    if !graph.edges.contains_key(id) {
        return Ok(false);
    }
    let version = graph.next_version()?;
    let removed = graph.unlink_edge(id);
    graph.version = version;
    drop(removed);
    Ok(true)
}

pub(super) fn unlink_edge<N, E>(graph: &mut SparseGraph<N, E>, id: SparseEdgeId) -> Edge<E> {
    let removed = graph.edges.remove(id).expect("live edge");
    graph.nodes[removed.source]
        .outgoing
        .ops(&mut graph.links)
        .remove(removed.out_link);
    graph.nodes[removed.target]
        .incoming
        .ops(&mut graph.links)
        .remove(removed.in_link);
    graph.edge_ids.swap_remove(removed.dense);
    if let Some(moved) = graph.edge_ids.get(removed.dense) {
        graph.edges[*moved].dense = removed.dense;
    }
    removed
}

pub(super) fn add_edge<N, E>(
    graph: &mut SparseGraph<N, E>,
    source: SparseNodeId,
    target: SparseNodeId,
    weight: E,
) -> Result<SparseEdgeId, GraphError> {
    if !graph.nodes.contains_key(source) || !graph.nodes.contains_key(target) {
        return Err(GraphError::MissingEndpoint);
    }
    let version = graph.next_version()?;
    graph.edge_ids.reserve(1);
    let id = graph.edges.insert(Edge {
        weight,
        source,
        target,
        dense: graph.edge_ids.len(),
        out_link: LinkId::default(),
        in_link: LinkId::default(),
    });
    let out_link = graph.nodes[source]
        .outgoing
        .ops(&mut graph.links)
        .push_back(id);
    let in_link = graph.nodes[target]
        .incoming
        .ops(&mut graph.links)
        .push_back(id);
    graph.edges[id].out_link = out_link;
    graph.edges[id].in_link = in_link;
    graph.edge_ids.push(id);
    graph.version = version;
    Ok(id)
}

pub(super) fn add_node<N, E>(
    graph: &mut SparseGraph<N, E>,
    weight: N,
) -> Result<SparseNodeId, GraphError> {
    let version = graph.next_version()?;
    graph.node_ids.reserve(1);
    let id = graph.nodes.insert(Node {
        weight,
        dense: graph.node_ids.len(),
        outgoing: Deque::new(),
        incoming: Deque::new(),
    });
    let slot = id.data().index() as usize;
    if graph.slots.len() <= slot {
        graph.slots.resize(slot + 1, None);
    }
    graph.slots[slot] = Some(id);
    graph.node_ids.push(id);
    graph.version = version;
    Ok(id)
}

pub(super) fn adjacent<N, E>(
    graph: &SparseGraph<N, E>,
    id: SparseNodeId,
    outgoing: bool,
) -> SparseEdges<'_> {
    SparseEdges {
        inner: graph.nodes.get(id).map(|n| {
            if outgoing {
                n.outgoing.iter(&graph.links)
            } else {
                n.incoming.iter(&graph.links)
            }
        }),
    }
}

pub(super) fn edge_data_mut<N, E>(
    graph: &mut SparseGraph<N, E>,
    id: SparseEdgeId,
) -> Option<(SparseNodeId, SparseNodeId, &mut E)> {
    graph
        .edges
        .get_mut(id)
        .map(|e| (e.source, e.target, &mut e.weight))
}

pub(super) fn edge_data<N, E>(
    graph: &SparseGraph<N, E>,
    id: SparseEdgeId,
) -> Option<(SparseNodeId, SparseNodeId, &E)> {
    graph.edges.get(id).map(|e| (e.source, e.target, &e.weight))
}

pub(super) fn node_weight_mut<N, E>(
    graph: &mut SparseGraph<N, E>,
    id: SparseNodeId,
) -> Option<&mut N> {
    graph.nodes.get_mut(id).map(|n| &mut n.weight)
}

pub(super) fn node_weight<N, E>(graph: &SparseGraph<N, E>, id: SparseNodeId) -> Option<&N> {
    graph.nodes.get(id).map(|n| &n.weight)
}

pub(super) fn edge_keys<N, E>(
    graph: &SparseGraph<N, E>,
) -> std::iter::Copied<std::slice::Iter<'_, SparseEdgeId>> {
    graph.edge_ids.iter().copied()
}

pub(super) fn node_keys<N, E>(
    graph: &SparseGraph<N, E>,
) -> std::iter::Copied<std::slice::Iter<'_, SparseNodeId>> {
    graph.node_ids.iter().copied()
}

pub(super) fn next_version<N, E>(graph: &SparseGraph<N, E>) -> Result<u64, GraphError> {
    graph
        .version
        .checked_add(1)
        .ok_or(GraphError::VersionExhausted)
}

pub(super) fn edge_count<N, E>(graph: &SparseGraph<N, E>) -> usize {
    graph.edges.len()
}

pub(super) fn node_count<N, E>(graph: &SparseGraph<N, E>) -> usize {
    graph.nodes.len()
}

pub(super) fn with_capacity<N, E>(nodes: usize, edges: usize) -> SparseGraph<N, E> {
    SparseGraph {
        nodes: SlotMap::with_capacity_and_key(nodes),
        edges: SlotMap::with_capacity_and_key(edges),
        links: DequeStorage::with_capacity(edges.saturating_mul(2)),
        node_ids: Vec::with_capacity(nodes),
        edge_ids: Vec::with_capacity(edges),
        slots: Vec::new(),
        version: 0,
    }
}

pub(super) fn new<N, E>() -> SparseGraph<N, E> {
    SparseGraph::with_capacity(0, 0)
}

/// 克隆实现仅由 SparseGraph::clone 壳调用；链接池与源图隔离。
pub(super) fn clone_graph<N: Clone, E: Clone>(graph: &SparseGraph<N, E>) -> SparseGraph<N, E> {
    let mut nodes = graph.nodes.clone();
    let mut edges = graph.edges.clone();
    let mut links = DequeStorage::with_capacity(graph.edge_ids.len().saturating_mul(2));
    for (id, node) in graph.nodes.iter() {
        for edge in node.outgoing.iter(&graph.links) {
            let link = nodes[id].outgoing.ops(&mut links).push_back(*edge);
            edges[*edge].out_link = link;
        }
        for edge in node.incoming.iter(&graph.links) {
            let link = nodes[id].incoming.ops(&mut links).push_back(*edge);
            edges[*edge].in_link = link;
        }
    }
    SparseGraph {
        nodes,
        edges,
        links,
        node_ids: graph.node_ids.clone(),
        edge_ids: graph.edge_ids.clone(),
        slots: graph.slots.clone(),
        version: graph.version,
    }
}

pub(super) fn index_node_at_index<N, E>(
    graph: &SparseGraph<N, E>,
    index: usize,
) -> Option<SparseNodeId> {
    graph.slots.get(index).copied().flatten()
}

pub(super) fn index_to_index<N, E>(graph: &SparseGraph<N, E>, id: SparseNodeId) -> usize {
    assert!(graph.nodes.contains_key(id));
    id.data().index() as usize
}

pub(super) fn index_node_bound<N, E>(graph: &SparseGraph<N, E>) -> usize {
    graph.slots.len()
}

pub(super) fn dense_node_at_dense<N, E>(
    graph: &SparseGraph<N, E>,
    index: usize,
) -> Option<SparseNodeId> {
    graph.node_ids.get(index).copied()
}

pub(super) fn dense_to_dense<N, E>(graph: &SparseGraph<N, E>, id: SparseNodeId) -> Option<usize> {
    graph.nodes.get(id).map(|node| node.dense)
}

pub(super) fn dense_node_count<N, E>(graph: &SparseGraph<N, E>) -> usize {
    graph.node_ids.len()
}
