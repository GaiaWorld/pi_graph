use super::{CycleError, TopoWorkspace};
use graph_core::{NodeIds, OutNeighbors};
use std::collections::{HashMap, VecDeque};
use std::fmt;

pub(super) fn workspace_new<N>() -> TopoWorkspace<N> {
    TopoWorkspace {
        _nodes: Vec::new(),
        _indegrees: HashMap::new(),
        _ready: VecDeque::new(),
    }
}
pub(super) fn toposort_with_workspace<G: NodeIds + OutNeighbors + ?Sized>(
    graph: &G,
    workspace: &mut TopoWorkspace<G::NodeId>,
) -> Result<Vec<G::NodeId>, CycleError<G::NodeId>> {
    workspace._nodes.clear();
    workspace._indegrees.clear();
    workspace._ready.clear();
    workspace._nodes.extend(graph.node_ids());
    workspace
        ._indegrees
        .extend(workspace._nodes.iter().map(|&id| (id, 0)));
    for &node in &workspace._nodes {
        for target in graph.out_neighbors(node) {
            let degree = workspace
                ._indegrees
                .get_mut(&target)
                .expect("neighbor must belong to node_ids");
            *degree += 1;
        }
    }
    workspace._ready.extend(
        workspace
            ._nodes
            .iter()
            .copied()
            .filter(|id| workspace._indegrees[id] == 0),
    );
    let mut order = Vec::with_capacity(workspace._nodes.len());
    while let Some(node) = workspace._ready.pop_front() {
        order.push(node);
        for target in graph.out_neighbors(node) {
            let degree = workspace
                ._indegrees
                .get_mut(&target)
                .expect("neighbor must belong to node_ids");
            *degree -= 1;
            if *degree == 0 {
                workspace._ready.push_back(target);
            }
        }
    }
    if order.len() == workspace._nodes.len() {
        Ok(order)
    } else {
        Err(CycleError {
            remaining: workspace
                ._nodes
                .iter()
                .copied()
                .filter(|id| workspace._indegrees[id] > 0)
                .collect(),
        })
    }
}

pub(super) fn format_cycle<N>(error: &CycleError<N>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
        f,
        "graph contains a cycle; {} nodes remain",
        error.remaining.len()
    )
}
pub(super) fn toposort<G: NodeIds + OutNeighbors + ?Sized>(
    graph: &G,
) -> Result<Vec<G::NodeId>, CycleError<G::NodeId>> {
    super::toposort_with_workspace(graph, &mut TopoWorkspace::new())
}
