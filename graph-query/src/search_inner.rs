use super::super::walker::Bfs;
use graph_core::{NodeAccess, NodeIds, NodeIndexable, NodeRef, OutNeighbors, VisitMap};

pub(super) fn reachable<G: OutNeighbors + ?Sized>(
    graph: &G,
    start: G::NodeId,
    target: G::NodeId,
) -> bool {
    reachable_with_bfs(graph, Bfs::new(graph, start), target)
}

pub(super) fn reachable_indexed<G: OutNeighbors + NodeIndexable + ?Sized>(
    graph: &G,
    start: G::NodeId,
    target: G::NodeId,
) -> bool {
    reachable_with_bfs(graph, Bfs::new_indexed(graph, start), target)
}

fn reachable_with_bfs<G: OutNeighbors + ?Sized, M: VisitMap<G::NodeId>>(
    graph: &G,
    mut bfs: Bfs<G::NodeId, M>,
    target: G::NodeId,
) -> bool {
    while let Some(node) = bfs.next(graph) {
        if node == target {
            return true;
        }
    }
    false
}
pub(super) fn find_node<G, F>(graph: &G, mut predicate: F) -> Option<G::NodeId>
where
    G: NodeIds + NodeAccess + ?Sized,
    F: FnMut(&G::Node) -> bool,
{
    graph
        .node_ids()
        .find(|id| graph.node(*id).is_some_and(|node| predicate(node.weight())))
}
