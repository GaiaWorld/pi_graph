use graph_core::OutNeighbors;
use graph_query::{Bfs, Walker};

pub(super) fn bfs<G: OutNeighbors + ?Sized>(graph: &G, start: G::NodeId) -> Vec<G::NodeId> {
    Bfs::new(graph, start).iter(graph).collect()
}
