use graph_core::OutNeighbors;
use graph_query::{Dfs, Walker};

pub(super) fn dfs<G: OutNeighbors + ?Sized>(graph: &G, start: G::NodeId) -> Vec<G::NodeId> {
    Dfs::new(graph, start).iter(graph).collect()
}
