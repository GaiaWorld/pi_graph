use super::{Bfs, Dfs, Walker, WalkerIter};
use graph_core::{GraphBase, OutNeighbors, VisitMap};
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

pub(super) fn bfs_with_visit_map<N: Copy, M: VisitMap<N>>(start: N, mut map: M) -> Bfs<N, M> {
    map.visit(start);
    Bfs {
        queue: VecDeque::from([start]),
        visited: map,
    }
}
pub(super) fn dfs_with_visit_map<N: Copy, M: VisitMap<N>>(start: N, map: M) -> Dfs<N, M> {
    Dfs {
        stack: vec![start],
        visited: map,
    }
}

pub(super) fn into_iter<G: GraphBase + ?Sized, W: Walker<G>>(
    walker: W,
    graph: &G,
) -> WalkerIter<'_, G, W> {
    WalkerIter { graph, walker }
}
pub(super) fn iter_next<G: GraphBase + ?Sized, W: Walker<G>>(
    iter: &mut WalkerIter<'_, G, W>,
) -> Option<G::NodeId> {
    iter.walker.next(iter.graph)
}
pub(super) fn bfs_new<N: Copy + Eq + Hash>(start: N) -> Bfs<N> {
    bfs_with_visit_map(start, HashSet::new())
}
pub(super) fn bfs_next<G: OutNeighbors + ?Sized, M: VisitMap<G::NodeId>>(
    state: &mut Bfs<G::NodeId, M>,
    graph: &G,
) -> Option<G::NodeId> {
    let node = state.queue.pop_front()?;
    for next in graph.out_neighbors(node) {
        if state.visited.visit(next) {
            state.queue.push_back(next);
        }
    }
    Some(node)
}
pub(super) fn dfs_new<N: Copy + Eq + Hash>(start: N) -> Dfs<N> {
    dfs_with_visit_map(start, HashSet::new())
}
pub(super) fn dfs_next<G: OutNeighbors + ?Sized, M: VisitMap<G::NodeId>>(
    state: &mut Dfs<G::NodeId, M>,
    graph: &G,
) -> Option<G::NodeId> {
    while let Some(node) = state.stack.pop() {
        if state.visited.visit(node) {
            state.stack.extend(
                graph
                    .out_neighbors(node)
                    .filter(|id| !state.visited.is_visited(*id)),
            );
            return Some(node);
        }
    }
    None
}
