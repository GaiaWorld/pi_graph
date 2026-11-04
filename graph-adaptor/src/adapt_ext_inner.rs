use super::*;
pub(super) fn map_ids<
    G: GraphBase,
    NM: crate::map_ids::IdMapping<G::NodeId>,
    EM: crate::map_ids::IdMapping<G::EdgeId>,
>(
    _graph: G,
    _nodes: NM,
    _edges: EM,
) -> crate::map_ids::MapIds<G, NM, EM> {
    // 通过 MapIds::new 壳组合，不直连另一芯。
    crate::map_ids::MapIds::new(_graph, _nodes, _edges)
}
pub(super) fn indexed_subgraph<
    G: crate::NodeIds + crate::EdgeIds + crate::EdgeEndpoints,
    N: Selection<G::NodeId>,
    E: Selection<G::EdgeId>,
>(
    _graph: G,
    _nodes: N,
    _edges: E,
) -> crate::indexed_subgraph::IndexedBuildResult<G> {
    // 通过 IndexedSubgraph::build 壳组合，保持构建错误与所有权语义。
    crate::indexed_subgraph::IndexedSubgraph::build(_graph, _nodes, _edges)
}
pub(super) fn subgraph<G: GraphBase, N: Selection<G::NodeId>>(
    _graph: G,
    _nodes: N,
) -> Subgraph<G, N, AllEdges> {
    // 通过 Subgraph::induced 壳组合，不直连其他芯。
    Subgraph::induced(_graph, _nodes)
}
pub(super) fn subgraph_with_edges<
    G: GraphBase,
    N: Selection<G::NodeId>,
    E: Selection<G::EdgeId>,
>(
    _graph: G,
    _nodes: N,
    _edges: E,
) -> Subgraph<G, N, SelectedEdges<E>> {
    // 通过 Subgraph::with_edges 壳组合。
    Subgraph::with_edges(_graph, _nodes, _edges)
}
pub(super) fn filter_nodes<G: GraphBase, F: Fn(G::NodeId) -> bool>(
    _graph: G,
    _predicate: F,
) -> Subgraph<G, Predicate<F>, AllEdges> {
    // 包装 Predicate，通过节点过滤壳组合。
    Subgraph::new(_graph, _predicate)
}
pub(super) fn filter_edges<G: GraphBase, F: Fn(G::EdgeId) -> bool>(
    _graph: G,
    _predicate: F,
) -> EdgeFiltered<G, F> {
    // 通过 EdgeFiltered::new 壳组合。
    EdgeFiltered::new(_graph, _predicate)
}
pub(super) fn reversed<G: GraphBase>(_graph: G) -> Reversed<G> {
    // 通过 Reversed::new 壳组合。
    Reversed::new(_graph)
}
pub(super) fn map_nodes<G: NodeAccess, F: ValueMapper<G::Node>>(
    _graph: G,
    _mapper: F,
) -> MapNodes<G, F> {
    // 通过 MapNodes::new 壳组合。
    MapNodes::new(_graph, _mapper)
}
pub(super) fn map_edges<G: EdgeAccess, F: ValueMapper<G::Edge>>(
    _graph: G,
    _mapper: F,
) -> MapEdges<G, F> {
    // 通过 MapEdges::new 壳组合。
    MapEdges::new(_graph, _mapper)
}
