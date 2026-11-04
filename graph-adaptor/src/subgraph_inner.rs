//! 子图配对实现；按壳侧选择契约筛选候选并转发可选能力。
use super::*;
#[cfg(test)]
#[test]
fn t98_filters_lazily_and_preserves_order_and_duplicates() {
    let calls = std::cell::Cell::new(0);
    let nodes = Predicate {
        predicate: |id| {
            calls.set(calls.get() + 1);
            id != 30
        },
    };
    let mut ids = SubgraphNodes {
        _nodes: &nodes,
        _iter: Some([30, 20, 10, 20, 30].into_iter()),
    };
    assert_eq!(calls.get(), 0);
    assert_eq!(ids.next(), Some(20));
    assert_eq!(calls.get(), 2);
    assert_eq!(ids.collect::<Vec<_>>(), vec![10, 20]);
    let mut hidden = SubgraphNodes::<_, std::iter::Empty<u32>> {
        _nodes: &nodes,
        _iter: None,
    };
    let previous = calls.get();
    assert_eq!(hidden.next(), None);
    assert_eq!(calls.get(), previous);
}
// 公共 seam：Iterator::next；直接字段构造仅用于候选夹具。
#[cfg(test)]
#[test]
fn t99() {
    struct Graph;
    impl GraphBase for Graph {
        type NodeId = u32;
        type EdgeId = u32;
    }
    impl EdgeEndpoints for Graph {
        fn endpoints(&self, id: u32) -> Option<(u32, u32)> {
            match id {
                7 | 8 => Some((10, 20)),
                9 => Some((10, 30)),
                _ => None,
            }
        }
    }
    let view = Subgraph::with_edges(
        Graph,
        Predicate {
            predicate: |id| id != 30,
        },
        Predicate {
            predicate: |id| id != 8,
        },
    );
    let mut iter = SubgraphEdges {
        _view: &view,
        _iter: Some([0, 7, 8, 9, 7].into_iter()),
    };
    assert_eq!(iter.next(), Some(7));
    assert_eq!(iter.next(), Some(7));
    assert_eq!(iter.next(), None);
}

impl<G: NodeIndexable, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> Subgraph<G, N, E> {
    pub(super) fn inner_node_bound(&self) -> usize {
        // 遵守壳侧稀疏位置与可见性规则，或创建底层独立空集合。
        self._graph.node_bound()
    }
    pub(super) fn inner_to_index(&self, _node: G::NodeId) -> usize {
        // 遵守壳侧稀疏位置与可见性规则，或创建底层独立空集合。
        self._graph.to_index(_node)
    }
    pub(super) fn inner_node_at_index(&self, _index: usize) -> Option<G::NodeId> {
        // 遵守壳侧稀疏位置与可见性规则，或创建底层独立空集合。
        self._graph
            .node_at_index(_index)
            .filter(|&id| self._nodes.contains(id))
    }
}

impl<G: Visitable, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> Subgraph<G, N, E> {
    pub(super) fn inner_visit_map(&self) -> G::Map {
        // 遵守壳侧稀疏位置与可见性规则，或创建底层独立空集合。
        self._graph.visit_map()
    }
}

impl<G, N, E> Subgraph<G, N, E> {
    pub(super) fn inner_new(_graph: G, _nodes: N, _edges: E) -> Self {
        Self {
            _graph,
            _nodes,
            _edges,
        }
    }
}

impl<N: Selection<I::Item>, I: Iterator> SubgraphNodes<'_, N, I>
where
    I::Item: Copy,
{
    pub(super) fn inner_next(&mut self) -> Option<I::Item> {
        // 执行壳侧统一选择流程。
        self._iter.as_mut()?.find(|id| self._nodes.contains(*id))
    }
}

impl<
        G: EdgeEndpoints,
        N: Selection<G::NodeId>,
        E: Selection<G::EdgeId>,
        I: Iterator<Item = G::EdgeId>,
    > SubgraphEdges<'_, G, N, E, I>
{
    pub(super) fn inner_next(&mut self) -> Option<G::EdgeId> {
        // 执行壳侧统一选择流程。
        let view = self._view;
        self._iter.as_mut()?.find(|&edge| {
            view._edges.contains(edge)
                && view._graph.endpoints(edge).is_some_and(|(source, target)| {
                    view._nodes.contains(source) && view._nodes.contains(target)
                })
        })
    }
}

impl<G: NodeIds, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> Subgraph<G, N, E> {
    pub(super) fn inner_node_ids(&self) -> <Self as NodeIds>::NodeIds<'_> {
        // 构造惰性候选，隐藏起点短路。
        SubgraphNodes {
            _nodes: &self._nodes,
            _iter: Some(self._graph.node_ids()),
        }
    }
}

impl<G: EdgeIds + EdgeEndpoints, N: Selection<G::NodeId>, E: Selection<G::EdgeId>>
    Subgraph<G, N, E>
{
    pub(super) fn inner_edge_ids(&self) -> <Self as EdgeIds>::EdgeIds<'_> {
        // 构造惰性候选，隐藏起点短路。
        SubgraphEdges {
            _view: self,
            _iter: Some(self._graph.edge_ids()),
        }
    }
}

impl<G: OutEdges + EdgeEndpoints, N: Selection<G::NodeId>, E: Selection<G::EdgeId>>
    Subgraph<G, N, E>
{
    pub(super) fn inner_out_edges(&self, _node: G::NodeId) -> <Self as OutEdges>::OutEdges<'_> {
        // 构造惰性候选，隐藏起点短路。
        SubgraphEdges {
            _view: self,
            _iter: self
                ._nodes
                .contains(_node)
                .then(|| self._graph.out_edges(_node)),
        }
    }
}

impl<G: InEdges + EdgeEndpoints, N: Selection<G::NodeId>, E: Selection<G::EdgeId>>
    Subgraph<G, N, E>
{
    pub(super) fn inner_in_edges(&self, _node: G::NodeId) -> <Self as InEdges>::InEdges<'_> {
        // 构造惰性候选，隐藏起点短路。
        SubgraphEdges {
            _view: self,
            _iter: self
                ._nodes
                .contains(_node)
                .then(|| self._graph.in_edges(_node)),
        }
    }
}

impl<G: NodeAccess, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> Subgraph<G, N, E> {
    pub(super) fn inner_node(&self, _id: G::NodeId) -> Option<G::NodeRef<'_>> {
        // 筛选后返回底层访问对象，不强制额外访问能力。
        self._nodes
            .contains(_id)
            .then(|| self._graph.node(_id))
            .flatten()
    }
}

impl<G: EdgeAccess, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> Subgraph<G, N, E> {
    pub(super) fn inner_edge(&self, _id: G::EdgeId) -> Option<G::EdgeRef<'_>> {
        // 筛选后返回底层访问对象，不强制额外访问能力。
        if !self._edges.contains(_id) {
            return None;
        }
        let reference = self._graph.edge(_id)?;
        (self._nodes.contains(reference.source()) && self._nodes.contains(reference.target()))
            .then_some(reference)
    }
}

impl<G: EdgeEndpoints, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> Subgraph<G, N, E> {
    pub(super) fn inner_endpoints(&self, _edge: G::EdgeId) -> Option<(G::NodeId, G::NodeId)> {
        // 执行边及双端点可见性规则。
        if !self._edges.contains(_edge) {
            return None;
        }
        self._graph
            .endpoints(_edge)
            .filter(|&(s, t)| self._nodes.contains(s) && self._nodes.contains(t))
    }
}

impl<G: OutNeighbors, N: Selection<G::NodeId>> Subgraph<G, N, AllEdges> {
    pub(super) fn inner_out_neighbors(
        &self,
        _node: G::NodeId,
    ) -> <Self as OutNeighbors>::OutNeighbors<'_> {
        // 通过对应壳侧候选入口组合，不调用其他芯。
        SubgraphNodes {
            _nodes: &self._nodes,
            _iter: self
                ._nodes
                .contains(_node)
                .then(|| self._graph.out_neighbors(_node)),
        }
    }
}

impl<G: OutEdges + EdgeEndpoints, N: Selection<G::NodeId>, S: Selection<G::EdgeId>>
    Subgraph<G, N, SelectedEdges<S>>
{
    pub(super) fn inner_out_neighbors(
        &self,
        _node: G::NodeId,
    ) -> <Self as OutNeighbors>::OutNeighbors<'_> {
        // 通过对应壳侧候选入口组合，不调用其他芯。
        crate::view::EdgeNeighbors::new(self, self.out_edges(_node), false)
    }
}

impl<G: InNeighbors, N: Selection<G::NodeId>> Subgraph<G, N, AllEdges> {
    pub(super) fn inner_in_neighbors(
        &self,
        _node: G::NodeId,
    ) -> <Self as InNeighbors>::InNeighbors<'_> {
        // 通过对应壳侧候选入口组合，不调用其他芯。
        SubgraphNodes {
            _nodes: &self._nodes,
            _iter: self
                ._nodes
                .contains(_node)
                .then(|| self._graph.in_neighbors(_node)),
        }
    }
}

impl<G: InEdges + EdgeEndpoints, N: Selection<G::NodeId>, S: Selection<G::EdgeId>>
    Subgraph<G, N, SelectedEdges<S>>
{
    pub(super) fn inner_in_neighbors(
        &self,
        _node: G::NodeId,
    ) -> <Self as InNeighbors>::InNeighbors<'_> {
        // 通过对应壳侧候选入口组合，不调用其他芯。
        crate::view::EdgeNeighbors::new(self, self.in_edges(_node), true)
    }
}
