//! 视图的配对实现；核心结构与字段真实定义在父壳。
use super::*;
impl<'a, G: EdgeEndpoints + ?Sized, I: Iterator<Item = G::EdgeId>> EdgeNeighbors<'a, G, I> {
    pub(super) fn inner_new(graph: &'a G, edges: I, incoming: bool) -> Self {
        Self {
            graph,
            edges,
            incoming,
        }
    }
}

impl<G: NodeIndexable> Reversed<G> {
    pub(super) fn inner_node_bound(&self) -> usize {
        self.graph.node_bound()
    }
    pub(super) fn inner_to_index(&self, _node: G::NodeId) -> usize {
        self.graph.to_index(_node)
    }
    pub(super) fn inner_node_at_index(&self, _index: usize) -> Option<G::NodeId> {
        self.graph.node_at_index(_index)
    }
}
impl<G: Visitable> Reversed<G> {
    pub(super) fn inner_visit_map(&self) -> G::Map {
        self.graph.visit_map()
    }
}

impl<G: DenseNodeIndex> Reversed<G> {
    pub(super) fn inner_node_count(&self) -> usize {
        self.graph.node_count()
    }
    pub(super) fn inner_to_dense(&self, _id: G::NodeId) -> Option<usize> {
        self.graph.to_dense(_id)
    }
    pub(super) fn inner_node_at_dense(&self, _index: usize) -> Option<G::NodeId> {
        self.graph.node_at_dense(_index)
    }
}

impl<G: NodeIndexable, F> EdgeFiltered<G, F> {
    pub(super) fn inner_node_bound(&self) -> usize {
        self.graph.node_bound()
    }
    pub(super) fn inner_to_index(&self, _node: G::NodeId) -> usize {
        self.graph.to_index(_node)
    }
    pub(super) fn inner_node_at_index(&self, _index: usize) -> Option<G::NodeId> {
        self.graph.node_at_index(_index)
    }
}
impl<G: Visitable, F> EdgeFiltered<G, F> {
    pub(super) fn inner_visit_map(&self) -> G::Map {
        self.graph.visit_map()
    }
}

impl<G: DenseNodeIndex, F> EdgeFiltered<G, F> {
    pub(super) fn inner_node_count(&self) -> usize {
        self.graph.node_count()
    }
    pub(super) fn inner_to_dense(&self, _id: G::NodeId) -> Option<usize> {
        self.graph.to_dense(_id)
    }
    pub(super) fn inner_node_at_dense(&self, _index: usize) -> Option<G::NodeId> {
        self.graph.node_at_dense(_index)
    }
}

impl<R: EdgeRef> ReversedEdgeRef<R> {
    pub(super) fn inner_id(&self) -> R::EdgeId {
        self.reference.id()
    }
    pub(super) fn inner_source(&self) -> R::NodeId {
        self.reference.target()
    }
    pub(super) fn inner_target(&self) -> R::NodeId {
        self.reference.source()
    }
    pub(super) fn inner_weight(&self) -> &R::Weight {
        self.reference.weight()
    }
}

impl<G> Reversed<G> {
    pub(super) fn inner_new(graph: G) -> Self {
        Self { graph }
    }
}

impl<G, F> EdgeFiltered<G, F> {
    pub(super) fn inner_new(graph: G, predicate: F) -> Self {
        Self { graph, predicate }
    }
}

impl<G: NodeIds> Reversed<G> {
    pub(super) fn inner_node_ids(&self) -> <Self as NodeIds>::NodeIds<'_> {
        self.graph.node_ids()
    }
}

impl<G: EdgeIds> Reversed<G> {
    pub(super) fn inner_edge_ids(&self) -> <Self as EdgeIds>::EdgeIds<'_> {
        self.graph.edge_ids()
    }
}

impl<G: InNeighbors> Reversed<G> {
    pub(super) fn inner_out_neighbors(
        &self,
        node: <Self as GraphBase>::NodeId,
    ) -> <Self as OutNeighbors>::OutNeighbors<'_> {
        self.graph.in_neighbors(node)
    }
}

impl<G: OutNeighbors> Reversed<G> {
    pub(super) fn inner_in_neighbors(
        &self,
        node: <Self as GraphBase>::NodeId,
    ) -> <Self as InNeighbors>::InNeighbors<'_> {
        self.graph.out_neighbors(node)
    }
}

impl<G: InEdges> Reversed<G> {
    pub(super) fn inner_out_edges(
        &self,
        node: <Self as GraphBase>::NodeId,
    ) -> <Self as OutEdges>::OutEdges<'_> {
        self.graph.in_edges(node)
    }
}

impl<G: OutEdges> Reversed<G> {
    pub(super) fn inner_in_edges(
        &self,
        node: <Self as GraphBase>::NodeId,
    ) -> <Self as InEdges>::InEdges<'_> {
        self.graph.out_edges(node)
    }
}

impl<G: EdgeEndpoints> Reversed<G> {
    pub(super) fn inner_endpoints(
        &self,
        edge: <Self as GraphBase>::EdgeId,
    ) -> Option<(<Self as GraphBase>::NodeId, <Self as GraphBase>::NodeId)> {
        self.graph
            .endpoints(edge)
            .map(|(source, target)| (target, source))
    }
}

impl<G: NodeAccess> Reversed<G> {
    pub(super) fn inner_node(
        &self,
        id: <Self as GraphBase>::NodeId,
    ) -> Option<<Self as NodeAccess>::NodeRef<'_>> {
        self.graph.node(id)
    }
}

impl<G: EdgeAccess> Reversed<G> {
    pub(super) fn inner_edge(
        &self,
        id: <Self as GraphBase>::EdgeId,
    ) -> Option<<Self as EdgeAccess>::EdgeRef<'_>> {
        self.graph
            .edge(id)
            .map(|reference| ReversedEdgeRef { reference })
    }
}

impl<G: NodeIds, F> EdgeFiltered<G, F> {
    pub(super) fn inner_node_ids(&self) -> <Self as NodeIds>::NodeIds<'_> {
        self.graph.node_ids()
    }
}

impl<G: NodeAccess, F> EdgeFiltered<G, F> {
    pub(super) fn inner_node(
        &self,
        id: <Self as GraphBase>::NodeId,
    ) -> Option<<Self as NodeAccess>::NodeRef<'_>> {
        self.graph.node(id)
    }
}

impl<G: EdgeAccess, F: Fn(G::EdgeId) -> bool> EdgeFiltered<G, F> {
    pub(super) fn inner_edge(
        &self,
        id: <Self as GraphBase>::EdgeId,
    ) -> Option<<Self as EdgeAccess>::EdgeRef<'_>> {
        if (self.predicate)(id) {
            self.graph.edge(id)
        } else {
            None
        }
    }
}

impl<G: EdgeEndpoints, F: Fn(G::EdgeId) -> bool> EdgeFiltered<G, F> {
    pub(super) fn inner_endpoints(
        &self,
        id: <Self as GraphBase>::EdgeId,
    ) -> Option<(<Self as GraphBase>::NodeId, <Self as GraphBase>::NodeId)> {
        if (self.predicate)(id) {
            self.graph.endpoints(id)
        } else {
            None
        }
    }
}

impl<I: Iterator, F: Fn(I::Item) -> bool> FilterEdges<'_, I, F>
where
    I::Item: Copy,
{
    pub(super) fn inner_next(&mut self) -> Option<<Self as Iterator>::Item> {
        self.iter.find(|id| (self.predicate)(*id))
    }
}

impl<G: EdgeEndpoints + ?Sized, I: Iterator<Item = G::EdgeId>> EdgeNeighbors<'_, G, I> {
    pub(super) fn inner_next(&mut self) -> Option<<Self as Iterator>::Item> {
        for edge in self.edges.by_ref() {
            if let Some((source, target)) = self.graph.endpoints(edge) {
                return Some(if self.incoming { source } else { target });
            }
        }
        None
    }
}

impl<G: EdgeIds, F: Fn(G::EdgeId) -> bool> EdgeFiltered<G, F> {
    pub(super) fn inner_edge_ids(&self) -> <Self as EdgeIds>::EdgeIds<'_> {
        FilterEdges {
            iter: self.graph.edge_ids(),
            predicate: &self.predicate,
        }
    }
}

impl<G: OutEdges, F: Fn(G::EdgeId) -> bool> EdgeFiltered<G, F> {
    pub(super) fn inner_out_edges(
        &self,
        node: <Self as GraphBase>::NodeId,
    ) -> <Self as OutEdges>::OutEdges<'_> {
        FilterEdges {
            iter: self.graph.out_edges(node),
            predicate: &self.predicate,
        }
    }
}

impl<G: InEdges, F: Fn(G::EdgeId) -> bool> EdgeFiltered<G, F> {
    pub(super) fn inner_in_edges(
        &self,
        node: <Self as GraphBase>::NodeId,
    ) -> <Self as InEdges>::InEdges<'_> {
        FilterEdges {
            iter: self.graph.in_edges(node),
            predicate: &self.predicate,
        }
    }
}

impl<G: OutEdges + EdgeEndpoints, F: Fn(G::EdgeId) -> bool> EdgeFiltered<G, F> {
    pub(super) fn inner_out_neighbors(
        &self,
        node: <Self as GraphBase>::NodeId,
    ) -> <Self as OutNeighbors>::OutNeighbors<'_> {
        EdgeNeighbors::new(&self.graph, self.out_edges(node), false)
    }
}

impl<G: InEdges + EdgeEndpoints, F: Fn(G::EdgeId) -> bool> EdgeFiltered<G, F> {
    pub(super) fn inner_in_neighbors(
        &self,
        node: <Self as GraphBase>::NodeId,
    ) -> <Self as InNeighbors>::InNeighbors<'_> {
        EdgeNeighbors::new(&self.graph, self.in_edges(node), true)
    }
}
