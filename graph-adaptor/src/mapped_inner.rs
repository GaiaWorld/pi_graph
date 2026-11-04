//! 数据映射配对实现；拥有映射结果并转发底层只读能力。
use super::*;
// 公共 seam：NodeRef / EdgeRef；私有字段仅用于建立独立值测试夹具。
#[cfg(test)]
#[test]
fn t86() {
    let r = MappedEdgeRef {
        _id: 7,
        _source: 10,
        _target: 20,
        _value: String::from("value"),
    };
    assert_eq!(EdgeRef::weight(&r), "value");
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅用于建立独立值测试夹具。
#[cfg(test)]
#[test]
fn t85() {
    let r = MappedEdgeRef {
        _id: 7,
        _source: 10,
        _target: 20,
        _value: String::from("value"),
    };
    assert_eq!(EdgeRef::target(&r), 20);
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅用于建立独立值测试夹具。
#[cfg(test)]
#[test]
fn t84() {
    let r = MappedEdgeRef {
        _id: 7,
        _source: 10,
        _target: 20,
        _value: String::from("value"),
    };
    assert_eq!(EdgeRef::source(&r), 10);
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅用于建立独立值测试夹具。
#[cfg(test)]
#[test]
fn t83() {
    let r = MappedEdgeRef {
        _id: 7,
        _source: 10,
        _target: 20,
        _value: String::from("value"),
    };
    assert_eq!(EdgeRef::id(&r), 7);
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅用于建立独立值测试夹具。
#[cfg(test)]
#[test]
fn t82() {
    let r = MappedNodeRef {
        _id: 10,
        _value: String::from("value"),
    };
    assert_eq!(NodeRef::weight(&r), "value");
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅用于建立独立值测试夹具。
#[cfg(test)]
#[test]
fn t81() {
    let r = MappedNodeRef {
        _id: 10,
        _value: String::from("value"),
    };
    assert_eq!(NodeRef::id(&r), 10);
}

impl<G: NodeIds, F> MapNodes<G, F> {
    pub(super) fn inner_node_ids(&self) -> G::NodeIds<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.node_ids()
    }
}

impl<G: EdgeIds, F> MapNodes<G, F> {
    pub(super) fn inner_edge_ids(&self) -> G::EdgeIds<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.edge_ids()
    }
}

impl<G: OutNeighbors, F> MapNodes<G, F> {
    pub(super) fn inner_out_neighbors(&self, _node: G::NodeId) -> G::OutNeighbors<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.out_neighbors(_node)
    }
}

impl<G: InNeighbors, F> MapNodes<G, F> {
    pub(super) fn inner_in_neighbors(&self, _node: G::NodeId) -> G::InNeighbors<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.in_neighbors(_node)
    }
}

impl<G: OutEdges, F> MapNodes<G, F> {
    pub(super) fn inner_out_edges(&self, _node: G::NodeId) -> G::OutEdges<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.out_edges(_node)
    }
}

impl<G: InEdges, F> MapNodes<G, F> {
    pub(super) fn inner_in_edges(&self, _node: G::NodeId) -> G::InEdges<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.in_edges(_node)
    }
}

impl<G: NodeIds, F> MapEdges<G, F> {
    pub(super) fn inner_node_ids(&self) -> G::NodeIds<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.node_ids()
    }
}

impl<G: EdgeIds, F> MapEdges<G, F> {
    pub(super) fn inner_edge_ids(&self) -> G::EdgeIds<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.edge_ids()
    }
}

impl<G: OutNeighbors, F> MapEdges<G, F> {
    pub(super) fn inner_out_neighbors(&self, _node: G::NodeId) -> G::OutNeighbors<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.out_neighbors(_node)
    }
}

impl<G: InNeighbors, F> MapEdges<G, F> {
    pub(super) fn inner_in_neighbors(&self, _node: G::NodeId) -> G::InNeighbors<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.in_neighbors(_node)
    }
}

impl<G: OutEdges, F> MapEdges<G, F> {
    pub(super) fn inner_out_edges(&self, _node: G::NodeId) -> G::OutEdges<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.out_edges(_node)
    }
}

impl<G: InEdges, F> MapEdges<G, F> {
    pub(super) fn inner_in_edges(&self, _node: G::NodeId) -> G::InEdges<'_> {
        // 原样转发底层拓扑能力，不执行数据映射。
        self._graph.in_edges(_node)
    }
}

impl<G: EdgeEndpoints, F> MapNodes<G, F> {
    pub(super) fn inner_endpoints(&self, _edge: G::EdgeId) -> Option<(G::NodeId, G::NodeId)> {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.endpoints(_edge)
    }
}

impl<G: NodeIndexable, F> MapNodes<G, F> {
    pub(super) fn inner_node_bound(&self) -> usize {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.node_bound()
    }
    pub(super) fn inner_to_index(&self, _node: G::NodeId) -> usize {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.to_index(_node)
    }
    pub(super) fn inner_node_at_index(&self, _index: usize) -> Option<G::NodeId> {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.node_at_index(_index)
    }
}

impl<G: DenseNodeIndex, F> MapNodes<G, F> {
    pub(super) fn inner_node_count(&self) -> usize {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.node_count()
    }
    pub(super) fn inner_to_dense(&self, _id: G::NodeId) -> Option<usize> {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.to_dense(_id)
    }
    pub(super) fn inner_node_at_dense(&self, _index: usize) -> Option<G::NodeId> {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.node_at_dense(_index)
    }
}

impl<G: Visitable, F> MapNodes<G, F> {
    pub(super) fn inner_visit_map(&self) -> G::Map {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.visit_map()
    }
}

impl<G: EdgeEndpoints, F> MapEdges<G, F> {
    pub(super) fn inner_endpoints(&self, _edge: G::EdgeId) -> Option<(G::NodeId, G::NodeId)> {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.endpoints(_edge)
    }
}

impl<G: NodeIndexable, F> MapEdges<G, F> {
    pub(super) fn inner_node_bound(&self) -> usize {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.node_bound()
    }
    pub(super) fn inner_to_index(&self, _node: G::NodeId) -> usize {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.to_index(_node)
    }
    pub(super) fn inner_node_at_index(&self, _index: usize) -> Option<G::NodeId> {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.node_at_index(_index)
    }
}

impl<G: DenseNodeIndex, F> MapEdges<G, F> {
    pub(super) fn inner_node_count(&self) -> usize {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.node_count()
    }
    pub(super) fn inner_to_dense(&self, _id: G::NodeId) -> Option<usize> {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.to_dense(_id)
    }
    pub(super) fn inner_node_at_dense(&self, _index: usize) -> Option<G::NodeId> {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.node_at_dense(_index)
    }
}

impl<G: Visitable, F> MapEdges<G, F> {
    pub(super) fn inner_visit_map(&self) -> G::Map {
        // 透传底层当前方向、位置或空访问集合，不缓存映射结果。
        self._graph.visit_map()
    }
}

impl<N: Copy, V> MappedNodeRef<N, V> {
    pub(super) fn inner_id(&self) -> N {
        // 读取壳侧真实字段，不重新求值映射。
        self._id
    }
    pub(super) fn inner_weight(&self) -> &V {
        // 读取壳侧真实字段，不重新求值映射。
        &self._value
    }
}

impl<N: Copy, E: Copy, V> MappedEdgeRef<N, E, V> {
    pub(super) fn inner_id(&self) -> E {
        // 读取壳侧真实字段，不重新求值映射。
        self._id
    }
    pub(super) fn inner_source(&self) -> N {
        // 读取壳侧真实字段，不重新求值映射。
        self._source
    }
    pub(super) fn inner_target(&self) -> N {
        // 读取壳侧真实字段，不重新求值映射。
        self._target
    }
    pub(super) fn inner_weight(&self) -> &V {
        // 读取壳侧真实字段，不重新求值映射。
        &self._value
    }
}

impl<G, F> MapNodes<G, F> {
    pub(super) fn inner_new(_graph: G, _mapper: F) -> Self {
        // 仅持有图和映射策略。
        Self { _graph, _mapper }
    }
}
impl<G: NodeAccess, F: ValueMapper<G::Node>> MapNodes<G, F> {
    pub(super) fn inner_node(&self, _id: G::NodeId) -> Option<<Self as NodeAccess>::NodeRef<'_>> {
        // 执行壳侧按值映射流程，不返回借自临时 Guard 的数据。
        let reference = self._graph.node(_id)?;
        let id = reference.id();
        let value = self._mapper.map(reference.weight());
        drop(reference);
        Some(MappedNodeRef {
            _id: id,
            _value: value,
        })
    }
}
impl<G: EdgeAccess, F> MapNodes<G, F> {
    pub(super) fn inner_edge(&self, _id: G::EdgeId) -> Option<G::EdgeRef<'_>> {
        // 原样委托底层另一类元素访问。
        self._graph.edge(_id)
    }
}

impl<G, F> MapEdges<G, F> {
    pub(super) fn inner_new(_graph: G, _mapper: F) -> Self {
        // 仅持有图和映射策略。
        Self { _graph, _mapper }
    }
}
impl<G: EdgeAccess, F: ValueMapper<G::Edge>> MapEdges<G, F> {
    pub(super) fn inner_edge(&self, _id: G::EdgeId) -> Option<<Self as EdgeAccess>::EdgeRef<'_>> {
        // 执行壳侧按值映射流程，不返回借自临时 Guard 的数据。
        let reference = self._graph.edge(_id)?;
        let (id, source, target) = (reference.id(), reference.source(), reference.target());
        let value = self._mapper.map(reference.weight());
        drop(reference);
        Some(MappedEdgeRef {
            _id: id,
            _source: source,
            _target: target,
            _value: value,
        })
    }
}
impl<G: NodeAccess, F> MapEdges<G, F> {
    pub(super) fn inner_node(&self, _id: G::NodeId) -> Option<G::NodeRef<'_>> {
        // 原样委托底层另一类元素访问。
        self._graph.node(_id)
    }
}
