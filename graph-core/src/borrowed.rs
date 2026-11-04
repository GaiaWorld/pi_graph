//! 图共享引用的逐项能力适配；不借 GraphView blanket impl 推断其他能力。
use crate::*;

impl<G: GraphBase + ?Sized> GraphBase for &G {
    type NodeId = G::NodeId;
    type EdgeId = G::EdgeId;
}

impl<G: NodeIds + ?Sized> NodeIds for &G {
    type NodeIds<'a>
        = G::NodeIds<'a>
    where
        Self: 'a;
    /// 仅转发共享访问，身份、顺序、空结果与底层一致。
    fn node_ids(&self) -> Self::NodeIds<'_> {
        (**self).node_ids()
    }
}

impl<G: EdgeIds + ?Sized> EdgeIds for &G {
    type EdgeIds<'a>
        = G::EdgeIds<'a>
    where
        Self: 'a;
    /// 仅转发共享访问，身份、顺序、空结果与底层一致。
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        (**self).edge_ids()
    }
}

impl<G: OutNeighbors + ?Sized> OutNeighbors for &G {
    type OutNeighbors<'a>
        = G::OutNeighbors<'a>
    where
        Self: 'a;
    /// node：底层节点身份；仅转发共享访问，身份、顺序、空结果与底层一致。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        (**self).out_neighbors(node)
    }
}

impl<G: InNeighbors + ?Sized> InNeighbors for &G {
    type InNeighbors<'a>
        = G::InNeighbors<'a>
    where
        Self: 'a;
    /// node：底层节点身份；仅转发共享访问，身份、顺序、空结果与底层一致。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        (**self).in_neighbors(node)
    }
}

impl<G: OutEdges + ?Sized> OutEdges for &G {
    type OutEdges<'a>
        = G::OutEdges<'a>
    where
        Self: 'a;
    /// node：底层节点身份；仅转发共享访问，身份、顺序、空结果与底层一致。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        (**self).out_edges(node)
    }
}

impl<G: InEdges + ?Sized> InEdges for &G {
    type InEdges<'a>
        = G::InEdges<'a>
    where
        Self: 'a;
    /// node：底层节点身份；仅转发共享访问，身份、顺序、空结果与底层一致。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        (**self).in_edges(node)
    }
}

impl<G: NodeAccess + ?Sized> NodeAccess for &G {
    type Node = G::Node;
    type NodeRef<'a>
        = G::NodeRef<'a>
    where
        Self: 'a;
    /// id：底层身份；返回底层只读引用对象或 None，Guard 及锁约束不变。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        (**self).node(id)
    }
}

impl<G: EdgeAccess + ?Sized> EdgeAccess for &G {
    type Edge = G::Edge;
    type EdgeRef<'a>
        = G::EdgeRef<'a>
    where
        Self: 'a;
    /// id：底层身份；返回底层只读引用对象或 None，Guard 及锁约束不变。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>> {
        (**self).edge(id)
    }
}

impl<G: EdgeEndpoints + ?Sized> EdgeEndpoints for &G {
    /// edge：底层边身份；返回底层当前方向端点或 None，不访问数据。
    fn endpoints(&self, edge: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)> {
        (**self).endpoints(edge)
    }
}
impl<G: NodeIndexable + ?Sized> NodeIndexable for &G {
    /// 返回底层索引上界，不修改图。
    fn node_bound(&self) -> usize {
        (**self).node_bound()
    }
    /// node：有效节点；返回底层原索引，前提不变。
    fn to_index(&self, node: Self::NodeId) -> usize {
        (**self).to_index(node)
    }
    /// index：底层索引；返回有效身份或 None。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        (**self).node_at_index(index)
    }
}
impl<G: DenseNodeIndex + ?Sized> DenseNodeIndex for &G {
    /// 返回底层可见节点数，不修改图。
    fn node_count(&self) -> usize {
        (**self).node_count()
    }
    /// id：待查身份；返回底层稠密位置或 None。
    fn to_dense(&self, id: Self::NodeId) -> Option<usize> {
        (**self).to_dense(id)
    }
    /// index：稠密位置；越界返回 None。
    fn node_at_dense(&self, index: usize) -> Option<Self::NodeId> {
        (**self).node_at_dense(index)
    }
}
impl<G: Visitable + ?Sized> Visitable for &G {
    type Map = G::Map;
    /// 返回底层新建的独立空集合。
    fn visit_map(&self) -> Self::Map {
        (**self).visit_map()
    }
}
impl<G: GraphVersion + ?Sized> GraphVersion for &G {
    type Version = G::Version;
    /// 返回底层拓扑令牌，不提供额外一致性保证。
    fn version(&self) -> Self::Version {
        (**self).version()
    }
}
