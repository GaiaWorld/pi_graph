//! 按值数据映射，不改变拓扑、身份或方向；零拷贝借用表示使用独立 Project。
#[path = "mapped_inner.rs"]
mod mapped_inner;
use crate::*;

impl<G: NodeIds, F> NodeIds for MapNodes<G, F> {
    type NodeIds<'a>
        = G::NodeIds<'a>
    where
        Self: 'a;
    /// 拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn node_ids(&self) -> Self::NodeIds<'_> {
        self.inner_node_ids()
    }
}

impl<G: EdgeIds, F> EdgeIds for MapNodes<G, F> {
    type EdgeIds<'a>
        = G::EdgeIds<'a>
    where
        Self: 'a;
    /// 拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        self.inner_edge_ids()
    }
}

impl<G: OutNeighbors, F> OutNeighbors for MapNodes<G, F> {
    type OutNeighbors<'a>
        = G::OutNeighbors<'a>
    where
        Self: 'a;
    /// node：待查节点；拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        self.inner_out_neighbors(node)
    }
}

impl<G: InNeighbors, F> InNeighbors for MapNodes<G, F> {
    type InNeighbors<'a>
        = G::InNeighbors<'a>
    where
        Self: 'a;
    /// node：待查节点；拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        self.inner_in_neighbors(node)
    }
}

impl<G: OutEdges, F> OutEdges for MapNodes<G, F> {
    type OutEdges<'a>
        = G::OutEdges<'a>
    where
        Self: 'a;
    /// node：待查节点；拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        self.inner_out_edges(node)
    }
}

impl<G: InEdges, F> InEdges for MapNodes<G, F> {
    type InEdges<'a>
        = G::InEdges<'a>
    where
        Self: 'a;
    /// node：待查节点；拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        self.inner_in_edges(node)
    }
}

impl<G: NodeIds, F> NodeIds for MapEdges<G, F> {
    type NodeIds<'a>
        = G::NodeIds<'a>
    where
        Self: 'a;
    /// 拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn node_ids(&self) -> Self::NodeIds<'_> {
        self.inner_node_ids()
    }
}

impl<G: EdgeIds, F> EdgeIds for MapEdges<G, F> {
    type EdgeIds<'a>
        = G::EdgeIds<'a>
    where
        Self: 'a;
    /// 拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        self.inner_edge_ids()
    }
}

impl<G: OutNeighbors, F> OutNeighbors for MapEdges<G, F> {
    type OutNeighbors<'a>
        = G::OutNeighbors<'a>
    where
        Self: 'a;
    /// node：待查节点；拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        self.inner_out_neighbors(node)
    }
}

impl<G: InNeighbors, F> InNeighbors for MapEdges<G, F> {
    type InNeighbors<'a>
        = G::InNeighbors<'a>
    where
        Self: 'a;
    /// node：待查节点；拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        self.inner_in_neighbors(node)
    }
}

impl<G: OutEdges, F> OutEdges for MapEdges<G, F> {
    type OutEdges<'a>
        = G::OutEdges<'a>
    where
        Self: 'a;
    /// node：待查节点；拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        self.inner_out_edges(node)
    }
}

impl<G: InEdges, F> InEdges for MapEdges<G, F> {
    type InEdges<'a>
        = G::InEdges<'a>
    where
        Self: 'a;
    /// node：待查节点；拓扑未改变，返回底层枚举，身份、顺序及空结果语义不变，不求值映射。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        self.inner_in_edges(node)
    }
}

impl<G: EdgeEndpoints, F> EdgeEndpoints for MapNodes<G, F> {
    /// edge：底层边 ID；返回当前底层方向的双端点，边不存在返回 None。
    /// 不修改图或求值映射。
    fn endpoints(&self, edge: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)> {
        self.inner_endpoints(edge)
    }
}

impl<G: NodeIndexable, F> NodeIndexable for MapNodes<G, F> {
    /// 原样保留底层能力语义，不修改图或求值映射。
    fn node_bound(&self) -> usize {
        self.inner_node_bound()
    }
    /// node：底层有效节点 ID；返回其原槽位索引，调用方必须保证节点有效。
    /// 无效节点的行为沿用底层前提，可 panic。
    /// 不修改图或求值映射。
    fn to_index(&self, node: Self::NodeId) -> usize {
        self.inner_to_index(node)
    }
    /// index：底层原槽位索引位置；返回该位置的原节点 ID，空洞或越界返回 None。
    /// 不重编号、不修改图或求值映射。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_index(index)
    }
}

impl<G: DenseNodeIndex, F> DenseNodeIndex for MapNodes<G, F> {
    /// 原样保留底层能力语义，不修改图或求值映射。
    fn node_count(&self) -> usize {
        self.inner_node_count()
    }
    /// id：底层节点 ID；有效节点返回原稠密位置，无效节点返回 None。
    /// 不修改图或求值映射。
    fn to_dense(&self, id: Self::NodeId) -> Option<usize> {
        self.inner_to_dense(id)
    }
    /// index：底层稠密位置；返回对应原节点 ID，越界返回 None。
    /// 保留底层稠密映射，不修改图或求值映射。
    fn node_at_dense(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_dense(index)
    }
}

impl<G: Visitable, F> Visitable for MapNodes<G, F> {
    type Map = G::Map;
    /// 原样保留底层能力语义，不修改图或求值映射。
    fn visit_map(&self) -> Self::Map {
        self.inner_visit_map()
    }
}

impl<G: EdgeEndpoints, F> EdgeEndpoints for MapEdges<G, F> {
    /// edge：底层边 ID；返回当前底层方向的双端点，边不存在返回 None。
    /// 不修改图或求值映射。
    fn endpoints(&self, edge: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)> {
        self.inner_endpoints(edge)
    }
}

impl<G: NodeIndexable, F> NodeIndexable for MapEdges<G, F> {
    /// 原样保留底层能力语义，不修改图或求值映射。
    fn node_bound(&self) -> usize {
        self.inner_node_bound()
    }
    /// node：底层有效节点 ID；返回其原槽位索引，调用方必须保证节点有效。
    /// 无效节点的行为沿用底层前提，可 panic。
    /// 不修改图或求值映射。
    fn to_index(&self, node: Self::NodeId) -> usize {
        self.inner_to_index(node)
    }
    /// index：底层原槽位索引位置；返回该位置的原节点 ID，空洞或越界返回 None。
    /// 不重编号、不修改图或求值映射。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_index(index)
    }
}

impl<G: DenseNodeIndex, F> DenseNodeIndex for MapEdges<G, F> {
    /// 原样保留底层能力语义，不修改图或求值映射。
    fn node_count(&self) -> usize {
        self.inner_node_count()
    }
    /// id：底层节点 ID；有效节点返回原稠密位置，无效节点返回 None。
    /// 不修改图或求值映射。
    fn to_dense(&self, id: Self::NodeId) -> Option<usize> {
        self.inner_to_dense(id)
    }
    /// index：底层稠密位置；返回对应原节点 ID，越界返回 None。
    /// 保留底层稠密映射，不修改图或求值映射。
    fn node_at_dense(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_dense(index)
    }
}

impl<G: Visitable, F> Visitable for MapEdges<G, F> {
    type Map = G::Map;
    /// 原样保留底层能力语义，不修改图或求值映射。
    fn visit_map(&self) -> Self::Map {
        self.inner_visit_map()
    }
}

/// 无借用逃逸的按值映射策略，Output 的类型独立于 map 调用的输入借用。
pub trait ValueMapper<T: ?Sized> {
    /// 映射结果，允许非 Clone；不得借用本次输入临时 Guard。
    type Output;
    /// value：本次只读数据，调用期有效；返回独立值。不得修改图，panic 原样传播。
    fn map(&self, value: &T) -> Self::Output;
}
impl<T: ?Sized, U, F: Fn(&T) -> U> ValueMapper<T> for F {
    type Output = U;
    fn map(&self, value: &T) -> U {
        self(value)
    }
}
/// 拥有按值计算结果的节点引用，不再依赖计算时的底层 Guard。
pub struct MappedNodeRef<N, V> {
    /// 原节点身份。
    _id: N,
    /// 本次访问计算得到的独立数据，不缓存至图。
    _value: V,
}
/// 拥有按值计算结果的边引用，端点来自当前底层视图方向。
pub struct MappedEdgeRef<N, E, V> {
    /// 原边身份。
    _id: E,
    /// 当前视图源节点。
    _source: N,
    /// 当前视图目标节点。
    _target: N,
    /// 本次访问产生的独立数据。
    _value: V,
}

impl<N: Copy, V> NodeRef for MappedNodeRef<N, V> {
    type NodeId = N;

    type Weight = V;
    /// 返回当前映射对象的身份或方向信息，不改变状态。
    fn id(&self) -> N {
        self.inner_id()
    }
    /// 返回当前映射对象的数据借用，不延长对象生命周期。
    fn weight(&self) -> &V {
        self.inner_weight()
    }
}

impl<N: Copy, E: Copy, V> EdgeRef for MappedEdgeRef<N, E, V> {
    type NodeId = N;
    type EdgeId = E;
    type Weight = V;
    /// 返回当前映射对象的身份或方向信息，不改变状态。
    fn id(&self) -> E {
        self.inner_id()
    }
    /// 返回当前映射对象的身份或方向信息，不改变状态。
    fn source(&self) -> N {
        self.inner_source()
    }
    /// 返回当前映射对象的身份或方向信息，不改变状态。
    fn target(&self) -> N {
        self.inner_target()
    }
    /// 返回当前映射对象的数据借用，不延长对象生命周期。
    fn weight(&self) -> &V {
        self.inner_weight()
    }
}

/// 节点按值数据适配器，拥有底层或前一层适配器，不复制图，不提供修改接口。
pub struct MapNodes<G, F> {
    /// 底层图、图引用或适配器。
    _graph: G,
    /// 只读值映射策略；每次访问重新求值，不承诺自动缓存或 Snapshot。
    _mapper: F,
}
impl<G, F> MapNodes<G, F> {
    /// graph、mapper：转移所有权；不求值、不复制图或生成 ID。
    pub fn new(graph: G, mapper: F) -> Self {
        Self::inner_new(graph, mapper)
    }
}
impl<G: GraphBase, F> GraphBase for MapNodes<G, F> {
    type NodeId = G::NodeId;
    type EdgeId = G::EdgeId;
}
impl<G: NodeAccess, F: ValueMapper<G::Node>> NodeAccess for MapNodes<G, F> {
    type Node = F::Output;
    type NodeRef<'a>
        = MappedNodeRef<G::NodeId, F::Output>
    where
        Self: 'a;
    /// id：底层身份；不存在 None，否则只求值一次并返回拥有结果的引用对象。
    /// 计算时保留底层 Guard，完成后释放；映射 panic 时 RAII 释放，不修改图。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        // 设计已确认：底层访问 -> 保存身份 -> 借用 weight 计算独立值 -> 释放底层引用 -> 返回拥有值的对象。
        self.inner_node(id)
    }
}
impl<G: EdgeAccess, F> EdgeAccess for MapNodes<G, F> {
    type Edge = G::Edge;
    type EdgeRef<'a>
        = G::EdgeRef<'a>
    where
        Self: 'a;
    /// id：未映射的元素身份，原样转发底层访问与 Guard，不执行映射。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>> {
        self.inner_edge(id)
    }
}

/// 边按值数据适配器，拥有底层或前一层适配器，不复制图，不提供修改接口。
pub struct MapEdges<G, F> {
    /// 底层图、图引用或适配器。
    _graph: G,
    /// 只读值映射策略；每次访问重新求值，不承诺自动缓存或 Snapshot。
    _mapper: F,
}
impl<G, F> MapEdges<G, F> {
    /// graph、mapper：转移所有权；不求值、不复制图或生成 ID。
    pub fn new(graph: G, mapper: F) -> Self {
        Self::inner_new(graph, mapper)
    }
}
impl<G: GraphBase, F> GraphBase for MapEdges<G, F> {
    type NodeId = G::NodeId;
    type EdgeId = G::EdgeId;
}
impl<G: EdgeAccess, F: ValueMapper<G::Edge>> EdgeAccess for MapEdges<G, F> {
    type Edge = F::Output;
    type EdgeRef<'a>
        = MappedEdgeRef<G::NodeId, G::EdgeId, F::Output>
    where
        Self: 'a;
    /// id：底层身份；不存在 None，否则只求值一次并返回拥有结果的引用对象。
    /// 计算时保留底层 Guard，完成后释放；映射 panic 时 RAII 释放，不修改图。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>> {
        // 设计已确认：复用 MapNodes::node 的 Guard 存活、按值映射与释放流程；
        // 差异为访问底层 edge，保存边 ID 及当前方向双端点，返回 MappedEdgeRef。
        self.inner_edge(id)
    }
}
impl<G: NodeAccess, F> NodeAccess for MapEdges<G, F> {
    type Node = G::Node;
    type NodeRef<'a>
        = G::NodeRef<'a>
    where
        Self: 'a;
    /// id：未映射的元素身份，原样转发底层访问与 Guard，不执行映射。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        self.inner_node(id)
    }
}
