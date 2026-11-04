//! 可拥有底层图、图引用或前一适配器的只读视图，不复制节点或边。
#[path = "view_inner.rs"]
mod view_inner;
use crate::*;

impl<G: NodeIndexable> NodeIndexable for Reversed<G> {
    /// 返回底层索引上界，保留原始位置与空洞，不修改图。
    fn node_bound(&self) -> usize {
        self.inner_node_bound()
    }
    /// node：当前视图有效节点。返回原始索引，不重编号；隐藏或无效节点不满足前提。
    fn to_index(&self, node: Self::NodeId) -> usize {
        self.inner_to_index(node)
    }
    /// index：底层位置；越界、空洞返回 None，否则返回原身份。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_index(index)
    }
}
impl<G: Visitable> Visitable for Reversed<G> {
    type Map = G::Map;
    /// 创建底层全新空集合，不复制旧标记，不修改图；可容纳当前可见节点的超集。
    /// 集合仍须匹配底层稳定身份与索引映射，不自动提供拓扑变更安全性。
    fn visit_map(&self) -> Self::Map {
        self.inner_visit_map()
    }
}

impl<G: DenseNodeIndex> DenseNodeIndex for Reversed<G> {
    /// 节点集合未改变，返回底层可见节点数量，不修改图。
    fn node_count(&self) -> usize {
        self.inner_node_count()
    }
    /// id：待查节点；转发底层稠密位置，不存在返回 None。
    fn to_dense(&self, id: Self::NodeId) -> Option<usize> {
        self.inner_to_dense(id)
    }
    /// index：底层稠密位置；越界返回 None，否则返回节点身份。
    fn node_at_dense(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_dense(index)
    }
}

impl<G: NodeIndexable, F> NodeIndexable for EdgeFiltered<G, F> {
    /// 返回底层索引上界，保留原始位置与空洞，不修改图。
    fn node_bound(&self) -> usize {
        self.inner_node_bound()
    }
    /// node：当前视图有效节点。返回原始索引，不重编号；隐藏或无效节点不满足前提。
    fn to_index(&self, node: Self::NodeId) -> usize {
        self.inner_to_index(node)
    }
    /// index：底层位置；越界、空洞返回 None，否则返回原身份。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_index(index)
    }
}
impl<G: Visitable, F> Visitable for EdgeFiltered<G, F> {
    type Map = G::Map;
    /// 创建底层全新空集合，不复制旧标记，不修改图；可容纳当前可见节点的超集。
    /// 集合仍须匹配底层稳定身份与索引映射，不自动提供拓扑变更安全性。
    fn visit_map(&self) -> Self::Map {
        self.inner_visit_map()
    }
}

impl<G: DenseNodeIndex, F> DenseNodeIndex for EdgeFiltered<G, F> {
    /// 节点集合未改变，返回底层可见节点数量，不修改图。
    fn node_count(&self) -> usize {
        self.inner_node_count()
    }
    /// id：待查节点；转发底层稠密位置，不存在返回 None。
    fn to_dense(&self, id: Self::NodeId) -> Option<usize> {
        self.inner_to_dense(id)
    }
    /// index：底层稠密位置；越界返回 None，否则返回节点身份。
    fn node_at_dense(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_dense(index)
    }
}

/// 反向边引用：拥有底层引用或 Guard，仅交换读出的源与目标。
pub struct ReversedEdgeRef<R> {
    /// 底层引用的唯一所有者，保持数据与 Guard 的生命周期。
    reference: R,
}
impl<R: EdgeRef> EdgeRef for ReversedEdgeRef<R> {
    type NodeId = R::NodeId;
    type EdgeId = R::EdgeId;
    type Weight = R::Weight;
    /// 返回底层边身份，不改变状态。
    fn id(&self) -> Self::EdgeId {
        self.inner_id()
    }
    /// 返回底层目标节点，使引用方向与 Reversed::endpoints 一致。
    fn source(&self) -> Self::NodeId {
        self.inner_source()
    }
    /// 返回底层源节点，使引用方向与 Reversed::endpoints 一致。
    fn target(&self) -> Self::NodeId {
        self.inner_target()
    }
    /// 返回借自底层引用对象的数据，不延长 Guard 生命周期。
    fn weight(&self) -> &Self::Weight {
        self.inner_weight()
    }
}

/// 反转全部有向边；节点与边 ID、业务数据保持不变。
pub struct Reversed<G> {
    /// 拥有底层图、图引用或前一层适配器，反向解释其节点与边。
    pub graph: G,
}
impl<G> Reversed<G> {
    /// graph：底层图。返回只读反向视图，不访问或修改存储。
    pub fn new(graph: G) -> Self {
        Self::inner_new(graph)
    }
}

/// 按边 ID 过滤，保留全部节点（包括过滤后产生的孤立节点）。
/// 谓词必须在一次遍历期间返回稳定的结果，并接受任意查询边 ID（包括不存在的 ID）。
/// 谓词返回 true 不证明边存在；需要权重的谓词须自行将缺失访问处理为 false。
pub struct EdgeFiltered<G, F> {
    /// 拥有底层图、图引用或前一层适配器，不复制存储，节点集合完整保留。
    graph: G,
    /// 拥有的边 ID 谓词，同次遍历内判定必须稳定。
    predicate: F,
}
impl<G, F> EdgeFiltered<G, F> {
    /// graph：底层图；predicate：拥有的稳定边 ID 谓词，true 表示可见。
    /// predicate 必须接受任意边 ID，包括不存在的身份；true 不证明底层存在。
    /// 返回保留全部节点的惰性视图；构造时不调用谓词、不修改图。
    pub fn new(graph: G, predicate: F) -> Self {
        Self::inner_new(graph, predicate)
    }
}

impl<G: GraphBase> GraphBase for Reversed<G> {
    type NodeId = G::NodeId;
    type EdgeId = G::EdgeId;
}
impl<G: GraphBase, F> GraphBase for EdgeFiltered<G, F> {
    type NodeId = G::NodeId;
    type EdgeId = G::EdgeId;
}

impl<G: NodeIds> NodeIds for Reversed<G> {
    type NodeIds<'a>
        = G::NodeIds<'a>
    where
        Self: 'a;
    /// 返回全部节点 ID，保留孤立节点与底层顺序，不修改图。
    fn node_ids(&self) -> Self::NodeIds<'_> {
        self.inner_node_ids()
    }
}

impl<G: EdgeIds> EdgeIds for Reversed<G> {
    type EdgeIds<'a>
        = G::EdgeIds<'a>
    where
        Self: 'a;
    /// 返回可见边的惰性枚举，保持底层顺序，不修改图。
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        self.inner_edge_ids()
    }
}

impl<G: InNeighbors> OutNeighbors for Reversed<G> {
    type OutNeighbors<'a>
        = G::InNeighbors<'a>
    where
        Self: 'a;
    /// node：查询节点，不存在返回空枚举；返回出邻居，平行边允许重复。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        self.inner_out_neighbors(node)
    }
}

impl<G: OutNeighbors> InNeighbors for Reversed<G> {
    type InNeighbors<'a>
        = G::OutNeighbors<'a>
    where
        Self: 'a;
    /// node：查询节点，不存在返回空枚举；返回入邻居，平行边允许重复。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        self.inner_in_neighbors(node)
    }
}

impl<G: InEdges> OutEdges for Reversed<G> {
    type OutEdges<'a>
        = G::InEdges<'a>
    where
        Self: 'a;
    /// node：查询节点，不存在返回空枚举；返回可见出边，不修改图。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        self.inner_out_edges(node)
    }
}

impl<G: OutEdges> InEdges for Reversed<G> {
    type InEdges<'a>
        = G::OutEdges<'a>
    where
        Self: 'a;
    /// node：查询节点，不存在返回空枚举；返回可见入边，不修改图。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        self.inner_in_edges(node)
    }
}

impl<G: EdgeEndpoints> EdgeEndpoints for Reversed<G> {
    /// edge：待查边；不存在返回 None，否则交换底层端点并返回 (源, 目标)，不修改图。
    fn endpoints(&self, edge: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)> {
        self.inner_endpoints(edge)
    }
}

impl<G: NodeAccess> NodeAccess for Reversed<G> {
    type Node = G::Node;
    type NodeRef<'a>
        = G::NodeRef<'a>
    where
        Self: 'a;
    /// id：待查节点；存在返回底层借用或 Guard，否则返回 None，不修改图。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        self.inner_node(id)
    }
}

impl<G: EdgeAccess> EdgeAccess for Reversed<G> {
    type Edge = G::Edge;
    type EdgeRef<'a>
        = ReversedEdgeRef<G::EdgeRef<'a>>
    where
        Self: 'a;
    /// id：待查边；可见且存在返回底层借用或 Guard，否则返回 None。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>> {
        self.inner_edge(id)
    }
}

impl<G: NodeIds, F> NodeIds for EdgeFiltered<G, F> {
    type NodeIds<'a>
        = G::NodeIds<'a>
    where
        Self: 'a;
    /// 返回全部节点 ID，保留孤立节点与底层顺序，不修改图。
    fn node_ids(&self) -> Self::NodeIds<'_> {
        self.inner_node_ids()
    }
}
impl<G: NodeAccess, F> NodeAccess for EdgeFiltered<G, F> {
    type Node = G::Node;
    type NodeRef<'a>
        = G::NodeRef<'a>
    where
        Self: 'a;
    /// id：待查节点；存在返回底层借用或 Guard，否则返回 None，不修改图。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        self.inner_node(id)
    }
}
impl<G: EdgeAccess, F: Fn(G::EdgeId) -> bool> EdgeAccess for EdgeFiltered<G, F> {
    type Edge = G::Edge;
    type EdgeRef<'a>
        = G::EdgeRef<'a>
    where
        Self: 'a;
    /// id：待查边；可见且存在返回底层借用或 Guard，否则返回 None。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>> {
        // 设计已确认：先按 ID 判定可见性；不可见返回 None，可见则委托底层访问。
        self.inner_edge(id)
    }
}
impl<G: EdgeEndpoints, F: Fn(G::EdgeId) -> bool> EdgeEndpoints for EdgeFiltered<G, F> {
    /// id：待查边；先调用谓词，不可见或不存在返回 None，否则返回底层 (源, 目标)。
    fn endpoints(&self, id: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)> {
        self.inner_endpoints(id)
    }
}

/// 视图的惰性边迭代器。
pub struct FilterEdges<'a, I, F> {
    /// 底层边 ID 迭代状态，保持原枚举顺序。
    iter: I,
    /// 借用的稳定过滤谓词。
    predicate: &'a F,
}
impl<I: Iterator, F: Fn(I::Item) -> bool> Iterator for FilterEdges<'_, I, F>
where
    I::Item: Copy,
{
    type Item = I::Item;
    /// 推进自身迭代状态，返回下一可见元素或 None，不修改图，保持底层顺序。
    fn next(&mut self) -> Option<Self::Item> {
        // 设计已确认：顺序取边并判定谓词，返回首条可见边；耗尽返回 None。
        self.inner_next()
    }
}

/// 从已过滤的关联边惰性映射邻居 ID。
pub struct EdgeNeighbors<'a, G: ?Sized, I> {
    /// 借用的图，用于将可见边 ID 解析为端点。
    graph: &'a G,
    /// 已过滤关联边的迭代状态，平行边不去重。
    edges: I,
    /// true 返回源端点（入邻居），false 返回目标端点（出邻居）。
    incoming: bool,
}
impl<'a, G: EdgeEndpoints + ?Sized, I: Iterator<Item = G::EdgeId>> EdgeNeighbors<'a, G, I> {
    /// graph：端点查询图；edges：同一图同一方向的可见关联边；incoming：true 取源端点，false 取目标端点。
    /// 调用方保证 edges 均关联同一查询起点且可见；构造不消费候选、不修改图或申请 Guard。
    /// 后续 next 的唯一流程见 Iterator 实现；图在迭代期间须稳定。
    pub(crate) fn new(graph: &'a G, edges: I, incoming: bool) -> Self {
        Self::inner_new(graph, edges, incoming)
    }
}
impl<G: EdgeEndpoints + ?Sized, I: Iterator<Item = G::EdgeId>> Iterator
    for EdgeNeighbors<'_, G, I>
{
    type Item = G::NodeId;
    /// 推进自身迭代状态，返回下一可见元素或 None，不修改图，保持底层顺序。
    fn next(&mut self) -> Option<Self::Item> {
        // 设计已确认：取已过滤关联边 -> 读取端点，缺失则跳过；按方向返回另一端，耗尽返回 None。
        self.inner_next()
    }
}

impl<G: EdgeIds, F: Fn(G::EdgeId) -> bool> EdgeIds for EdgeFiltered<G, F> {
    type EdgeIds<'a>
        = FilterEdges<'a, G::EdgeIds<'a>, F>
    where
        Self: 'a;
    /// 返回可见边的惰性枚举，保持底层顺序，不修改图。
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        self.inner_edge_ids()
    }
}

impl<G: OutEdges, F: Fn(G::EdgeId) -> bool> OutEdges for EdgeFiltered<G, F> {
    type OutEdges<'a>
        = FilterEdges<'a, G::OutEdges<'a>, F>
    where
        Self: 'a;
    /// node：查询节点，不存在返回空枚举；返回可见出边，不修改图。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        self.inner_out_edges(node)
    }
}

impl<G: InEdges, F: Fn(G::EdgeId) -> bool> InEdges for EdgeFiltered<G, F> {
    type InEdges<'a>
        = FilterEdges<'a, G::InEdges<'a>, F>
    where
        Self: 'a;
    /// node：查询节点，不存在返回空枚举；返回可见入边，不修改图。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        self.inner_in_edges(node)
    }
}

impl<G: OutEdges + EdgeEndpoints, F: Fn(G::EdgeId) -> bool> OutNeighbors for EdgeFiltered<G, F> {
    type OutNeighbors<'a>
        = EdgeNeighbors<'a, G, <Self as OutEdges>::OutEdges<'a>>
    where
        Self: 'a;
    /// node：查询节点，不存在返回空枚举；返回出邻居，平行边允许重复。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        self.inner_out_neighbors(node)
    }
}

impl<G: InEdges + EdgeEndpoints, F: Fn(G::EdgeId) -> bool> InNeighbors for EdgeFiltered<G, F> {
    type InNeighbors<'a>
        = EdgeNeighbors<'a, G, <Self as InEdges>::InEdges<'a>>
    where
        Self: 'a;
    /// node：查询节点，不存在返回空枚举；返回入邻居，平行边允许重复。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        self.inner_in_neighbors(node)
    }
}
