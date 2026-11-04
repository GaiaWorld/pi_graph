//! 双向身份翻译适配器；不分配存储 ID，不过滤原图，不复制业务数据。
#[path = "map_ids_inner.rs"]
mod map_ids_inner;
use crate::*;
use std::hash::Hash;

/// 身份的稳定双向表示；映射须覆盖当前可见元素，且在其上为双射。
/// 对成功转换的身份必须满足正反往返一致；无对应身份返回 None。
/// 遍历期间映射与底层身份不得改变，不保证在底层删除或复用后继续有效。
pub trait IdMapping<I> {
    /// 对外身份，须满足 GraphBase 的身份约束。
    type External: Copy + Eq + Hash;
    /// id：内部身份；有对应表示则返回外部身份，无对应关系 None，不修改状态。
    fn to_external(&self, id: I) -> Option<Self::External>;
    /// id：外部身份；有对应表示则返回内部身份，无对应关系 None，不修改状态。
    fn to_internal(&self, id: Self::External) -> Option<I>;
}

/// 拥有底层图和两类映射，拓扑不变，所有对外身份与端点均翻译。
/// 构造不穷举验证；完备性与单射由调用方和映射实现保证。
/// 已存在元素无法转换时必须 panic，不得静默跳过造成隐式子图。
/// 不自动转发写入、GraphVersion、Search 或 GraphSnapshot。
pub struct MapIds<G, NM, EM> {
    /// 底层图、图引用或上一层适配器。
    _graph: G,
    /// 节点双向映射，覆盖全部当前可见节点。
    _nodes: NM,
    /// 边双向映射，覆盖全部当前可见边。
    _edges: EM,
}
impl<G, NM, EM> MapIds<G, NM, EM> {
    /// graph、nodes、edges：转移底层及映射所有权，不复制图或生成身份。
    /// 调用方保证映射覆盖全部可见元素且为双射；构造本身不验证，不读图。
    pub fn new(graph: G, nodes: NM, edges: EM) -> Self {
        Self::inner_new(graph, nodes, edges)
    }
}
impl<G: GraphBase, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> GraphBase
    for MapIds<G, NM, EM>
{
    type NodeId = NM::External;
    type EdgeId = EM::External;
}
/// 翻译身份的惰性迭代器，顺序和重复计数保持不变。
/// 迭代要求内部候选为 Copy + Eq，以在输出前核验正反往返回到原身份。
pub struct TranslatedIds<'a, M, I> {
    /// 当前稳定映射的借用。
    _mapping: &'a M,
    /// 底层已存在身份的候选状态。
    _iter: I,
}
impl<I: Iterator, M: IdMapping<I::Item>> Iterator for TranslatedIds<'_, M, I>
where
    I::Item: Copy + Eq,
{
    type Item = M::External;
    /// 返回翻译后的下一身份或 None；映射缺失或反查不等于原候选为契约违例，panic，不跳过元素。
    fn next(&mut self) -> Option<Self::Item> {
        // 设计已确认：底层 next -> None 则结束；Some 必须成功翻译且反查等于原候选，否则 panic。
        self.inner_next()
    }
}
/// 翻译后的节点引用，拥有底层引用或 Guard，不复制权重。
pub struct TranslatedNodeRef<R, N> {
    /// 底层引用的唯一所有者，数据借用不超过它的生命周期。
    _reference: R,
    /// 已验证转换的外部节点身份。
    _id: N,
}
/// 翻译后的边引用，方向沿用当前底层视图。
pub struct TranslatedEdgeRef<R, N, E> {
    /// 底层边引用或 Guard 的所有者。
    _reference: R,
    /// 外部边身份。
    _id: E,
    /// 外部源节点身份。
    _source: N,
    /// 外部目标节点身份。
    _target: N,
}
/// 不借用图的翻译访问集合，拥有底层标记和等价映射副本。
/// 副本须与创建时映射完全等价且稳定，不能因 Clone 后共享可变状态改变映射。
pub struct TranslatedVisitMap<I, M, V> {
    /// 对节点身份的稳定映射副本。
    _mapping: M,
    /// 内部身份类型标记，不拥有或分配节点。
    _identity: std::marker::PhantomData<fn(I)>,
    /// 底层图创建的全新空标记集合。
    _visited: V,
}

impl<R: NodeRef, N: Copy> NodeRef for TranslatedNodeRef<R, N> {
    type NodeId = N;

    type Weight = R::Weight;
    /// 返回已翻译的身份/端点或底层权重借用，不修改状态、不延长 Guard 生命周期。
    fn id(&self) -> N {
        self.inner_id()
    }
    /// 返回已翻译的身份/端点或底层权重借用，不修改状态、不延长 Guard 生命周期。
    fn weight(&self) -> &R::Weight {
        self.inner_weight()
    }
}

impl<R: EdgeRef, N: Copy, E: Copy> EdgeRef for TranslatedEdgeRef<R, N, E> {
    type NodeId = N;
    type EdgeId = E;
    type Weight = R::Weight;
    /// 返回已翻译的身份/端点或底层权重借用，不修改状态、不延长 Guard 生命周期。
    fn id(&self) -> E {
        self.inner_id()
    }
    /// 返回已翻译的身份/端点或底层权重借用，不修改状态、不延长 Guard 生命周期。
    fn source(&self) -> N {
        self.inner_source()
    }
    /// 返回已翻译的身份/端点或底层权重借用，不修改状态、不延长 Guard 生命周期。
    fn target(&self) -> N {
        self.inner_target()
    }
    /// 返回已翻译的身份/端点或底层权重借用，不修改状态、不延长 Guard 生命周期。
    fn weight(&self) -> &R::Weight {
        self.inner_weight()
    }
}

impl<G: NodeIds, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> NodeIds for MapIds<G, NM, EM> {
    type NodeIds<'a>
        = TranslatedIds<'a, NM, std::iter::Flatten<std::option::IntoIter<G::NodeIds<'a>>>>
    where
        Self: 'a;
    /// 保留底层顺序，已有身份映射缺失或正反往返不一致时 panic。
    fn node_ids(&self) -> Self::NodeIds<'_> {
        // 设计已确认：底层 node_ids 候选交给 TranslatedIds::next，使用节点映射翻译。
        self.inner_node_ids()
    }
}

impl<G: EdgeIds, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> EdgeIds for MapIds<G, NM, EM> {
    type EdgeIds<'a>
        = TranslatedIds<'a, EM, std::iter::Flatten<std::option::IntoIter<G::EdgeIds<'a>>>>
    where
        Self: 'a;
    /// 保留底层顺序，已有身份映射缺失或正反往返不一致时 panic。
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        // 设计已确认：底层 edge_ids 候选交给 TranslatedIds::next，使用边映射翻译。
        self.inner_edge_ids()
    }
}

impl<G: OutNeighbors, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> OutNeighbors
    for MapIds<G, NM, EM>
{
    type OutNeighbors<'a>
        = TranslatedIds<'a, NM, std::iter::Flatten<std::option::IntoIter<G::OutNeighbors<'a>>>>
    where
        Self: 'a;
    /// node：外部起点，反查未命中为空，底层不存在也为空；保留底层顺序，已有身份映射缺失或正反往返不一致时 panic。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        // 设计已确认：用节点映射反查起点并校验往返；无反查则返回空迭代器。
        // 候选取底层 out_neighbors(内部起点)，沿当前视图出邻居方向；
        // 候选交给 TranslatedIds::next，使用节点映射翻译。
        self.inner_out_neighbors(node)
    }
}

impl<G: InNeighbors, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> InNeighbors
    for MapIds<G, NM, EM>
{
    type InNeighbors<'a>
        = TranslatedIds<'a, NM, std::iter::Flatten<std::option::IntoIter<G::InNeighbors<'a>>>>
    where
        Self: 'a;
    /// node：外部起点，反查未命中为空，底层不存在也为空；保留底层顺序，已有身份映射缺失或正反往返不一致时 panic。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        // 设计已确认：复用 MapIds::out_neighbors 的关联查询主体；
        // 差异为候选取底层 in_neighbors(内部起点)，沿当前视图入邻居方向，
        // 使用节点映射翻译候选。
        self.inner_in_neighbors(node)
    }
}

impl<G: OutEdges, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> OutEdges
    for MapIds<G, NM, EM>
{
    type OutEdges<'a>
        = TranslatedIds<'a, EM, std::iter::Flatten<std::option::IntoIter<G::OutEdges<'a>>>>
    where
        Self: 'a;
    /// node：外部起点，反查未命中为空，底层不存在也为空；保留底层顺序，已有身份映射缺失或正反往返不一致时 panic。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        // 设计已确认：复用 MapIds::out_neighbors 的关联查询主体；
        // 差异为候选取底层 out_edges(内部起点)，沿当前视图出边方向，
        // 使用边映射翻译候选。
        self.inner_out_edges(node)
    }
}

impl<G: InEdges, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> InEdges for MapIds<G, NM, EM> {
    type InEdges<'a>
        = TranslatedIds<'a, EM, std::iter::Flatten<std::option::IntoIter<G::InEdges<'a>>>>
    where
        Self: 'a;
    /// node：外部起点，反查未命中为空，底层不存在也为空；保留底层顺序，已有身份映射缺失或正反往返不一致时 panic。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        // 设计已确认：复用 MapIds::out_neighbors 的关联查询主体；
        // 差异为候选取底层 in_edges(内部起点)，沿当前视图入边方向，
        // 使用边映射翻译候选。
        self.inner_in_edges(node)
    }
}

impl<G: NodeAccess, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> NodeAccess
    for MapIds<G, NM, EM>
{
    type Node = G::Node;
    type NodeRef<'a>
        = TranslatedNodeRef<G::NodeRef<'a>, NM::External>
    where
        Self: 'a;
    /// id：外部身份；无反查或底层不存在 None，否则拥有底层引用并翻译身份。
    /// 已有元素正反转换不一致或缺失为契约违例，panic；异常路径由 RAII 释放 Guard。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        // 设计已确认：节点映射反查 id，无反查返回 None；底层 node 不存在也返回 None。
        // 已有节点校验正反往返并转换引用身份；缺失或不一致则 panic，由 RAII 释放 Guard。
        // 成功时 TranslatedNodeRef 拥有底层引用或 Guard，不复制 weight。
        self.inner_node(id)
    }
}

impl<G: EdgeAccess, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> EdgeAccess
    for MapIds<G, NM, EM>
{
    type Edge = G::Edge;
    type EdgeRef<'a>
        = TranslatedEdgeRef<G::EdgeRef<'a>, NM::External, EM::External>
    where
        Self: 'a;
    /// id：外部身份；无反查或底层不存在 None，否则拥有底层引用并翻译身份与双端点。
    /// 已有元素正反转换不一致或缺失为契约违例，panic；异常路径由 RAII 释放 Guard。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>> {
        // 设计已确认：复用 MapIds::node 的反查、底层访问、往返校验及 Guard 所有权流程；
        // 差异为用边映射访问 edge，翻译边 ID，并用节点映射转换当前方向双端点，
        // 返回 TranslatedEdgeRef；任一已有元数据转换违约仍 panic，不复制 weight。
        self.inner_edge(id)
    }
}

impl<G: EdgeEndpoints, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> EdgeEndpoints
    for MapIds<G, NM, EM>
{
    /// edge：外部边身份；用边映射反查后读取底层当前方向端点，再翻译为外部节点身份。
    /// 无反查或底层边不存在返回 None；已有元素映射缺失或往返违约 panic，不修改图。
    fn endpoints(&self, edge: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)> {
        self.inner_endpoints(edge)
    }
}

impl<G: NodeIndexable, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> NodeIndexable
    for MapIds<G, NM, EM>
{
    /// 返回底层节点槽位索引的排他上界，保留位置与空洞，不修改图。
    fn node_bound(&self) -> usize {
        self.inner_node_bound()
    }
    /// node：有效外部节点身份，不是索引位置；反查内部节点后返回其底层槽位索引。
    /// 调用方须保证节点有效；无反查或映射违约可 panic，不修改图。
    fn to_index(&self, node: Self::NodeId) -> usize {
        self.inner_to_index(node)
    }
    /// index：底层槽位索引位置，不是节点身份；命中后返回翻译的外部节点 ID。
    /// 空洞或越界返回 None；已有节点映射缺失或往返违约 panic，不修改图。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_index(index)
    }
}

impl<G: DenseNodeIndex, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> DenseNodeIndex
    for MapIds<G, NM, EM>
{
    /// 返回底层当前可见节点数量，也是稠密位置的排他上界，不修改图。
    fn node_count(&self) -> usize {
        self.inner_node_count()
    }
    /// id：外部节点身份，不是稠密位置；反查内部节点后返回其底层稠密位置。
    /// 无反查或底层节点无效返回 None；已有节点映射违约 panic，不修改图。
    fn to_dense(&self, id: Self::NodeId) -> Option<usize> {
        self.inner_to_dense(id)
    }
    /// index：底层稠密位置，不是节点身份；命中后返回翻译的外部节点 ID。
    /// 越界返回 None；已有节点映射缺失或往返违约 panic，不修改图。
    fn node_at_dense(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_dense(index)
    }
}

impl<G: Visitable, NM: IdMapping<G::NodeId> + Clone, EM: IdMapping<G::EdgeId>> Visitable
    for MapIds<G, NM, EM>
{
    type Map = TranslatedVisitMap<G::NodeId, NM, G::Map>;
    /// 返回翻译集合；仅此能力要求 NM: Clone，普通读取不要求 Clone。
    /// 映射副本须语义等价，集合不借用图且初始为空；不得在遍历期间修改映射。
    fn visit_map(&self) -> Self::Map {
        self.inner_visit_map()
    }
}
impl<I: Copy, M: IdMapping<I>, V: VisitMap<I>> VisitMap<M::External>
    for TranslatedVisitMap<I, M, V>
{
    /// node：当前视图有效外部节点。翻译后标记，首次 true；无法反查或往返违约 panic。
    fn visit(&mut self, node: M::External) -> bool {
        self.inner_visit(node)
    }
    /// node：当前视图有效外部节点。翻译后查询；无法反查或往返违约 panic。
    fn is_visited(&self, node: M::External) -> bool {
        self.inner_is_visited(node)
    }
}
