//! 统一惰性结构子图；拥有底层 G（可为引用或前一层适配器），不复制图。
#[path = "subgraph_inner.rs"]
mod subgraph_inner;
use crate::*;

impl<G: NodeIndexable, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> NodeIndexable
    for Subgraph<G, N, E>
{
    /// 返回底层索引空间的排他上界，保留空洞，不重编号或修改图。
    fn node_bound(&self) -> usize {
        self.inner_node_bound()
    }
    /// node：有效可见的原节点身份；返回底层槽位索引，不重编号、不修改图。
    /// 调用方须保证节点有效且可见，沿用底层有效性前提。
    fn to_index(&self, node: Self::NodeId) -> usize {
        self.inner_to_index(node)
    }
    /// index：底层槽位索引位置；返回该位置的可见原节点身份。
    /// 空洞、越界或节点隐藏返回 None，不修改图。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_index(index)
    }
}

impl<G: Visitable, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> Visitable
    for Subgraph<G, N, E>
{
    type Map = G::Map;
    /// 返回底层创建的全新空集合，各次返回互不影响，不修改图。
    /// 集合可容纳可见节点的超集，不要求重编号。
    fn visit_map(&self) -> Self::Map {
        self.inner_visit_map()
    }
}

use crate::selection::{AllEdges, Predicate, SelectedEdges, Selection};
impl<G, F> Subgraph<G, Predicate<F>, AllEdges> {
    /// graph：底层图或引用；predicate：节点 ID 谓词。保持 NodeFiltered::new 的构造习惯。
    pub fn new(graph: G, predicate: F) -> Self {
        Self::inner_new(graph, Predicate { predicate }, AllEdges)
    }
}
/// 统一节点和边选择；只有存在且被选中的节点，以及边选择通过且双端点可见的边可见。
/// ID、方向及底层枚举顺序不变；选择与拓扑在遍历期间须稳定，不提供快照或重编号。
pub struct Subgraph<G, N, E> {
    /// 拥有底层图或引用，若 G 是适配器则拥有整个中间层。
    _graph: G,
    /// 节点身份选择条件，不代表存在性。
    _nodes: N,
    /// 边身份选择条件，不读取权重；SelectedEdges 表示显式选边。
    _edges: E,
}
impl<G, N> Subgraph<G, N, AllEdges> {
    /// graph：底层图；nodes：节点策略；仅移动所有权，不求值或复制数据。
    pub fn induced(graph: G, nodes: N) -> Self {
        Self::inner_new(graph, nodes, AllEdges)
    }
}
impl<G, N, S> Subgraph<G, N, SelectedEdges<S>> {
    /// graph、nodes、edges：底层图及节点/边选择策略；仅移动所有权，不求值。
    pub fn with_edges(graph: G, nodes: N, edges: S) -> Self {
        Self::inner_new(graph, nodes, SelectedEdges { selection: edges })
    }
}
impl<G: GraphBase, N, E> GraphBase for Subgraph<G, N, E> {
    type NodeId = G::NodeId;
    type EdgeId = G::EdgeId;
}
/// 节点惰性候选，按节点策略筛选并保留重复邻居。
/// 候选身份须 Copy：Selection 按值判定后，迭代器仍返回同一身份；与 GraphBase::NodeId 约束一致。
pub struct SubgraphNodes<'a, N, I> {
    /// 当前节点选择策略借用。
    _nodes: &'a N,
    /// 候选状态；None 表示起点隐藏，不读取底层邻接。
    _iter: Option<I>,
}
/// 边惰性候选，统一检查边策略和双端点。
pub struct SubgraphEdges<'a, G, N, E, I> {
    /// 当前子图借用，保持底层和策略有效。
    _view: &'a Subgraph<G, N, E>,
    /// 候选状态；隐藏起点为 None，平行边独立保留。
    _iter: Option<I>,
}

impl<N: Selection<I::Item>, I: Iterator> Iterator for SubgraphNodes<'_, N, I>
where
    I::Item: Copy,
{
    type Item = I::Item;
    /// 推进候选，返回下一可见身份或 None，不改变图、顺序及平行边计数。
    fn next(&mut self) -> Option<Self::Item> {
        // 设计已确认：_iter 为 None 则返回 None，不读取底层候选。
        // 循环取下一节点；节点策略选中则返回该节点，否则继续；候选耗尽返回 None。
        self.inner_next()
    }
}

impl<
        G: EdgeEndpoints,
        N: Selection<G::NodeId>,
        E: Selection<G::EdgeId>,
        I: Iterator<Item = G::EdgeId>,
    > Iterator for SubgraphEdges<'_, G, N, E, I>
{
    type Item = G::EdgeId;
    /// 推进候选，返回下一可见身份或 None，不改变图、顺序及平行边计数。
    fn next(&mut self) -> Option<Self::Item> {
        // 设计已确认：_iter 为 None 则返回 None，不读取底层候选。
        // 循环取下一边；边策略未选中则继续；读取底层端点，缺失则继续。
        // 两端点均被节点策略选中则返回该边，否则继续；候选耗尽返回 None。
        self.inner_next()
    }
}

impl<G: NodeIds, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> NodeIds for Subgraph<G, N, E> {
    type NodeIds<'a>
        = SubgraphNodes<'a, N, G::NodeIds<'a>>
    where
        Self: 'a;
    /// 返回当前可见身份，保持底层顺序，不修改图。
    fn node_ids(&self) -> Self::NodeIds<'_> {
        // 设计已确认：全局枚举无起点；底层 node_ids 候选交给 SubgraphNodes::next 筛选。
        self.inner_node_ids()
    }
}

impl<G: EdgeIds + EdgeEndpoints, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> EdgeIds
    for Subgraph<G, N, E>
{
    type EdgeIds<'a>
        = SubgraphEdges<'a, G, N, E, G::EdgeIds<'a>>
    where
        Self: 'a;
    /// 返回当前可见身份，保持底层顺序，不修改图。
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        // 设计已确认：全局枚举无起点；底层 edge_ids 候选交给 SubgraphEdges::next 筛选。
        self.inner_edge_ids()
    }
}

impl<G: OutEdges + EdgeEndpoints, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> OutEdges
    for Subgraph<G, N, E>
{
    type OutEdges<'a>
        = SubgraphEdges<'a, G, N, E, G::OutEdges<'a>>
    where
        Self: 'a;
    /// node：查询起点，隐藏或不存在为空。返回当前可见身份，保持底层顺序，不修改图。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        // 设计已确认：节点策略隐藏起点 -> 候选为 None，不调用底层邻接。
        // 否则以底层 out_edges(node) 为候选，交给 SubgraphEdges::next 筛选；
        // 不存在起点的空结果由底层关联边契约保证。
        self.inner_out_edges(node)
    }
}

impl<G: InEdges + EdgeEndpoints, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> InEdges
    for Subgraph<G, N, E>
{
    type InEdges<'a>
        = SubgraphEdges<'a, G, N, E, G::InEdges<'a>>
    where
        Self: 'a;
    /// node：查询起点，隐藏或不存在为空。返回当前可见身份，保持底层顺序，不修改图。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        // 设计已确认：复用 Subgraph::out_edges 的隐藏起点短路与候选筛选流程；
        // 差异仅为候选取底层 in_edges(node)，沿当前视图入边方向。
        self.inner_in_edges(node)
    }
}

impl<G: NodeAccess, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> NodeAccess
    for Subgraph<G, N, E>
{
    type Node = G::Node;
    type NodeRef<'a>
        = G::NodeRef<'a>
    where
        Self: 'a;
    /// id：底层身份；不存在或隐藏返回 None，否则底层引用，Guard 生命周期不变。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        // 设计已确认：先节点选择，再底层访问。
        self.inner_node(id)
    }
}

impl<G: EdgeAccess, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> EdgeAccess
    for Subgraph<G, N, E>
{
    type Edge = G::Edge;
    type EdgeRef<'a>
        = G::EdgeRef<'a>
    where
        Self: 'a;
    /// id：底层身份；不存在或隐藏返回 None，否则底层引用，Guard 生命周期不变。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>> {
        // 设计已确认：先边选择，再底层引用的双端点选择；隐藏则释放引用。
        self.inner_edge(id)
    }
}

impl<G: EdgeEndpoints, N: Selection<G::NodeId>, E: Selection<G::EdgeId>> EdgeEndpoints
    for Subgraph<G, N, E>
{
    /// edge：底层身份；未选中、不存在或任一端点隐藏 None，否则原方向端点。
    fn endpoints(&self, edge: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)> {
        // 设计已确认：边选择 -> 底层端点 -> 双端点选择；无需权重。
        self.inner_endpoints(edge)
    }
}

impl<G: OutNeighbors, N: Selection<G::NodeId>> OutNeighbors for Subgraph<G, N, AllEdges> {
    type OutNeighbors<'a>
        = SubgraphNodes<'a, N, G::OutNeighbors<'a>>
    where
        Self: 'a;
    /// node：查询起点；隐藏或不存在为空，可见邻居保持重复及底层顺序。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        // 设计已确认：节点策略隐藏起点 -> 候选为 None，不调用底层邻接。
        // 否则以底层 out_neighbors(node) 为候选，交给 SubgraphNodes::next 筛选；
        // 不存在起点由底层保证为空，不要求额外边能力。
        self.inner_out_neighbors(node)
    }
}

impl<G: OutEdges + EdgeEndpoints, N: Selection<G::NodeId>, S: Selection<G::EdgeId>> OutNeighbors
    for Subgraph<G, N, SelectedEdges<S>>
{
    type OutNeighbors<'a>
        = crate::view::EdgeNeighbors<'a, Self, <Self as OutEdges>::OutEdges<'a>>
    where
        Self: 'a;
    /// node：查询起点；隐藏或不存在为空，可见邻居保持重复及底层顺序。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        // 设计已确认：起点短路与边筛选复用 Subgraph::out_edges；
        // 其结果交给 view::EdgeNeighbors::next，取当前方向目标端点作为出邻居。
        self.inner_out_neighbors(node)
    }
}

impl<G: InNeighbors, N: Selection<G::NodeId>> InNeighbors for Subgraph<G, N, AllEdges> {
    type InNeighbors<'a>
        = SubgraphNodes<'a, N, G::InNeighbors<'a>>
    where
        Self: 'a;
    /// node：查询起点；隐藏或不存在为空，可见邻居保持重复及底层顺序。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        // 设计已确认：复用 Subgraph<G, N, AllEdges>::out_neighbors 的起点短路流程；
        // 差异仅为候选取底层 in_neighbors(node)，仍由 SubgraphNodes::next 筛选。
        self.inner_in_neighbors(node)
    }
}

impl<G: InEdges + EdgeEndpoints, N: Selection<G::NodeId>, S: Selection<G::EdgeId>> InNeighbors
    for Subgraph<G, N, SelectedEdges<S>>
{
    type InNeighbors<'a>
        = crate::view::EdgeNeighbors<'a, Self, <Self as InEdges>::InEdges<'a>>
    where
        Self: 'a;
    /// node：查询起点；隐藏或不存在为空，可见邻居保持重复及底层顺序。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        // 设计已确认：复用 Subgraph<G, N, SelectedEdges<S>>::out_neighbors 的组合流程；
        // 差异为调用 Subgraph::in_edges，EdgeNeighbors::next 取源端点作为入邻居。
        self.inner_in_neighbors(node)
    }
}
