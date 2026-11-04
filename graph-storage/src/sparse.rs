//! SlotMap 稳定代际身份及共享 deque 邻接；边的双向链接均由存储私有维护。
use crate::{GraphError, SlotVisitMap};
use graph_core::{DenseNodeIndex, NodeIndexable};

#[path = "sparse_inner.rs"]
mod sparse_inner;
use pi_slot_deque::{Deque, DequeStorage};
use pi_slotmap::{new_key_type, Key, SlotMap};

new_key_type! {
    /// 稀疏节点代际键，仅可用于创建它的图；失效保证限于同槽位代际尚未回绕的区间。
    pub struct SparseNodeId;
    /// 稀疏边代际键；自环和平行边分别拥有独立身份，回绕边界与 SparseGraph 相同。
    pub struct SparseEdgeId;
    /// 私有邻接链接代际键；仅属于本图的共享链接池。
    struct LinkId;
}
/// 稀疏节点生产状态；dense 与 node_ids 互为反查，邻接链接来自本图共享池。
struct Node<N> {
    /// 本存储拥有的业务权重；可变访问不递增拓扑版本。
    weight: N,
    /// 在紧凑身份表中的位置；末项填洞时同步反查。
    dense: usize,
    /// 按插入顺序保存的出边关联；自环同时占据出入两个关联。
    outgoing: Deque<LinkId, SparseEdgeId>,
    /// 按插入顺序保存的入边关联；删除保留其余关联相对顺序。
    incoming: Deque<LinkId, SparseEdgeId>,
}
/// 稀疏边生产状态；两个链接分别属于源出队列和目标入队列。
struct Edge<E> {
    /// 本存储拥有的业务权重；可变访问不递增拓扑版本。
    weight: E,
    /// 源节点的当前完整身份；必须与源节点出边关联一致。
    source: SparseNodeId,
    /// 目标节点的当前完整身份；必须与目标节点入边关联一致。
    target: SparseNodeId,
    /// 在紧凑身份表中的位置；末项填洞时同步反查。
    dense: usize,
    /// 共享链接池中源节点出向关联的键；必须指向本边。
    out_link: LinkId,
    /// 共享链接池中目标节点入向关联的键；必须指向本边。
    in_link: LinkId,
}

/// 稳定 ID 的有向多重图；节点/边插入与单边删除为 O(1)，删节点为 O(关联边数)。
/// 枚举按紧凑身份表顺序，删除时末项填洞；邻接按边插入顺序。
/// pi_slotmap 0.5 默认槽位代际在同槽位约 2^31 次删除/复用后重复；不得将失效键保留至回绕后使用。
/// u64 拓扑版本不阻止较早的槽位回绕；不提供整个图生命周期内旧键永不复活的保证。
/// Clone 的结构成本为 O(Sn + Se + Bn + V + E)：Sn/Se 为已初始化节点/边槽位数（含空洞），
/// Bn 为节点槽位反查表长度，V/E 为活节点/边数；权重 Clone 成本另计。
pub struct SparseGraph<N, E> {
    /// 节点主存储；每个有效身份仅对应一项，内部反查与关联保持一致。
    nodes: SlotMap<SparseNodeId, Node<N>>,
    /// 边主存储；端点与双向邻接保持一致。
    edges: SlotMap<SparseEdgeId, Edge<E>>,
    /// 全图共享双向关联池；每条活边拥有彼此独立的出、入链接。
    links: DequeStorage<LinkId, SparseEdgeId>,
    /// 紧凑节点身份表；与节点 dense 字段互为反查。
    node_ids: Vec<SparseNodeId>,
    /// 紧凑边身份表；与边 dense 字段互为反查。
    edge_ids: Vec<SparseEdgeId>,
    // 槽位反查保留完整代际键，避免通过索引扫描或伪造当前版本。
    /// 槽位到完整代际键的反查；空槽为 None，删除后清空。
    slots: Vec<Option<SparseNodeId>>,
    /// 成功拓扑修改的单调版本；权重修改与无操作删除不改变它。
    version: u64,
}

/// 私有链接类型不泄漏给调用者；空或无效节点产生空迭代。
pub struct SparseEdges<'a> {
    /// 借用现存关联序列的迭代状态；无效节点为 None，不承载图主状态。
    inner: Option<pi_slot_deque::Iter<'a, LinkId, SparseEdgeId>>,
}
impl Iterator for SparseEdges<'_> {
    type Item = SparseEdgeId;
    /// self 为剩余借用迭代状态；返回下一个身份，耗尽为 None，只推进迭代器、不修改图。
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.as_mut()?.next().copied()
    }
}

impl<N, E> SparseGraph<N, E> {
    /// 创建空存储；无参数，返回独立空状态，不修改既有对象。
    pub fn new() -> Self {
        sparse_inner::new()
    }
    /// nodes、edges 分别为节点和边的预留容量，不创建元素；返回版本为零的空图。
    /// 容量分配沿用 Rust 容器行为，无业务错误。
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        sparse_inner::with_capacity(nodes, edges)
    }
    /// self 为当前图；返回有效节点数，无副作用。
    pub fn node_count(&self) -> usize {
        sparse_inner::node_count(self)
    }
    /// self 为当前图；返回有效边数，无副作用。
    pub fn edge_count(&self) -> usize {
        sparse_inner::edge_count(self)
    }
    /// self 为当前图；返回下一拓扑版本，耗尽返回 VersionExhausted；不修改状态。
    fn next_version(&self) -> Result<u64, GraphError> {
        sparse_inner::next_version(self)
    }
    /// self 为当前图；返回借用的节点身份枚举，按当前紧凑存储顺序，无副作用。
    fn node_keys(&self) -> std::iter::Copied<std::slice::Iter<'_, SparseNodeId>> {
        sparse_inner::node_keys(self)
    }
    /// self 为当前图；返回借用的边身份枚举，按当前紧凑存储顺序，无副作用。
    fn edge_keys(&self) -> std::iter::Copied<std::slice::Iter<'_, SparseEdgeId>> {
        sparse_inner::edge_keys(self)
    }
    /// id 为本图节点身份；返回共享权重借用，失效为 None，不修改状态。
    fn node_weight(&self, id: SparseNodeId) -> Option<&N> {
        sparse_inner::node_weight(self, id)
    }
    /// id 为本图节点身份；返回独占权重借用，失效为 None；修改权重不改变拓扑版本。
    fn node_weight_mut(&mut self, id: SparseNodeId) -> Option<&mut N> {
        sparse_inner::node_weight_mut(self, id)
    }
    /// id 为本图边身份；返回当前源、目标及共享权重，失效为 None，无副作用。
    fn edge_data(&self, id: SparseEdgeId) -> Option<(SparseNodeId, SparseNodeId, &E)> {
        sparse_inner::edge_data(self, id)
    }
    /// id 为本图边身份；返回当前源、目标及独占权重，失效为 None；仅权重可变，版本不变。
    fn edge_data_mut(&mut self, id: SparseEdgeId) -> Option<(SparseNodeId, SparseNodeId, &mut E)> {
        sparse_inner::edge_data_mut(self, id)
    }
    /// id 为本图节点身份，outgoing 为 true 选出边、false 选入边。
    /// 返回按插入顺序的借用迭代器；失效节点为空，无副作用。
    fn adjacent(&self, id: SparseNodeId, outgoing: bool) -> SparseEdges<'_> {
        sparse_inner::adjacent(self, id, outgoing)
    }
    /// weight 为移入存储的节点权重；返回新身份并递增版本。
    /// 版本耗尽返回 VersionExhausted，图不变；分配/Drop panic 不捕获。
    fn add_node(&mut self, weight: N) -> Result<SparseNodeId, GraphError> {
        // 设计已确认：检查下一版本；准备容量，写入带权重的节点和空邻接。
        // 登记本存储索引/反查后发布版本，返回新身份。
        sparse_inner::add_node(self, weight)
    }
    /// source、target 为本图端点，weight 为移入的边权重；允许自环和平行边。
    /// 返回新边身份并在双向邻接末尾登记，版本递增；先校验端点再检查版本。
    /// 端点缺失返回 MissingEndpoint，版本耗尽返回 VersionExhausted，业务错误不改变图。
    fn add_edge(
        &mut self,
        source: SparseNodeId,
        target: SparseNodeId,
        weight: E,
    ) -> Result<SparseEdgeId, GraphError> {
        // 设计已确认：先校验两个端点，再检查下一版本并准备存储。
        // 写入边，按源出边/目标入边顺序登记两条关联，再发布版本。
        sparse_inner::add_edge(self, source, target, weight)
    }
    /// id 必须为有效边；返回移出的边对象，断开双向关联并修复紧凑索引。
    /// 不递增版本、不主动销毁返回权重；失效 id 违反内部前提可能 panic。
    fn unlink_edge(&mut self, id: SparseEdgeId) -> Edge<E> {
        // 设计已确认：摘除边和双向关联；末边/身份填洞，修复对应反查或关联身份。
        // 返回移出对象，由调用方在拓扑和版本发布后销毁权重。
        sparse_inner::unlink_edge(self, id)
    }
    /// id 为本图边身份；有效时删除并返回 true，失效时幂等返回 false 且版本不变。
    /// 版本耗尽返回 VersionExhausted 且图不变；先完成拓扑和版本修复，再销毁权重。
    fn delete_edge(&mut self, id: SparseEdgeId) -> Result<bool, GraphError> {
        // 设计已确认：失效返回 false；预检版本，调用 unlink_edge 的唯一摘边流程。
        // 发布版本，再销毁移出边，返回 true。
        sparse_inner::delete_edge(self, id)
    }
    /// id 为本图节点身份；有效时连同所有关联边删除并返回 true；自环只删除一次。
    /// 失效幂等返回 false；版本耗尽返回 VersionExhausted 且图不变；先修复拓扑再销毁权重。
    fn delete_node(&mut self, id: SparseNodeId) -> Result<bool, GraphError> {
        // 设计已确认：失效返回 false；预检版本；按当前出边优先、再入边顺序逐条摘边，自环不重复。
        // 摘除节点，紧凑身份表填洞并修复反查，清槽位，发布版本后销毁边和节点。
        sparse_inner::delete_node(self, id)
    }
    /// id 为当前边身份，ids 为本批双向映射；删除并同步身份失效及搬移位置。
    /// 返回是否删除；失效幂等返回 false，版本耗尽无修改；Drop panic 不捕获。
    fn batch_delete_edge(
        &mut self,
        id: SparseEdgeId,
        ids: &mut crate::common::BatchIds<SparseNodeId, SparseEdgeId>,
    ) -> Result<bool, GraphError> {
        // 设计已确认：调用 delete_edge；成功时从 ids 移除该边的批前身份，不存在位置搬移。
        sparse_inner::batch_delete_edge(self, id, ids)
    }
    /// id 为当前节点身份，ids 为本批双向映射；删除并同步关联失效及搬移位置。
    /// 返回是否删除；失效幂等返回 false，版本耗尽无修改；Drop panic 不捕获。
    fn batch_delete_node(
        &mut self,
        id: SparseNodeId,
        ids: &mut crate::common::BatchIds<SparseNodeId, SparseEdgeId>,
    ) -> Result<bool, GraphError> {
        // 设计已确认：先按出边、入边顺序收集当前关联身份，再执行 delete_node。
        // 删除成功才从 ids 移除节点和全部收集边；自环重复移除幂等。
        sparse_inner::batch_delete_node(self, id, ids)
    }
}

// Deque 不允许直接 Clone；重建各自链接，保留节点/边 SlotMap 的完整代际身份。
impl<N: Clone, E: Clone> Clone for SparseGraph<N, E> {
    /// self 为源对象；返回独立拓扑并保持完整身份、版本与顺序，权重依其 Clone 语义。
    /// Clone/分配 panic 原样传播，不修改源图；权重的内部共享状态不保证隔离。
    fn clone(&self) -> Self {
        // 设计已确认：克隆节点和边的权重与完整代际身份；节点克隆暂留空邻接。
        // 按各节点原双向邻接顺序在独立链接池重建并回写边链接；复制索引与版本。
        sparse_inner::clone_graph(self)
    }
}
impl<N: Clone> Clone for Node<N> {
    /// self 为源节点；克隆 weight、复制 dense，但双向邻接置空，供 SparseGraph::clone 重建。
    /// 返回值不是已链接节点；权重 Clone panic 传播，不修改源节点。
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            dense: self.dense,
            outgoing: Deque::new(),
            incoming: Deque::new(),
        }
    }
}
impl<E: Clone> Clone for Edge<E> {
    /// self 为源边；克隆 weight，原样复制端点、dense 和旧链接键。
    /// 返回值的链接键由 SparseGraph::clone 重建后替换；Clone panic 传播，不修改源边。
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            source: self.source,
            target: self.target,
            dense: self.dense,
            out_link: self.out_link,
            in_link: self.in_link,
        }
    }
}
impl<N, E> NodeIndexable for SparseGraph<N, E> {
    /// self 为当前图；返回节点索引上界，无副作用，不一定等于有效节点数。
    fn node_bound(&self) -> usize {
        sparse_inner::index_node_bound(self)
    }
    /// id 必须为当前图有效节点身份；返回稳定槽位或当前紧凑位置；无效时断言 panic，无修改。
    fn to_index(&self, id: SparseNodeId) -> usize {
        sparse_inner::index_to_index(self, id)
    }
    /// index 为槽位/位置；返回该处完整当前身份，越界或空槽为 None，无副作用。
    fn node_at_index(&self, index: usize) -> Option<SparseNodeId> {
        sparse_inner::index_node_at_index(self, index)
    }
}
impl<N, E> DenseNodeIndex for SparseGraph<N, E> {
    /// self 为当前图；返回有效节点数，无副作用。
    fn node_count(&self) -> usize {
        sparse_inner::dense_node_count(self)
    }
    /// id 为当前图节点身份；返回当前紧凑位置，失效为 None，无副作用。
    fn to_dense(&self, id: SparseNodeId) -> Option<usize> {
        sparse_inner::dense_to_dense(self, id)
    }
    /// index 为当前紧凑位置；返回当前完整节点身份，越界为 None，无副作用。
    fn node_at_dense(&self, index: usize) -> Option<SparseNodeId> {
        sparse_inner::dense_node_at_dense(self, index)
    }
}
impl_graph!(
    SparseGraph,
    SparseNodeId,
    SparseEdgeId,
    std::iter::Copied<std::slice::Iter<'a, SparseNodeId>>,
    std::iter::Copied<std::slice::Iter<'a, SparseEdgeId>>,
    SparseEdges<'a>,
    SlotVisitMap<SparseNodeId>
);
