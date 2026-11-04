//! Vec 紧凑存储；ID 包含位置与代际，删除使用 swap_remove 填洞。
//! 未移动元素的 ID 保持有效；末尾搬移元素的旧 ID 失效，须通过索引重新获取。
use crate::{GraphError, VecVisitMap};
use graph_core::{DenseNodeIndex, NodeIndexable, VisitMap};

#[path = "dense_inner.rs"]
mod dense_inner;

/// 稠密节点位置与分配代际；字段私有，防止旧位置误命中新元素。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DenseNodeId {
    /// 当前位置索引；必须结合 generation 校验，搬移后旧位置身份失效。
    index: usize,
    /// 分配时的拓扑代际；搬移保留代际，防止位置复用误命中。
    generation: u64,
}
/// 稠密边位置与分配代际；搬移后旧 ID 失效。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DenseEdgeId {
    /// 当前位置索引；必须结合 generation 校验，搬移后旧位置身份失效。
    index: usize,
    /// 分配时的拓扑代际；搬移保留代际，防止位置复用误命中。
    generation: u64,
}
/// 节点的真实生产状态；权重与双向邻接共同维护拓扑不变量。
#[derive(Clone)]
struct Node<N> {
    /// 本存储拥有的业务权重；可变访问不递增拓扑版本。
    weight: N,
    /// 分配时的拓扑代际；搬移保留代际，防止位置复用误命中。
    generation: u64,
    /// 按插入顺序保存的出边关联；自环同时占据出入两个关联。
    outgoing: Vec<DenseEdgeId>,
    /// 按插入顺序保存的入边关联；删除保留其余关联相对顺序。
    incoming: Vec<DenseEdgeId>,
}
/// 边的真实生产状态；端点必须匹配双向关联。
#[derive(Clone)]
struct Edge<E> {
    /// 本存储拥有的业务权重；可变访问不递增拓扑版本。
    weight: E,
    /// 分配时的拓扑代际；搬移保留代际，防止位置复用误命中。
    generation: u64,
    /// 源节点的当前完整身份；必须与源节点出边关联一致。
    source: DenseNodeId,
    /// 目标节点的当前完整身份；必须与目标节点入边关联一致。
    target: DenseNodeId,
}

/// 紧凑 Vec 有向多重图；邻接按插入顺序，删除保留其余邻接相对顺序。
/// 单边删除扫描被删边和搬移边的端点邻接表以保序。
/// 删除度数 d 的节点最坏为 O(d * 最大度数)，另需修复搬移节点的关联边端点。
#[derive(Clone)]
/// Clone 的结构成本为 O(V+E)，包含当前全部邻接项；业务权重 Clone 成本另计。
pub struct DenseGraph<N, E> {
    /// 节点主存储；每个有效身份仅对应一项，内部反查与关联保持一致。
    nodes: Vec<Node<N>>,
    /// 边主存储；端点与双向邻接保持一致。
    edges: Vec<Edge<E>>,
    /// 成功拓扑修改的单调版本；权重修改与无操作删除不改变它。
    version: u64,
}
/// 无分配的节点 ID 枚举。
pub struct DenseNodes<'a, N> {
    /// 借用节点切片的带索引迭代状态；不拥有或修改节点。
    nodes: std::iter::Enumerate<std::slice::Iter<'a, Node<N>>>,
}
impl<N> Iterator for DenseNodes<'_, N> {
    type Item = DenseNodeId;
    /// self 为剩余借用迭代状态；返回下一个身份，耗尽为 None，只推进迭代器、不修改图。
    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.next().map(|(index, n)| DenseNodeId {
            index,
            generation: n.generation,
        })
    }
}
/// 无分配的边 ID 枚举。
pub struct DenseEdges<'a, E> {
    /// 边主存储；端点与双向邻接保持一致，枚举状态随迭代推进。
    edges: std::iter::Enumerate<std::slice::Iter<'a, Edge<E>>>,
}
impl<E> Iterator for DenseEdges<'_, E> {
    type Item = DenseEdgeId;
    /// self 为剩余借用迭代状态；返回下一个身份，耗尽为 None，只推进迭代器、不修改图。
    fn next(&mut self) -> Option<Self::Item> {
        self.edges.next().map(|(index, e)| DenseEdgeId {
            index,
            generation: e.generation,
        })
    }
}
impl VisitMap<DenseNodeId> for VecVisitMap {
    /// node 为创建位图时拓扑内的有效节点/索引；置位并返回是否首次访问。
    /// 不校验代际，拓扑变化后须重建；越界遵循 FixedBitSet::put 的 panic 行为。
    fn visit(&mut self, node: DenseNodeId) -> bool {
        !self.0.put(node.index)
    }
    /// node 为同一稳定拓扑内的节点/索引；返回是否已访问，不修改位图，不校验代际。
    fn is_visited(&self, node: DenseNodeId) -> bool {
        self.0.contains(node.index)
    }
}
impl<N, E> DenseGraph<N, E> {
    /// 创建空存储；无参数，返回独立空状态，不修改既有对象。
    pub fn new() -> Self {
        dense_inner::new()
    }
    /// nodes、edges 分别为节点和边的预留容量，不创建元素；返回版本为零的空图。
    /// 容量分配沿用 Rust 容器行为，无业务错误。
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        dense_inner::with_capacity(nodes, edges)
    }
    /// self 为当前图；返回有效节点数，无副作用。
    pub fn node_count(&self) -> usize {
        dense_inner::node_count(self)
    }
    /// self 为当前图；返回有效边数，无副作用。
    pub fn edge_count(&self) -> usize {
        dense_inner::edge_count(self)
    }
    /// self 为当前图；返回下一拓扑版本，耗尽返回 VersionExhausted；不修改状态。
    fn next_version(&self) -> Result<u64, GraphError> {
        dense_inner::next_version(self)
    }
    /// id 为待校验节点身份；返回位置及代际是否同时有效，无副作用。
    fn valid_node(&self, id: DenseNodeId) -> bool {
        dense_inner::valid_node(self, id)
    }
    /// id 为待校验边身份；返回位置及代际是否同时有效，无副作用。
    fn valid_edge(&self, id: DenseEdgeId) -> bool {
        dense_inner::valid_edge(self, id)
    }
    /// self 为当前图；返回借用的节点身份枚举，按当前紧凑存储顺序，无副作用。
    fn node_keys(&self) -> DenseNodes<'_, N> {
        dense_inner::node_keys(self)
    }
    /// self 为当前图；返回借用的边身份枚举，按当前紧凑存储顺序，无副作用。
    fn edge_keys(&self) -> DenseEdges<'_, E> {
        dense_inner::edge_keys(self)
    }
    /// id 为本图节点身份；返回共享权重借用，失效为 None，不修改状态。
    fn node_weight(&self, id: DenseNodeId) -> Option<&N> {
        dense_inner::node_weight(self, id)
    }
    /// id 为本图节点身份；返回独占权重借用，失效为 None；修改权重不改变拓扑版本。
    fn node_weight_mut(&mut self, id: DenseNodeId) -> Option<&mut N> {
        dense_inner::node_weight_mut(self, id)
    }
    /// id 为本图边身份；返回当前源、目标及共享权重，失效为 None，无副作用。
    fn edge_data(&self, id: DenseEdgeId) -> Option<(DenseNodeId, DenseNodeId, &E)> {
        dense_inner::edge_data(self, id)
    }
    /// id 为本图边身份；返回当前源、目标及独占权重，失效为 None；仅权重可变，版本不变。
    fn edge_data_mut(&mut self, id: DenseEdgeId) -> Option<(DenseNodeId, DenseNodeId, &mut E)> {
        dense_inner::edge_data_mut(self, id)
    }
    /// id 为本图节点身份，outgoing 为 true 选出边、false 选入边。
    /// 返回按插入顺序的借用迭代器；失效节点为空，无副作用。
    fn adjacent(
        &self,
        id: DenseNodeId,
        outgoing: bool,
    ) -> std::iter::Copied<std::slice::Iter<'_, DenseEdgeId>> {
        dense_inner::adjacent(self, id, outgoing)
    }
    /// weight 为移入存储的节点权重；返回新身份并递增版本。
    /// 版本耗尽返回 VersionExhausted，图不变；分配/Drop panic 不捕获。
    fn add_node(&mut self, weight: N) -> Result<DenseNodeId, GraphError> {
        // 设计已确认：检查下一版本，用当前 Vec 长度和新代际构造身份。
        // 追加带权重、代际及空邻接的节点；发布版本并返回身份。
        dense_inner::add_node(self, weight)
    }
    /// source、target 为本图端点，weight 为移入的边权重；允许自环和平行边。
    /// 返回新边身份并在双向邻接末尾登记，版本递增；先校验端点再检查版本。
    /// 端点缺失返回 MissingEndpoint，版本耗尽返回 VersionExhausted，业务错误不改变图。
    fn add_edge(
        &mut self,
        source: DenseNodeId,
        target: DenseNodeId,
        weight: E,
    ) -> Result<DenseEdgeId, GraphError> {
        // 设计已确认：先校验两个端点，再检查下一版本并准备存储。
        // 写入边，按源出边/目标入边顺序登记两条关联，再发布版本。
        dense_inner::add_edge(self, source, target, weight)
    }
    /// id 为本图边身份；有效时删除并返回 true，失效时幂等返回 false 且版本不变。
    /// 版本耗尽返回 VersionExhausted 且图不变；先完成拓扑和版本修复，再销毁权重。
    fn delete_edge(&mut self, id: DenseEdgeId) -> Result<bool, GraphError> {
        // 设计已确认：失效返回 false；预检版本，调用 unlink_edge 的唯一摘边流程。
        // 发布版本，再销毁移出边，返回 true。
        dense_inner::delete_edge(self, id)
    }
    /// id 必须为有效边；返回移出的边对象，断开双向关联并修复紧凑索引。
    /// 不递增版本、不主动销毁返回权重；失效 id 违反内部前提可能 panic。
    fn unlink_edge(&mut self, id: DenseEdgeId) -> Edge<E> {
        // 设计已确认：摘除边和双向关联；末边/身份填洞，修复对应反查或关联身份。
        // 返回移出对象，由调用方在拓扑和版本发布后销毁权重。
        dense_inner::unlink_edge(self, id)
    }
    /// id 为本图节点身份；有效时连同所有关联边删除并返回 true；自环只删除一次。
    /// 失效幂等返回 false；版本耗尽返回 VersionExhausted 且图不变；先修复拓扑再销毁权重。
    fn delete_node(&mut self, id: DenseNodeId) -> Result<bool, GraphError> {
        // 设计已确认：委托 delete_node_tracked 的唯一删节点流程，ids 传 None，不建立批次映射。
        dense_inner::delete_node(self, id)
    }
    /// id 为本图节点身份，ids 为可选批前身份映射；删除语义同 delete_node。
    /// 删除关联边及搬移节点/边时同步 ids；版本耗尽无修改，失效返回 false。
    fn delete_node_tracked(
        &mut self,
        id: DenseNodeId,
        ids: Option<&mut crate::common::BatchIds<DenseNodeId, DenseEdgeId>>,
    ) -> Result<bool, GraphError> {
        // 设计已确认：失效返回 false；预检版本；从出边尾部优先、再入边尾部逐条摘边。
        // 每次同步 ids 中边的删除和末项搬移；节点填洞后修复所有关联边端点及节点映射。
        // 移除节点原身份映射；发布版本，再按原顺序销毁边集合和节点。
        dense_inner::delete_node_tracked(self, id, ids)
    }
    /// id 为当前边身份，ids 为本批双向映射；删除并同步身份失效及搬移位置。
    /// 返回是否删除；失效幂等返回 false，版本耗尽无修改；Drop panic 不捕获。
    fn batch_delete_edge(
        &mut self,
        id: DenseEdgeId,
        ids: &mut crate::common::BatchIds<DenseNodeId, DenseEdgeId>,
    ) -> Result<bool, GraphError> {
        // 设计已确认：失效先返回 false；记录删除前末边身份，再执行 delete_edge。
        // 从 ids 移除被删身份；如末边填洞，登记其当前新位置。
        dense_inner::batch_delete_edge(self, id, ids)
    }
    /// id 为当前节点身份，ids 为本批双向映射；删除并同步关联失效及搬移位置。
    /// 返回是否删除；失效幂等返回 false，版本耗尽无修改；Drop panic 不捕获。
    fn batch_delete_node(
        &mut self,
        id: DenseNodeId,
        ids: &mut crate::common::BatchIds<DenseNodeId, DenseEdgeId>,
    ) -> Result<bool, GraphError> {
        dense_inner::batch_delete_node(self, id, ids)
    }
}
impl<N, E> NodeIndexable for DenseGraph<N, E> {
    /// self 为当前图；返回节点索引上界，无副作用，不一定等于有效节点数。
    fn node_bound(&self) -> usize {
        dense_inner::index_node_bound(self)
    }
    /// id 必须为当前图有效节点身份；返回稳定槽位或当前紧凑位置；无效时断言 panic，无修改。
    fn to_index(&self, id: DenseNodeId) -> usize {
        dense_inner::index_to_index(self, id)
    }
    /// index 为槽位/位置；返回该处完整当前身份，越界或空槽为 None，无副作用。
    fn node_at_index(&self, index: usize) -> Option<DenseNodeId> {
        dense_inner::index_node_at_index(self, index)
    }
}
impl<N, E> DenseNodeIndex for DenseGraph<N, E> {
    /// self 为当前图；返回有效节点数，无副作用。
    fn node_count(&self) -> usize {
        dense_inner::dense_node_count(self)
    }
    /// id 为当前图节点身份；返回当前紧凑位置，失效为 None，无副作用。
    fn to_dense(&self, id: DenseNodeId) -> Option<usize> {
        dense_inner::dense_to_dense(self, id)
    }
    /// index 为当前紧凑位置；返回当前完整节点身份，越界为 None，无副作用。
    fn node_at_dense(&self, index: usize) -> Option<DenseNodeId> {
        dense_inner::dense_node_at_dense(self, index)
    }
}
impl_graph!(
    DenseGraph,
    DenseNodeId,
    DenseEdgeId,
    DenseNodes<'a, N>,
    DenseEdges<'a, E>,
    std::iter::Copied<std::slice::Iter<'a, DenseEdgeId>>,
    VecVisitMap
);
