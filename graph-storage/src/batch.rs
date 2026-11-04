//! 两种存储共用的批次壳；身份映射由 common::BatchIds 真实声明。
use crate::{common::BatchIds, GraphError};
use graph_core::*;
#[path = "batch_inner.rs"]
mod batch_inner;

/// 批次需要的最小存储协议；各实现转发本类型的真实壳方法，不跨芯调用。
pub(crate) trait BatchStorage:
    GraphBatchEdit<Error = GraphError> + NodeAccess + EdgeAccess + GraphVersion<Version = u64>
{
    /// weight 为新节点权重；成功返回身份并递增版本，版本耗尽则不变。
    fn add_node(
        &mut self,
        weight: <Self as GraphBatchEdit>::Node,
    ) -> Result<Self::NodeId, GraphError>;
    /// source、target 为当前端点，weight 为移入边权重；成功返回身份并递增版本。
    /// MissingEndpoint 或 VersionExhausted 时图不变；允许自环和平行边。
    fn add_edge(
        &mut self,
        source: Self::NodeId,
        target: Self::NodeId,
        weight: <Self as GraphBatchEdit>::Edge,
    ) -> Result<Self::EdgeId, GraphError>;
    /// id 为当前节点，ids 为批前映射；成功返回 true、删除关联边并同步失效/搬移。
    /// 失效幂等返回 false，VersionExhausted 无修改；版本发布后销毁权重。
    fn batch_delete_node(
        &mut self,
        id: Self::NodeId,
        ids: &mut BatchIds<Self::NodeId, Self::EdgeId>,
    ) -> Result<bool, GraphError>;
    /// id 为当前边，ids 为批前映射；成功同步映射，失效返回 false，版本耗尽无修改。
    fn batch_delete_edge(
        &mut self,
        id: Self::EdgeId,
        ids: &mut BatchIds<Self::NodeId, Self::EdgeId>,
    ) -> Result<bool, GraphError>;
}

/// graph 为独占存储，edits 按批前身份解释且按输入顺序消费。
/// 返回逐项结果；首错返回失败下标和已完成前缀，前缀副作用保留。
/// 空批不修改；失效删除幂等；权重 Drop panic 与分配失败沿用原容器行为。
/// 若删除提交后权重 Drop 单次 panic 且成功展开，图拓扑及版本已自洽，可继续使用；
/// 已完成前缀保留但不返回结果。其他操作阶段 panic 不保证恢复，调用方应丢弃图。
pub(crate) fn edit_batch<G: BatchStorage>(
    graph: &mut G,
    edits: Vec<BatchOperation<G>>,
) -> BatchOutcome<G> {
    // 设计已确认：空批直接返回；仅收集本批引用的有效身份建立双向映射。
    // 依次执行：插入节点；映射端点后插边；映射删除目标并同步搬移/失效。
    // 每项成功追加执行时结果；首错返回已完成前缀及下标，不撤销此前修改。
    batch_inner::edit_batch(graph, edits)
}

/// graph 为独占存储，edits 与 edit_batch 同义；权重 Clone 不隔离内部共享状态。
/// 全成功发布克隆图；首错仅返回下标/错误，原拓扑不变；panic 不承诺回滚。
/// 空批不克隆，全无操作删除不替换原权重；发布后才销毁旧图。
/// 克隆/克隆图执行阶段 panic 成功展开后，源图拓扑与版本不变且可继续使用；
/// 发布后旧图 Drop panic 成功展开后，新图拓扑与版本已提交且可继续使用。
/// 上述保证仅针对图结构，不撤销业务 Clone/Drop 对共享状态的副作用；二次 panic/终止无法恢复。
/// 时间与额外空间含存储 Clone 成本（见 DenseGraph/SparseGraph），另加批次准备与执行成本。
pub(crate) fn edit_batch_atomic<G: BatchStorage + Clone>(
    graph: &mut G,
    edits: Vec<BatchOperation<G>>,
) -> AtomicBatchOutcome<G> {
    // 设计已确认：空批返回；克隆 graph，在克隆上执行 edit_batch 的唯一普通批流程。
    // 失败丢弃克隆并返回下标；成功仅在版本改变时替换原图，再销毁旧图。
    batch_inner::edit_batch_atomic(graph, edits)
}
