//! 有序拓扑操作与普通、原子批量能力；不引入具体存储或默认执行实现。
use crate::{GraphBase, GraphVersion};

/// 指定后端的一次拓扑操作，展开为该后端的身份与数据类型。
pub type BatchOperation<G> = TopologyEdit<
    <G as GraphBase>::NodeId,
    <G as GraphBase>::EdgeId,
    <G as GraphBatchEdit>::Node,
    <G as GraphBatchEdit>::Edge,
>;
/// 指定后端按输入顺序返回的批量操作结果。
pub type BatchResults<G> =
    Vec<TopologyEditResult<<G as GraphBase>::NodeId, <G as GraphBase>::EdgeId>>;
/// 普通批次的完整调用结果；失败保存已完成前缀。
pub type BatchOutcome<G> = Result<
    BatchResults<G>,
    BatchEditError<
        <G as GraphBase>::NodeId,
        <G as GraphBase>::EdgeId,
        <G as GraphBatchEdit>::Error,
    >,
>;
/// 原子批次的完整调用结果；失败不包含已提交前缀。
pub type AtomicBatchOutcome<G> =
    Result<BatchResults<G>, AtomicBatchEditError<<G as GraphBatchEdit>::Error>>;

/// 批量拓扑操作；身份只可引用批次开始前已有元素，不支持批内新建元素句柄。
/// 后续删除可使前面存在的身份失效；实际执行时仍须验证端点与目标。
pub enum TopologyEdit<N, E, NW, EW> {
    /// 插入独立节点，产生的新 ID 只能从本批结果读取。
    InsertNode {
        /// 转移给后端的节点数据。
        weight: NW,
    },
    /// 在批次开始前存在、且执行此操作时仍有效的节点间插入边。
    InsertEdge {
        /// 源节点身份。
        source: N,
        /// 目标节点身份。
        target: N,
        /// 转移给后端的边数据。
        weight: EW,
    },
    /// 删除节点及全部关联边；不存在为成功无操作。
    RemoveNode {
        /// 目标节点身份。
        node: N,
    },
    /// 删除边；不存在为成功无操作。
    RemoveEdge {
        /// 目标边身份。
        edge: E,
    },
}

/// 与输入操作逐项对应的结果；无操作删除仍占一个结果位置。
pub enum TopologyEditResult<N, E> {
    /// 节点插入成功。
    NodeInserted {
        /// 后端分配的节点身份，后续批次操作可能不再保证其有效。
        id: N,
    },
    /// 边插入成功。
    EdgeInserted {
        /// 后端分配的边身份，后续批次删除端点可能已将其级联删除。
        id: E,
    },
    /// 节点删除操作成功。
    NodeRemoved {
        /// true 表示确实删除节点及关联边；false 表示目标不存在。
        removed: bool,
    },
    /// 边删除操作成功。
    EdgeRemoved {
        /// true 表示确实删除边；false 表示目标不存在。
        removed: bool,
    },
}

/// 普通批次失败；已完成前缀保留，失败项自身不得留下部分更新。
pub struct BatchEditError<N, E, X> {
    /// 与输入顺序一致的已完成前缀，允许包含无操作删除的结果。
    pub completed: Vec<TopologyEditResult<N, E>>,
    /// 首个失败操作的零基位置，等于 completed.len()；之后的操作未执行。
    pub failed_at: usize,
    /// 失败操作的后端错误，不保证返还已消费的数据。
    pub error: X,
}

/// 原子批次失败，无操作结果被提交。
pub struct AtomicBatchEditError<X> {
    /// 能归因到某项操作时返回其零基位置；批次准备或整体提交失败为 None。
    pub failed_at: Option<usize>,
    /// 后端验证、准备或提交错误。
    pub error: X,
}

/// 有序批量拓扑编辑；普通失败保留完成前缀，不承诺批次中间状态隔离。
/// 与 GraphInsert / GraphRemove 独立，后端只实现其真实支持的能力。
pub trait GraphBatchEdit: GraphVersion {
    /// 插入节点的数据类型。
    type Node;
    /// 插入边的数据类型。
    type Edge;
    /// 单项或批次准备错误。
    type Error;
    /// edits：消费的有限操作序列，按输入顺序执行；类型不要求 Clone。
    /// 空批次返回空结果且版本不变。成功返回逐项结果，失败停止并保留完成前缀。
    /// 无法开始执行时用 failed_at=0、completed=[] 返回错误。失败项数据及拓扑不变，
    /// 未执行项不会调用后端修改，但整个输入 Vec 已消费，不承诺返还剩余数据。
    /// 插入、级联删除、自环与平行边规则与后端单项编辑一致；数据和相关索引同步维护。
    /// 若前缀或成功批次实际提交任何拓扑变化，返回前版本须不同于批前，即使最终
    /// 拓扑恰好恢复原状；全为无操作删除时版本不变。不承诺版本变化次数或数值增量。
    /// 不发 NodePropertyChanged（该事件仅用于属性设置），不承诺多次操作回滚。
    /// panic 的恢复由后端声明；若允许继续使用，数据、拓扑及索引仍须自洽。
    fn edit_batch(&mut self, edits: Vec<BatchOperation<Self>>) -> BatchOutcome<Self>;
}

/// 可选原子批次：同时支持普通批次，并额外保证原子入口失败零提交。
pub trait AtomicGraphBatchEdit: GraphBatchEdit {
    /// edits：有序有限操作序列，身份及值所有权规则同 GraphBatchEdit::edit_batch。
    /// 全部成功才发布全部数据、拓扑和索引；读取者只可观察提交前或提交后的状态，
    /// 不暴露中间前缀。正常 Err 时图数据、拓扑、索引及版本与批前一致，不发成功通知。
    /// 失败可在验证、准备或最终提交发生，按 AtomicBatchEditError 说明位置。
    /// 成功结果逐项对应；有实际拓扑变更时版本不同于批前，无操作批次版本不变。
    /// 不承诺持久化事务或 panic 回滚；后端必须明确 panic 后可否继续使用。
    fn edit_batch_atomic(&mut self, edits: Vec<BatchOperation<Self>>) -> AtomicBatchOutcome<Self>;
}
