//! 拓扑排序壳：失败数据与排序契约的唯一声明。
#[path = "topo_inner.rs"]
mod topo_inner;
use graph_core::{NodeIds, OutNeighbors};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::Hash;

/// 可复用的 Kahn 临时状态，不保存图借用、图身份或跨调用有效的版本缓存。
/// 每次排序都重新初始化逻辑内容，保留容量；成功或有环失败后均可换图复用。
/// 下划线私有字段为可复用的真实工作缓冲，配对芯承接初始化和排序实现。
pub struct TopoWorkspace<N> {
    /// 本次图的节点枚举，维持后端顺序；不能复用为下次图的有效节点清单。
    _nodes: Vec<N>,
    /// 本次每个节点剩余入度，支持稀疏和自定义 ID。
    _indegrees: HashMap<N, usize>,
    /// 本次零入度节点的先进先出队列。
    _ready: VecDeque<N>,
}
impl<N> TopoWorkspace<N> {
    /// 创建空工作区，不读取图、不预分配与节点数有关的容量。
    pub fn new() -> Self {
        topo_inner::workspace_new()
    }
}
impl<N> Default for TopoWorkspace<N> {
    /// 等同 TopoWorkspace::new，不引入额外状态。
    fn default() -> Self {
        Self::new()
    }
}

/// graph：本次拓扑稳定的图，其能力前提与 toposort 相同。
/// workspace：独占的可复用工作区，可来自另一张同 ID 类型图或之前有环的运行。
/// 每次清除旧逻辑状态再读取当前图，不依赖 GraphVersion 或稠密索引。
/// 成功返回独立拥有的完整排序；有环返回独立拥有的 CycleError，工作区仍可再次使用。
/// 复用节点表、入度表和队列容量，不承诺返回结果 Vec 零分配或固定容量上限。
/// 不修改图；顺序及环错误语义与 toposort 一致。
/// # Panics
/// 后端违反枚举与邻接契约时可 panic；再次使用仍须先重新初始化工作区。
pub fn toposort_with_workspace<G>(
    graph: &G,
    workspace: &mut TopoWorkspace<G::NodeId>,
) -> Result<Vec<G::NodeId>, CycleError<G::NodeId>>
where
    G: NodeIds + OutNeighbors + ?Sized,
    G::NodeId: Eq + Hash,
{
    // 设计已确认：清空全部逻辑状态但保留容量 -> 按当前图重建节点与入度。
    // 执行 toposort 声明的唯一 Kahn 主流程；结果与错误不借用工作区。
    topo_inner::toposort_with_workspace(graph, workspace)
}

/// 拓扑排序受环阻塞；不代表一条精确的环路径。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleError<N> {
    /// 环内节点与受其阻塞的下游节点，沿 node_ids 原枚举顺序保留。
    pub remaining: Vec<N>,
}
impl<N: fmt::Debug> fmt::Display for CycleError<N> {
    /// f：错误文本写入目标；输出剩余节点数，转发格式器写入错误。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        topo_inner::format_cycle(self, f)
    }
}
impl<N: fmt::Debug> std::error::Error for CycleError<N> {}

/// 对 graph 执行 Kahn 拓扑排序；graph 及视图谓词在调用期间须保持拓扑稳定。
/// node_ids 必须唯一枚举所有可见节点，所有出邻居必须属于该枚举。
/// 成功返回含孤立节点的完整序列；支持非连续 ID、平行边；自环判为有环。
/// 多个可用节点按节点与邻接枚举顺序入队。不修改图，相同枚举顺序产生相同结果。
/// # Errors
/// 存在环时返回 CycleError，remaining 含环与其阻塞的下游节点，不返回部分排序。
/// # Panics
/// 后端违反邻居属于 node_ids 的能力契约时可能 panic。
pub fn toposort<G: NodeIds + OutNeighbors + ?Sized>(
    graph: &G,
) -> Result<Vec<G::NodeId>, CycleError<G::NodeId>> {
    // 设计已确认（整理既有 Kahn 流程）：
    // 收集节点 -> 每条出边分别累计入度（保留平行边计数）。
    // 按原枚举顺序将零入度节点入队。
    // 循环出队并输出；每条出边将目标入度减一，归零时入队。
    // 全部输出 -> Ok；否则按原枚举顺序收集正入度节点 -> Err。
    topo_inner::toposort(graph)
}
