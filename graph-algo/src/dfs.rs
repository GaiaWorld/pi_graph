//! 深度优先批量遍历入口。
#[path = "dfs_inner.rs"]
mod dfs_inner;
use graph_core::OutNeighbors;

/// graph：遍历期间拓扑稳定的图；start：该图中有效的起点。
/// 返回包含起点的深度优先可达节点序列，每个节点最多一次，不修改图。
/// 顺序与状态规则见 graph_query::Dfs::next；本入口一次性消费全部结果。
pub fn dfs<G: OutNeighbors + ?Sized>(graph: &G, start: G::NodeId) -> Vec<G::NodeId> {
    // 设计已确认：创建 Dfs Walker -> 通过标准迭代适配收集全部节点。
    dfs_inner::dfs(graph, start)
}
