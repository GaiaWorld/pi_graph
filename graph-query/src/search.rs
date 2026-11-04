//! 查询壳：通过公开能力和 Walker 组合，不绑定存储。
#[path = "search_inner.rs"]
mod search_inner;
use graph_core::{NodeAccess, NodeIds, NodeIndexable, OutNeighbors};

/// 使用 FixedBitSet 的可达性查询；start、target 必须为有效节点。
/// 图拓扑及索引映射须稳定；空间为 O(node_bound) 位，找到目标即停止。
pub fn reachable_indexed<G: OutNeighbors + NodeIndexable + ?Sized>(
    graph: &G,
    start: G::NodeId,
    target: G::NodeId,
) -> bool {
    search_inner::reachable_indexed(graph, start, target)
}

/// graph：查询期间拓扑稳定的图；start、target：该图中的有效节点。
/// 返回 start 到 target 是否存在有向路径，零长度路径也返回 true。
/// 不修改图；不可达返回 false，搜索在首次找到目标时停止。
pub fn reachable<G: OutNeighbors + ?Sized>(graph: &G, start: G::NodeId, target: G::NodeId) -> bool {
    // 设计已确认：按 Bfs::next 的唯一遍历流程推进；
    // 遇到 target 返回 true，耗尽返回 false。
    search_inner::reachable(graph, start, target)
}

/// graph：枚举与访问能力来自同一视图，查询期间拓扑稳定。
/// predicate：按节点枚举顺序调用，可变闭包允许累计调用方状态。
/// 返回首个匹配节点 ID；访问返回 None 的节点跳过，无匹配返回 None。
/// 每次判定结束即释放节点引用或 Guard；不修改图，闭包副作用由调用方负责。
pub fn find_node<G, F>(graph: &G, predicate: F) -> Option<G::NodeId>
where
    G: NodeIds + NodeAccess + ?Sized,
    F: FnMut(&G::Node) -> bool,
{
    // 设计已确认：顺序枚举 ID -> 尝试访问 -> 判定后释放引用；
    // 首次命中返回 ID，全部跳过或未命中返回 None。
    search_inner::find_node(graph, predicate)
}
