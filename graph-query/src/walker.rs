//! 遍历壳：真实状态、中文契约与主流程；实现位于配对芯。
#[path = "walker_inner.rs"]
mod walker_inner;

use graph_core::{GraphBase, IndexedVisitMap, NodeIndexable, OutNeighbors, VisitMap};
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

/// 逐步推进遍历的状态协议；一次遍历必须使用同一个图及稳定拓扑。
/// 默认访问集合不借用图；自定义集合（如 IndexedVisitMap）可以持有图借用。
pub trait Walker<G: GraphBase + ?Sized> {
    /// graph：初始化时对应的图；节点 ID 与拓扑必须保持有效。
    /// 推进自身状态并返回下一个节点；耗尽返回 None；不修改图。
    fn next(&mut self, graph: &G) -> Option<G::NodeId>;

    /// graph：遍历所用的稳定拓扑。消费 self，返回持有图借用的标准迭代器。
    /// 包装不推进遍历；之后的 next 遵循当前 Walker 的顺序与耗尽语义。
    fn iter(self, graph: &G) -> WalkerIter<'_, G, Self>
    where
        Self: Sized,
    {
        walker_inner::into_iter(self, graph)
    }
}

/// 将 Walker 适配为持有图借用的标准迭代器。
pub struct WalkerIter<'g, G: ?Sized, W> {
    /// 整个标准迭代期间借用的图，拓扑必须保持稳定。
    graph: &'g G,
    /// 当前遍历状态的唯一所有者。
    walker: W,
}
impl<G: GraphBase + ?Sized, W: Walker<G>> Iterator for WalkerIter<'_, G, W> {
    type Item = G::NodeId;
    /// 推进持有的 Walker；返回下一节点或耗尽标志，不修改图。
    fn next(&mut self) -> Option<Self::Item> {
        walker_inner::iter_next(self)
    }
}

/// 广度优先遍历；入队时标记，每个可达 ID 最多返回一次。
pub struct Bfs<N, M = HashSet<N>> {
    /// 已发现、尚未返回的节点；先进先出维持距离层序。
    queue: VecDeque<N>,
    /// 全部已入队节点，阻止自环、平行边与环重复入队。
    visited: M,
}
impl<N: Copy + Eq + Hash> Bfs<N> {
    /// _graph：确定 ID 类型及遍历目标，不保存借用，也不检查起点存在性。
    /// start：必须属于该图的有效节点。返回仅将起点入队并标记的新状态。
    pub fn new<G: GraphBase<NodeId = N> + ?Sized>(_graph: &G, start: N) -> Self {
        walker_inner::bfs_new(start)
    }
}
impl<N: Copy, M: VisitMap<N>> Bfs<N, M> {
    /// _graph：确定节点身份及目标图，不保存借用、不检查起点存在性。
    /// start：有效起点；map：转移给 Walker 的全新空集合，须正确区分全部节点身份。
    /// 集合的索引映射须适用于当前图，且遍历期间保持稳定。
    /// 不检查或清空已有标记，不支持预标记跳过，不修改图。
    /// 返回仅起点入队且已标记的状态；集合实现的 panic 原样传播。
    pub fn with_visit_map<G: GraphBase<NodeId = N> + ?Sized>(_graph: &G, start: N, map: M) -> Self {
        walker_inner::bfs_with_visit_map(start, map)
    }
    /// graph：初始化时的同一图，拓扑须稳定；self 可暂停恢复。
    /// 返回下一节点并发现其邻居；耗尽后持续返回 None，不修改图。
    /// 同层顺序由邻接枚举顺序决定；两次调用间可在集合借用约束内修改业务数据。
    /// 默认集合允许独占修改图；索引集合持有共享借用，须使用内部可变性等方式。
    pub fn next<G: OutNeighbors<NodeId = N> + ?Sized>(&mut self, graph: &G) -> Option<N> {
        // 设计已确认（本轮仅整理既有行为）：
        // 队列为空 -> 返回 None，不改变状态。
        // 否则取队首；按邻接顺序枚举出邻居。
        // 首次发现的邻居先标记再入队；最后返回队首节点。
        walker_inner::bfs_next(self, graph)
    }
}
impl<'g, G: NodeIndexable + ?Sized> Bfs<G::NodeId, IndexedVisitMap<'g, G>> {
    /// 使用 FixedBitSet 标记索引；start 必须有效，图和索引映射须稳定。
    /// 此构造持有图共享借用，适合索引空间紧凑的图；遍历顺序与 new 相同。
    pub fn new_indexed(graph: &'g G, start: G::NodeId) -> Self {
        Self::with_visit_map(graph, start, IndexedVisitMap::new(graph))
    }
}
impl<G: OutNeighbors + ?Sized, M: VisitMap<G::NodeId>> Walker<G> for Bfs<G::NodeId, M> {
    /// graph 的前提、返回顺序与状态变化均同 Bfs::next。
    fn next(&mut self, graph: &G) -> Option<G::NodeId> {
        Bfs::next(self, graph)
    }
}

/// 深度优先遍历；弹栈时标记，兄弟节点按邻接顺序逆序展开。
pub struct Dfs<N, M = HashSet<N>> {
    /// 已发现的候选节点栈；允许尚未访问的同一 ID 重复入栈。
    stack: Vec<N>,
    /// 已返回的节点；弹栈时去重，避免重复返回与环导致的无限展开。
    visited: M,
}
impl<N: Copy + Eq + Hash> Dfs<N> {
    /// _graph：确定 ID 类型及遍历目标，不保存借用，也不检查起点存在性。
    /// start：必须属于该图的有效节点。返回仅将起点压栈、尚未标记的新状态。
    pub fn new<G: GraphBase<NodeId = N> + ?Sized>(_graph: &G, start: N) -> Self {
        walker_inner::dfs_new(start)
    }
}
impl<N: Copy, M: VisitMap<N>> Dfs<N, M> {
    /// _graph：确定节点身份及目标图，不保存借用、不检查起点存在性。
    /// start：有效起点；map：转移给 Walker 的全新空集合，须正确区分全部节点身份。
    /// 集合的索引映射须适用于当前图，且遍历期间保持稳定。
    /// 不检查或清空已有标记，不支持预标记跳过，不修改图。
    /// 返回仅起点压栈但尚未标记的状态；集合实现的 panic 原样传播。
    pub fn with_visit_map<G: GraphBase<NodeId = N> + ?Sized>(_graph: &G, start: N, map: M) -> Self {
        walker_inner::dfs_with_visit_map(start, map)
    }
    /// graph：初始化时的同一图，拓扑须稳定；self 可暂停恢复。
    /// 返回下一未访问节点；耗尽后持续返回 None，不修改图。
    /// 兄弟节点按邻接顺序逆序展开；两次调用间可在集合借用约束内修改业务数据。
    /// 默认集合允许独占修改图；索引集合持有共享借用，须使用内部可变性等方式。
    pub fn next<G: OutNeighbors<NodeId = N> + ?Sized>(&mut self, graph: &G) -> Option<N> {
        // 设计已确认（本轮仅整理既有行为）：
        // 循环弹栈，跳过已访问节点；首次访问时标记。
        // 按邻接顺序压入尚未访问的出邻居，再返回当前节点。
        // 栈耗尽 -> 返回 None；只在弹栈时去重以保留深度优先顺序。
        walker_inner::dfs_next(self, graph)
    }
}
impl<'g, G: NodeIndexable + ?Sized> Dfs<G::NodeId, IndexedVisitMap<'g, G>> {
    /// 使用 FixedBitSet 标记索引；start 必须有效，图和索引映射须稳定。
    /// 此构造持有图共享借用，适合索引空间紧凑的图；遍历顺序与 new 相同。
    pub fn new_indexed(graph: &'g G, start: G::NodeId) -> Self {
        Self::with_visit_map(graph, start, IndexedVisitMap::new(graph))
    }
}
impl<G: OutNeighbors + ?Sized, M: VisitMap<G::NodeId>> Walker<G> for Dfs<G::NodeId, M> {
    /// graph 的前提、返回顺序与状态变化均同 Dfs::next。
    fn next(&mut self, graph: &G) -> Option<G::NodeId> {
        Dfs::next(self, graph)
    }
}
