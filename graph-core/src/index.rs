use crate::GraphBase;
use fixedbitset::FixedBitSet;
use std::collections::HashSet;
use std::hash::Hash;

/// 可选索引能力；有效节点的索引唯一且小于 node_bound，允许空洞。
pub trait NodeIndexable: GraphBase {
    /// 返回索引空间的排他上界，不是节点数量；不修改图。
    fn node_bound(&self) -> usize;
    /// node：有效节点身份。返回对应索引；调用方必须保证节点有效。
    /// 拓扑稳定期间映射保持稳定，不修改图。
    fn to_index(&self, node: Self::NodeId) -> usize;
    /// index：索引位置。存在有效节点则返回其身份，空洞或越界返回 None，不修改图。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId>;
}

/// 当前可见节点到 0..node_count() 的双射，与稀疏槽位索引独立。
/// 拓扑与视图谓词稳定期间映射必须稳定；不承诺跨拓扑修改继续有效。
pub trait DenseNodeIndex: GraphBase {
    /// 返回当前可见节点数量，也是稠密索引的排他上界，不修改图。
    fn node_count(&self) -> usize;
    /// id：待查身份；当前视图有效节点返回唯一稠密位置，否则 None，不修改图。
    fn to_dense(&self, id: Self::NodeId) -> Option<usize>;
    /// index：稠密位置；小于 node_count 时返回对应节点，否则 None，不修改图。
    fn node_at_dense(&self, index: usize) -> Option<Self::NodeId>;
}

/// 对节点身份进行幂等访问标记，不依赖图的具体存储。
pub trait VisitMap<N> {
    /// node：待标记身份。首次标记返回 true，已标记返回 false；只修改访问集合。
    fn visit(&mut self, node: N) -> bool;
    /// node：待查询身份。已标记返回 true，否则返回 false，不修改状态。
    fn is_visited(&self, node: N) -> bool;
}

/// 使用节点槽位索引的访问集合，空间为 O(node_bound) 位，允许索引空洞。
/// 持有图的共享借用；集合使用期间必须保持同一图及稳定索引映射。
pub struct IndexedVisitMap<'g, G: NodeIndexable + ?Sized> {
    /// 创建集合时图的共享借用，用于将节点身份映射到稳定槽位索引。
    graph: &'g G,
    /// 位范围为创建时的 0..graph.node_bound()，初始全部清零；允许索引空洞。
    visited: FixedBitSet,
}

impl<'g, G: NodeIndexable + ?Sized> IndexedVisitMap<'g, G> {
    /// 创建全新空集合；按 node_bound 分配位图，不枚举或哈希节点身份。
    pub fn new(graph: &'g G) -> Self {
        Self {
            graph,
            visited: FixedBitSet::with_capacity(graph.node_bound()),
        }
    }
}

impl<G: NodeIndexable + ?Sized> VisitMap<G::NodeId> for IndexedVisitMap<'_, G> {
    /// node 必须是创建时图中的有效节点；首次访问返回 true。
    fn visit(&mut self, node: G::NodeId) -> bool {
        !self.visited.put(self.graph.to_index(node))
    }

    /// node 必须是创建时图中的有效节点；不修改集合。
    fn is_visited(&self, node: G::NodeId) -> bool {
        self.visited.contains(self.graph.to_index(node))
    }
}

// 通用集合适配为独立定义，不承担图遍历主流程，无需另拆实现芯。
impl<N: Eq + Hash> VisitMap<N> for HashSet<N> {
    fn visit(&mut self, node: N) -> bool {
        self.insert(node)
    }
    fn is_visited(&self, node: N) -> bool {
        self.contains(&node)
    }
}

/// 后端可选提供自己的空访问集合，可显式传给 Walker 的 with_visit_map 入口。
pub trait Visitable: GraphBase {
    /// 节点访问集合，生命周期与图借用独立。
    type Map: VisitMap<Self::NodeId>;
    /// 返回全新空集合；不修改图，各次返回的集合互不影响。
    fn visit_map(&self) -> Self::Map;
}
