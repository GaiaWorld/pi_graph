//! 固定一次选择结果的结构缓存；不复制权重，不自动获得数据快照。
#[path = "indexed_subgraph_inner.rs"]
mod indexed_subgraph_inner;
use crate::selection::Selection;
use crate::*;
use std::collections::{HashMap, HashSet};

/// 缓存选中身份、端点与双向邻接，保持底层 ID；占用 O(V'+E') 结构空间。
/// 构建及全部使用期间底层拓扑、方向和身份须稳定；可用 Snapshot::View 作为 G。
/// 构建后不再调用选择策略；选择变化需显式重建，不自动检查版本或增量维护。
pub struct IndexedSubgraph<G: GraphBase> {
    /// 底层图、共享引用或一致读取对象，权重访问仍从这里取得。
    _graph: G,
    /// 可见节点，按构建时底层节点枚举顺序排列，也是缓存自己的稠密反查。
    _nodes: Vec<G::NodeId>,
    /// 原节点身份到缓存紧凑位置，不等于底层槽位索引。
    _positions: HashMap<G::NodeId, usize>,
    /// 可见边，按底层边枚举顺序排列；平行边各占一项。
    _edges: Vec<G::EdgeId>,
    /// 缓存边的当前方向端点，自环保留。
    _endpoints: HashMap<G::EdgeId, (G::NodeId, G::NodeId)>,
    /// 与 nodes 对齐的出边身份列表，内部顺序沿底层 edge_ids。
    _out_edges: Vec<Vec<G::EdgeId>>,
    /// 与 nodes 对齐的入边身份列表，内部顺序沿底层 edge_ids。
    _in_edges: Vec<Vec<G::EdgeId>>,
    /// 与 nodes 对齐的出邻居列表，平行边允许重复。
    _out_neighbors: Vec<Vec<G::NodeId>>,
    /// 与 nodes 对齐的入邻居列表，平行边允许重复。
    _in_neighbors: Vec<Vec<G::NodeId>>,
}
/// 可正常报告的构建错误；底层图不被修改。
#[derive(Debug)]
pub enum IndexedSubgraphError<N, E> {
    /// 底层重复枚举同一节点，违反身份枚举契约。
    DuplicateNode {
        /// 重复节点身份。
        node: N,
    },
    /// 底层重复枚举同一边。
    DuplicateEdge {
        /// 重复边身份。
        edge: E,
    },
    /// 枚举到的边无法读取端点。
    MissingEndpoints {
        /// 缺失端点的边身份。
        edge: E,
    },
    /// 枚举边的端点不属于底层节点枚举。
    UnknownEndpoint {
        /// 违规边身份。
        edge: E,
        /// 未枚举的端点身份。
        node: N,
    },
}
/// 指定底层图的构建结果，Err 不包含半成品缓存，不承诺返还已消费的 G。
pub type IndexedBuildResult<G> = Result<
    IndexedSubgraph<G>,
    IndexedSubgraphError<<G as GraphBase>::NodeId, <G as GraphBase>::EdgeId>,
>;
impl<G: NodeIds + EdgeIds + EdgeEndpoints> IndexedSubgraph<G> {
    /// graph：构建和使用期间稳定的底层；nodes、edges：本次固定的选择策略。
    /// 枚举节点并判定一次，边按输入顺序检查完整性后判定选择及双端点可见性。
    /// 成功缓存结构，不保留策略；没有匹配时返回有效空缓存，保留选中的孤立节点。
    /// Err 释放半成品，图拓扑数据不变；不保证返还 graph，借用 G 时不消耗原图。
    /// 不捕获谓词 panic 或分配失败；RAII 释放已取得资源，不自动生成 Snapshot。
    pub fn build<N: Selection<G::NodeId>, E: Selection<G::EdgeId>>(
        graph: G,
        nodes: N,
        edges: E,
    ) -> IndexedBuildResult<G> {
        // 设计已确认：扫描节点/重复检测 -> 固定可见节点和紧凑位置。
        // 扫描边/完整性检测 -> 选边且双端点可见 -> 记录端点、双向边及邻居。
        // 完整构建后返回；错误不发布半成品。每个节点和边的策略只判定一次。
        Self::inner_build(graph, nodes, edges)
    }
}
impl<G: GraphBase> GraphBase for IndexedSubgraph<G> {
    type NodeId = G::NodeId;
    type EdgeId = G::EdgeId;
}

impl<G: GraphBase> NodeIds for IndexedSubgraph<G> {
    type NodeIds<'a>
        = std::iter::Copied<std::slice::Iter<'a, G::NodeId>>
    where
        Self: 'a;
    /// 只访问缓存，不重新判定选择；顺序见结构字段，不读底层拓扑。
    fn node_ids(&self) -> Self::NodeIds<'_> {
        self.inner_node_ids()
    }
}

impl<G: GraphBase> EdgeIds for IndexedSubgraph<G> {
    type EdgeIds<'a>
        = std::iter::Copied<std::slice::Iter<'a, G::EdgeId>>
    where
        Self: 'a;
    /// 只访问缓存，不重新判定选择；顺序见结构字段，不读底层拓扑。
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        self.inner_edge_ids()
    }
}

impl<G: GraphBase> OutNeighbors for IndexedSubgraph<G> {
    type OutNeighbors<'a>
        = std::iter::Copied<std::slice::Iter<'a, G::NodeId>>
    where
        Self: 'a;
    /// node：原始节点身份，缓存中不存在则为空；只访问缓存，不重新判定选择；顺序见结构字段，不读底层拓扑。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        self.inner_out_neighbors(node)
    }
}

impl<G: GraphBase> InNeighbors for IndexedSubgraph<G> {
    type InNeighbors<'a>
        = std::iter::Copied<std::slice::Iter<'a, G::NodeId>>
    where
        Self: 'a;
    /// node：原始节点身份，缓存中不存在则为空；只访问缓存，不重新判定选择；顺序见结构字段，不读底层拓扑。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        self.inner_in_neighbors(node)
    }
}

impl<G: GraphBase> OutEdges for IndexedSubgraph<G> {
    type OutEdges<'a>
        = std::iter::Copied<std::slice::Iter<'a, G::EdgeId>>
    where
        Self: 'a;
    /// node：原始节点身份，缓存中不存在则为空；只访问缓存，不重新判定选择；顺序见结构字段，不读底层拓扑。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        self.inner_out_edges(node)
    }
}

impl<G: GraphBase> InEdges for IndexedSubgraph<G> {
    type InEdges<'a>
        = std::iter::Copied<std::slice::Iter<'a, G::EdgeId>>
    where
        Self: 'a;
    /// node：原始节点身份，缓存中不存在则为空；只访问缓存，不重新判定选择；顺序见结构字段，不读底层拓扑。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        self.inner_in_edges(node)
    }
}

impl<G: NodeAccess> NodeAccess for IndexedSubgraph<G> {
    type Node = G::Node;
    type NodeRef<'a>
        = G::NodeRef<'a>
    where
        Self: 'a;
    /// id：原始身份；不在缓存 None，否则底层数据访问。权重不固定，Guard 语义沿用后端。
    /// 底层已有缓存元素消失或边端点改变是稳定拓扑前提违例，不代表正常缓存失效检测。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        self.inner_node(id)
    }
}

impl<G: EdgeAccess> EdgeAccess for IndexedSubgraph<G> {
    type Edge = G::Edge;
    type EdgeRef<'a>
        = G::EdgeRef<'a>
    where
        Self: 'a;
    /// id：原始身份；不在缓存 None，否则底层数据访问。权重不固定，Guard 语义沿用后端。
    /// 底层已有缓存元素消失或边端点改变是稳定拓扑前提违例，不代表正常缓存失效检测。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>> {
        self.inner_edge(id)
    }
}

impl<G: GraphBase> EdgeEndpoints for IndexedSubgraph<G> {
    /// edge：原边身份，不是缓存位置；返回缓存中该边的原方向双端点。
    /// 边不在缓存中返回 None；仅查缓存，不读取底层拓扑或修改图。
    fn endpoints(&self, edge: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)> {
        self.inner_endpoints(edge)
    }
}

impl<G: GraphBase> NodeIndexable for IndexedSubgraph<G> {
    /// 返回缓存节点数量，也是缓存自有紧凑位置 0.._nodes.len() 的排他上界。
    /// 与 DenseNodeIndex 使用同一位置空间，不保证等于底层槽位；不修改图。
    fn node_bound(&self) -> usize {
        self.inner_node_bound()
    }
    /// node：缓存中的有效原节点 ID，不是位置；返回缓存自有紧凑位置。
    /// 不在缓存中允许 panic；仅查缓存，不修改图，不保证等于底层槽位。
    fn to_index(&self, node: Self::NodeId) -> usize {
        self.inner_to_index(node)
    }
    /// index：缓存自有紧凑位置，不是原节点 ID 或底层槽位；返回对应原节点 ID。
    /// 越界返回 None；仅查缓存，不修改图。
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_index(index)
    }
}

impl<G: GraphBase> DenseNodeIndex for IndexedSubgraph<G> {
    /// 返回缓存可见节点数量，与 node_bound 相同；不读取底层拓扑或修改图。
    fn node_count(&self) -> usize {
        self.inner_node_count()
    }
    /// id：原节点 ID，不是缓存位置；在缓存中返回自有紧凑位置，否则 None。
    /// 位置与 NodeIndexable 相同，不保证等于底层槽位；仅查缓存，不修改图。
    fn to_dense(&self, id: Self::NodeId) -> Option<usize> {
        self.inner_to_dense(id)
    }
    /// index：缓存自有稠密位置，与 node_at_index 的位置空间相同。
    /// 返回对应原节点 ID，越界返回 None；仅查缓存，不修改图。
    fn node_at_dense(&self, index: usize) -> Option<Self::NodeId> {
        self.inner_node_at_dense(index)
    }
}

impl<G: GraphBase> Visitable for IndexedSubgraph<G> {
    type Map = HashSet<G::NodeId>;
    /// 返回全新空 HashSet，各次集合互不影响，不依赖底层访问集合。
    /// 不读取底层拓扑或修改图。
    fn visit_map(&self) -> Self::Map {
        self.inner_visit_map()
    }
}
