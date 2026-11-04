//! 共用协议入口、访问集合、错误和借用迭代定义；批次主流程位于 crate::batch。
use fixedbitset::FixedBitSet;
use graph_core::*;
use pi_slotmap::Key;
use std::{fmt, marker::PhantomData};

/// 普通图的拓扑编辑错误；失败不修改图。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphError {
    /// 插入边时至少一个端点不存在。
    MissingEndpoint,
    /// 同一图的拓扑版本已经耗尽，拒绝继续编辑。
    VersionExhausted,
}
impl fmt::Display for GraphError {
    /// f 为格式化器；写出既有英文错误文本，返回写入结果/错误，不修改图。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MissingEndpoint => "edge endpoint does not exist",
            Self::VersionExhausted => "graph topology version exhausted",
        })
    }
}
impl std::error::Error for GraphError {}

/// 不借用图的槽位位图，适用于同一稳定拓扑内的有效代际键。
pub struct SlotVisitMap<K> {
    /// 稳定拓扑内按槽位索引的访问位图；不保存代际。
    bits: FixedBitSet,
    /// 绑定节点键类型的零大小标记，不拥有图。
    marker: PhantomData<K>,
}
impl<K: Key> SlotVisitMap<K> {
    /// bound 为稳定拓扑的节点索引上界；返回独立清零位图，无图副作用。
    pub(crate) fn new(bound: usize) -> Self {
        Self {
            bits: FixedBitSet::with_capacity(bound),
            marker: PhantomData,
        }
    }
}
impl<K: Key> VisitMap<K> for SlotVisitMap<K> {
    /// node 为同一稳定拓扑内有效节点/索引；置位并返回是否首次访问，不校验代际；越界遵循 FixedBitSet::put 的 panic。
    fn visit(&mut self, node: K) -> bool {
        !self.bits.put(node.data().index() as usize)
    }
    /// node 为同一稳定拓扑内节点/索引；返回访问位，越界为 false，不修改状态。
    fn is_visited(&self, node: K) -> bool {
        self.bits.contains(node.data().index() as usize)
    }
}

/// 不借用图的紧凑位置位图；拓扑改变后须重新创建。
pub struct VecVisitMap(
    /// 当前紧凑节点索引的访问位；拓扑改变后必须重建。
    pub(crate) FixedBitSet,
);
impl VecVisitMap {
    /// bound 为稳定拓扑的节点索引上界；返回独立清零位图，无图副作用。
    pub(crate) fn new(bound: usize) -> Self {
        Self(FixedBitSet::with_capacity(bound))
    }
}
impl VisitMap<usize> for VecVisitMap {
    /// node 为同一稳定拓扑内有效节点/索引；置位并返回是否首次访问，不校验代际；越界遵循 FixedBitSet::put 的 panic。
    fn visit(&mut self, node: usize) -> bool {
        !self.0.put(node)
    }
    /// node 为同一稳定拓扑内节点/索引；返回访问位，越界为 false，不修改状态。
    fn is_visited(&self, node: usize) -> bool {
        self.0.contains(node)
    }
}

/// 沿有向关联边迭代邻居；平行边产生重复身份。
pub struct Neighbors<'a, G: OutEdges + InEdges + EdgeEndpoints> {
    /// 共享借用的图；拓扑固定但权重可有内部可变性。
    graph: &'a G,
    /// 剩余关联边迭代，保持方向、顺序和重复项。
    edges: DirectionEdges<'a, G>,
}
/// 邻居投影方向；载荷保存对应的剩余边迭代。
enum DirectionEdges<'a, G: OutEdges + InEdges + 'a> {
    /// 出边分支，投影目标节点。
    Out(
        /// 借用图的剩余出边。
        G::OutEdges<'a>,
    ),
    /// 入边分支，投影源节点。
    In(
        /// 借用图的剩余入边。
        G::InEdges<'a>,
    ),
}
impl<'a, G: OutEdges + InEdges + EdgeEndpoints> Neighbors<'a, G> {
    /// graph 为借用图，node 为起点；返回按出边顺序投影目标的迭代器，失效节点为空，无图副作用。
    pub(crate) fn outgoing(graph: &'a G, node: G::NodeId) -> Self {
        Self {
            graph,
            edges: DirectionEdges::Out(graph.out_edges(node)),
        }
    }
    /// graph 为借用图，node 为终点；返回按入边顺序投影源的迭代器，失效节点为空，无图副作用。
    pub(crate) fn incoming(graph: &'a G, node: G::NodeId) -> Self {
        Self {
            graph,
            edges: DirectionEdges::In(graph.in_edges(node)),
        }
    }
}
impl<G: OutEdges + InEdges + EdgeEndpoints> Iterator for Neighbors<'_, G> {
    type Item = G::NodeId;
    /// self 为剩余迭代状态；返回下一匹配身份或 None，保持底层顺序和重复项，仅推进迭代器；谓词/不变量 panic 原样传播。
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.edges {
            DirectionEdges::Out(edges) => edges
                .next()
                .map(|id| self.graph.endpoints(id).expect("live adjacency").1),
            DirectionEdges::In(edges) => edges
                .next()
                .map(|id| self.graph.endpoints(id).expect("live adjacency").0),
        }
    }
}

/// 批前身份映射的真实生产状态；四张表仅保存本批引用的活身份。
/// 删除同步移除两向记录，搬移保持互逆，不扫描全图。
pub(crate) struct BatchIds<N, E> {
    /// 批前节点身份到当前位置的映射，仅含本批引用的活节点，与 node_origins 互逆。
    pub(crate) nodes: std::collections::HashMap<N, N>,
    /// 批前边身份到当前位置的映射，仅含本批引用的活边，与 edge_origins 互逆。
    pub(crate) edges: std::collections::HashMap<E, E>,
    /// 当前节点身份到批前身份的反查；与 nodes 互逆。
    node_origins: std::collections::HashMap<N, N>,
    /// 当前边身份到批前身份的反查；与 edges 互逆。
    edge_origins: std::collections::HashMap<E, E>,
}
impl<N: Copy + Eq + std::hash::Hash, E: Copy + Eq + std::hash::Hash> BatchIds<N, E> {
    /// from 为旧当前位置，to 为搬移后节点身份；同步两向映射，未跟踪 from 无操作，前提 to 不对应其他活节点，无返回值。
    pub(crate) fn relocate_node(&mut self, from: N, to: N) {
        if let Some(origin) = self.node_origins.remove(&from) {
            self.nodes.insert(origin, to);
            self.node_origins.insert(to, origin);
        }
    }
    /// from 为旧当前位置，to 为搬移后边身份；同步两向映射，未跟踪 from 无操作，前提 to 不对应其他活边，无返回值。
    pub(crate) fn relocate_edge(&mut self, from: E, to: E) {
        if let Some(origin) = self.edge_origins.remove(&from) {
            self.edges.insert(origin, to);
            self.edge_origins.insert(to, origin);
        }
    }
    /// graph 为批前图，edits 为输入借用；返回仅含被引用且当前有效身份的互逆映射，重复引用合并，不扫描全图、不修改图。
    pub(crate) fn new<G>(graph: &G, edits: &[BatchOperation<G>]) -> Self
    where
        G: GraphBatchEdit<NodeId = N, EdgeId = E> + NodeAccess + EdgeAccess,
    {
        // 仅保存本批引用的已有身份，准备成本与批次大小相关，不扫描整个图。
        let mut result = Self {
            nodes: Default::default(),
            edges: Default::default(),
            node_origins: Default::default(),
            edge_origins: Default::default(),
        };
        for edit in edits {
            match edit {
                TopologyEdit::InsertEdge { source, target, .. } => {
                    for id in [*source, *target] {
                        if graph.node(id).is_some() {
                            result.nodes.insert(id, id);
                            result.node_origins.insert(id, id);
                        }
                    }
                }
                TopologyEdit::RemoveNode { node } => {
                    if graph.node(*node).is_some() {
                        result.nodes.insert(*node, *node);
                        result.node_origins.insert(*node, *node);
                    }
                }
                TopologyEdit::RemoveEdge { edge } => {
                    if graph.edge(*edge).is_some() {
                        result.edges.insert(*edge, *edge);
                        result.edge_origins.insert(*edge, *edge);
                    }
                }
                TopologyEdit::InsertNode { .. } => {}
            }
        }
        result
    }
    /// id 为已删除当前身份；只移除两向记录，未跟踪身份移除幂等；搬移须另传完整新身份给 relocate_node。
    pub(crate) fn remove_node(&mut self, id: N) {
        if let Some(origin) = self.node_origins.remove(&id) {
            self.nodes.remove(&origin);
        }
    }
    /// id 为已删除当前身份；只移除两向记录，未跟踪身份移除幂等；搬移须另传完整新身份给 relocate_edge。
    pub(crate) fn remove_edge(&mut self, id: E) {
        if let Some(origin) = self.edge_origins.remove(&id) {
            self.edges.remove(&origin);
        }
    }
}

/// 以权重谓词扫描存储的节点查询迭代器；按节点枚举顺序，惰性读取当前权重。
pub struct NodeMatches<'a, G: NodeIds + NodeAccess, Q: ?Sized> {
    /// 共享借用的图；拓扑固定但权重可有内部可变性。
    pub(crate) graph: &'a G,
    /// 共享借用的权重谓词；惰性求值，panic 原样传播。
    pub(crate) query: &'a Q,
    /// 按底层顺序推进的剩余身份迭代器。
    pub(crate) ids: G::NodeIds<'a>,
}
impl<G: NodeIds + NodeAccess, Q: Fn(&G::Node) -> bool + ?Sized> Iterator for NodeMatches<'_, G, Q> {
    type Item = G::NodeId;
    /// self 为剩余迭代状态；返回下一匹配身份或 None，保持底层顺序和重复项，仅推进迭代器；谓词/不变量 panic 原样传播。
    fn next(&mut self) -> Option<Self::Item> {
        self.ids.find(|id| {
            self.graph
                .node(*id)
                .is_some_and(|node| (self.query)(node.weight()))
        })
    }
}
/// 以权重谓词扫描存储的边查询迭代器；不维护需要额外失效处理的属性缓存。
pub struct EdgeMatches<'a, G: EdgeIds + EdgeAccess, Q: ?Sized> {
    /// 共享借用的图；拓扑固定但权重可有内部可变性。
    pub(crate) graph: &'a G,
    /// 共享借用的权重谓词；惰性求值，panic 原样传播。
    pub(crate) query: &'a Q,
    /// 按底层顺序推进的剩余身份迭代器。
    pub(crate) ids: G::EdgeIds<'a>,
}
impl<G: EdgeIds + EdgeAccess, Q: Fn(&G::Edge) -> bool + ?Sized> Iterator for EdgeMatches<'_, G, Q> {
    type Item = G::EdgeId;
    /// self 为剩余迭代状态；返回下一匹配身份或 None，保持底层顺序和重复项，仅推进迭代器；谓词/不变量 panic 原样传播。
    fn next(&mut self) -> Option<Self::Item> {
        self.ids.find(|id| {
            self.graph
                .edge(*id)
                .is_some_and(|edge| (self.query)(edge.weight()))
        })
    }
}

// 两种存储共用协议适配；具体存储及拓扑维护仍在各自模块中。
macro_rules! impl_graph {
    ($graph:ident, $node:ty, $edge:ty, $nodes:ty, $edges:ty, $adj:ty, $map:ty) => {
        impl<N, E> $crate::batch::BatchStorage for $graph<N, E> {
            /// weight 为新节点权重；转发本类型 add_node 壳，成功返回身份并递增版本，VersionExhausted 无修改。
            fn add_node(&mut self, weight: N) -> Result<$node, $crate::GraphError> {
                self.add_node(weight)
            }
            /// source、target 为当前有效端点，weight 为权重；转发本类型 add_edge 壳，返回新边或 MissingEndpoint/VersionExhausted，业务错误无修改。
            fn add_edge(
                &mut self,
                source: $node,
                target: $node,
                weight: E,
            ) -> Result<$edge, $crate::GraphError> {
                self.add_edge(source, target, weight)
            }
            /// id 为当前身份，ids 为双向批前映射；转发本类型同名壳，删除成功同步失效和搬移，失效返回 false，VersionExhausted 无修改。
            fn batch_delete_node(
                &mut self,
                id: $node,
                ids: &mut $crate::common::BatchIds<$node, $edge>,
            ) -> Result<bool, $crate::GraphError> {
                self.batch_delete_node(id, ids)
            }
            /// id 为当前身份，ids 为双向批前映射；转发本类型同名壳，删除成功同步失效和搬移，失效返回 false，VersionExhausted 无修改。
            fn batch_delete_edge(
                &mut self,
                id: $edge,
                ids: &mut $crate::common::BatchIds<$node, $edge>,
            ) -> Result<bool, $crate::GraphError> {
                self.batch_delete_edge(id, ids)
            }
        }
        impl<N, E> graph_core::GraphBase for $graph<N, E> {
            type NodeId = $node;
            type EdgeId = $edge;
        }
        impl<N, E> Default for $graph<N, E> {
            /// 无参数；返回新空图，等同本类型 new，不修改已有对象。
            fn default() -> Self {
                Self::new()
            }
        }
        impl<N, E> graph_core::NodeIds for $graph<N, E> {
            type NodeIds<'a>
                = $nodes
            where
                Self: 'a;
            /// self 为当前图；返回当前紧凑存储顺序的节点身份枚举，无副作用。
            fn node_ids(&self) -> Self::NodeIds<'_> {
                self.node_keys()
            }
        }
        impl<N, E> graph_core::EdgeIds for $graph<N, E> {
            type EdgeIds<'a>
                = $edges
            where
                Self: 'a;
            /// self 为当前图；返回当前紧凑存储顺序的边身份枚举，无副作用。
            fn edge_ids(&self) -> Self::EdgeIds<'_> {
                self.edge_keys()
            }
        }
        impl<N, E> graph_core::NodeAccess for $graph<N, E> {
            type Node = N;
            type NodeRef<'a>
                = graph_core::NodeReference<$node, &'a N>
            where
                Self: 'a;
            /// id 为本图节点身份；返回带身份共享权重引用，失效为 None，无副作用。
            fn node(&self, id: $node) -> Option<Self::NodeRef<'_>> {
                self.node_weight(id)
                    .map(|weight| graph_core::NodeReference::new(id, weight))
            }
        }
        impl<N, E> graph_core::NodeMut for $graph<N, E> {
            type NodeRefMut<'a>
                = graph_core::NodeReference<$node, &'a mut N>
            where
                Self: 'a;
            /// id 为本图节点身份；返回带身份独占权重引用，失效为 None；仅权重可变，不改变版本。
            fn node_mut(&mut self, id: $node) -> Option<Self::NodeRefMut<'_>> {
                self.node_weight_mut(id)
                    .map(|weight| graph_core::NodeReference::new(id, weight))
            }
        }
        impl<N, E> graph_core::EdgeAccess for $graph<N, E> {
            type Edge = E;
            type EdgeRef<'a>
                = graph_core::EdgeReference<$node, $edge, &'a E>
            where
                Self: 'a;
            /// id 为本图边身份；返回携带当前端点的共享权重引用，失效为 None，无副作用。
            fn edge(&self, id: $edge) -> Option<Self::EdgeRef<'_>> {
                self.edge_data(id).map(|(source, target, weight)| {
                    graph_core::EdgeReference::new(id, source, target, weight)
                })
            }
        }
        impl<N, E> graph_core::EdgeMut for $graph<N, E> {
            type EdgeRefMut<'a>
                = graph_core::EdgeReference<$node, $edge, &'a mut E>
            where
                Self: 'a;
            /// id 为本图边身份；返回携带当前端点的独占权重引用，失效为 None；端点不可变，权重修改不改变版本。
            fn edge_mut(&mut self, id: $edge) -> Option<Self::EdgeRefMut<'_>> {
                self.edge_data_mut(id).map(|(source, target, weight)| {
                    graph_core::EdgeReference::new(id, source, target, weight)
                })
            }
        }
        impl<N, E> graph_core::EdgeEndpoints for $graph<N, E> {
            /// id 为本图边身份；返回当前源、目标，失效为 None，无副作用。
            fn endpoints(&self, id: $edge) -> Option<($node, $node)> {
                self.edge_data(id)
                    .map(|(source, target, _)| (source, target))
            }
        }
        impl<N, E> graph_core::OutEdges for $graph<N, E> {
            type OutEdges<'a>
                = $adj
            where
                Self: 'a;
            /// node 为本图节点；返回按插入顺序的出边，失效节点为空，无副作用。
            fn out_edges(&self, node: $node) -> Self::OutEdges<'_> {
                self.adjacent(node, true)
            }
        }
        impl<N, E> graph_core::InEdges for $graph<N, E> {
            type InEdges<'a>
                = $adj
            where
                Self: 'a;
            /// node 为本图节点；返回按插入顺序的入边，失效节点为空，无副作用。
            fn in_edges(&self, node: $node) -> Self::InEdges<'_> {
                self.adjacent(node, false)
            }
        }
        impl<N, E> graph_core::OutNeighbors for $graph<N, E> {
            type OutNeighbors<'a>
                = $crate::common::Neighbors<'a, Self>
            where
                Self: 'a;
            /// node 为本图节点；返回出边目标序列，保留自环、平行边重复和邻接顺序，失效节点为空，无副作用。
            fn out_neighbors(&self, node: $node) -> Self::OutNeighbors<'_> {
                $crate::common::Neighbors::outgoing(self, node)
            }
        }
        impl<N, E> graph_core::InNeighbors for $graph<N, E> {
            type InNeighbors<'a>
                = $crate::common::Neighbors<'a, Self>
            where
                Self: 'a;
            /// node 为本图节点；返回入边源序列，保留自环、平行边重复和邻接顺序，失效节点为空，无副作用。
            fn in_neighbors(&self, node: $node) -> Self::InNeighbors<'_> {
                $crate::common::Neighbors::incoming(self, node)
            }
        }
        impl<N, E> graph_core::GraphVersion for $graph<N, E> {
            type Version = u64;
            /// self 为当前图；返回单调拓扑版本，仅成功拓扑修改递增，无副作用。
            fn version(&self) -> u64 {
                self.version
            }
        }
        impl<N, E> graph_core::GraphInsert for $graph<N, E> {
            type Node = N;
            type Edge = E;
            type Error = $crate::GraphError;
            /// weight 为移入权重；返回新节点身份并递增版本；VersionExhausted 时图不变，流程见本类型 add_node。
            fn insert_node(&mut self, weight: N) -> Result<$node, Self::Error> {
                self.add_node(weight)
            }
            /// source、target 为本图端点，weight 为移入权重；成功返回新边身份并递增版本；MissingEndpoint/VersionExhausted 无修改，流程见本类型 add_edge。
            fn insert_edge(
                &mut self,
                source: $node,
                target: $node,
                weight: E,
            ) -> Result<$edge, Self::Error> {
                self.add_edge(source, target, weight)
            }
        }
        impl<N, E> graph_core::GraphRemove for $graph<N, E> {
            type Error = $crate::GraphError;
            /// id 为本图身份；有效时删除并返回 true、递增版本，失效幂等返回 false；VersionExhausted 时图不变；拓扑发布后销毁权重，流程见本类型 delete_node。
            fn remove_node(&mut self, id: $node) -> Result<bool, Self::Error> {
                self.delete_node(id)
            }
            /// id 为本图身份；有效时删除并返回 true、递增版本，失效幂等返回 false；VersionExhausted 时图不变；拓扑发布后销毁权重，流程见本类型 delete_edge。
            fn remove_edge(&mut self, id: $edge) -> Result<bool, Self::Error> {
                self.delete_edge(id)
            }
        }
        impl<N, E> graph_core::Visitable for $graph<N, E> {
            type Map = $map;
            /// self 为当前图；返回按 node_bound 创建的独立清零位图，无图副作用；拓扑改变后必须重建。
            fn visit_map(&self) -> Self::Map {
                <$map>::new(graph_core::NodeIndexable::node_bound(self))
            }
        }
        impl<N, E> graph_core::GraphSnapshot for $graph<N, E> {
            type Error = std::convert::Infallible;
            type View<'a>
                = &'a Self
            where
                Self: 'a;
            // 拓扑只能通过 &mut Self 修改；N/E 可能包含内部可变性，属性不保证固定。
            const DATA_STABLE: bool = false;
            /// self 为共享图；返回借用自身的固定拓扑视图，无业务错误、无副作用；内部可变权重不保证固定（DATA_STABLE=false）。
            fn snapshot(&self) -> Result<Self::View<'_>, Self::Error> {
                Ok(self)
            }
        }
        impl<N, E, Q: Fn(&N) -> bool + ?Sized> graph_query::NodeSearch<Q> for $graph<N, E> {
            type Error = std::convert::Infallible;
            type Iter<'a>
                = $crate::common::NodeMatches<'a, Self, Q>
            where
                Self: 'a,
                Q: 'a;
            /// 借用权重谓词执行全扫描；不会创建属性索引，谓词 panic 原样传播。
            /// query 为借用的节点权重谓词；返回按枚举顺序的惰性扫描，无业务错误，不建索引，谓词 panic 传播，不修改图。
            fn search_nodes<'a>(&'a self, query: &'a Q) -> Result<Self::Iter<'a>, Self::Error> {
                Ok($crate::common::NodeMatches {
                    graph: self,
                    query,
                    ids: self.node_keys(),
                })
            }
        }
        impl<N, E, Q: Fn(&E) -> bool + ?Sized> graph_query::EdgeSearch<Q> for $graph<N, E> {
            type Error = std::convert::Infallible;
            type Iter<'a>
                = $crate::common::EdgeMatches<'a, Self, Q>
            where
                Self: 'a,
                Q: 'a;
            /// query 为借用的边权重谓词；返回按枚举顺序的惰性扫描，无业务错误，不建索引，谓词 panic 传播，不修改图。
            fn search_edges<'a>(&'a self, query: &'a Q) -> Result<Self::Iter<'a>, Self::Error> {
                Ok($crate::common::EdgeMatches {
                    graph: self,
                    query,
                    ids: self.edge_keys(),
                })
            }
        }
        impl<N, E> graph_core::GraphBatchEdit for $graph<N, E> {
            type Node = N;
            type Edge = E;
            type Error = $crate::GraphError;
            /// 按批前身份执行；稠密删除的移动会在批内映射；身份有效期受各存储代际边界限制。
            /// 返回的稠密位置为各项执行时的位置，后续删除可能使它失效。
            /// edits 按批前身份、输入顺序消费；返回执行时身份结果，后续搬移可能使其失效；首错返回下标和已完成前缀并保留其修改，空批无修改；核心流程见 crate::batch::edit_batch。
            fn edit_batch(
                &mut self,
                edits: Vec<graph_core::BatchOperation<Self>>,
            ) -> graph_core::BatchOutcome<Self> {
                $crate::batch::edit_batch(self, edits)
            }
        }
        impl<N: Clone, E: Clone> graph_core::AtomicGraphBatchEdit for $graph<N, E> {
            /// 在克隆的结构与权重上执行，全部成功后替换存储；成本为本图 Clone 加批次执行，见各存储类型的克隆成本说明。
            /// Clone 不隔离权重内部共享状态；操作不通过共享权重修改拓扑。
            /// panic 不承诺回滚；业务 Drop panic 时已发布拓扑仍自洽。
            /// 可继续使用范围按阶段区分，见 crate::batch::edit_batch_atomic 的 panic 契约。
            /// edits 同普通批语义；克隆执行后全成功才发布，错误保留原拓扑并返回下标/原因；空批不克隆，无操作删除不替换权重；Clone 不隔离内部共享，panic 不承诺回滚，流程见 crate::batch::edit_batch_atomic。
            fn edit_batch_atomic(
                &mut self,
                edits: Vec<graph_core::BatchOperation<Self>>,
            ) -> graph_core::AtomicBatchOutcome<Self> {
                $crate::batch::edit_batch_atomic(self, edits)
            }
        }
    };
}
