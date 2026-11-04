//! 消费 self 的组合入口；不强制统一 GraphView 基类。
#[path = "adapt_ext_inner.rs"]
mod adapt_ext_inner;
use crate::mapped::{MapEdges, MapNodes, ValueMapper};
use crate::selection::{AllEdges, Predicate, SelectedEdges, Selection};
use crate::subgraph::Subgraph;
use crate::view::{EdgeFiltered, Reversed};
use crate::{EdgeAccess, GraphBase, NodeAccess};

/// 所有图的只读组合入口；self 可为 &G 或已有适配器，返回值拥有中间层。
/// 构造不复制底层图；仅 map_ids 改变对外身份表示，不创建存储身份。
/// 能力按具体适配器逐项传播，不自动暴露修改接口；indexed_subgraph 额外构建结构缓存。
pub trait GraphAdaptExt: GraphBase + Sized {
    /// nodes、edges：覆盖当前可见元素的稳定双向映射；构造不验证全图，不复制数据。
    /// 对外 ID 类型会改变，拓扑不变；已有元素无法转换是契约违例，不静默过滤。
    fn map_ids<
        NM: crate::map_ids::IdMapping<Self::NodeId>,
        EM: crate::map_ids::IdMapping<Self::EdgeId>,
    >(
        self,
        nodes: NM,
        edges: EM,
    ) -> crate::map_ids::MapIds<Self, NM, EM> {
        adapt_ext_inner::map_ids(self, nodes, edges)
    }
    /// nodes、edges：本次选择策略，构建后不再调用；self 须在构建和全部使用期间拓扑稳定。
    /// 扫描完整身份与端点，成功返回固定结构缓存，错误不发布半成品，不保证返还 self。
    /// 自有稠密索引不等于底层槽位，权重不复制；全部失败语义见 IndexedSubgraph::build。
    fn indexed_subgraph<N: Selection<Self::NodeId>, E: Selection<Self::EdgeId>>(
        self,
        nodes: N,
        edges: E,
    ) -> crate::indexed_subgraph::IndexedBuildResult<Self>
    where
        Self: crate::NodeIds + crate::EdgeIds + crate::EdgeEndpoints,
    {
        adapt_ext_inner::indexed_subgraph(self, nodes, edges)
    }
    /// nodes：节点集合或选择策略，须在遍历期间稳定；构造惰性节点诱导子图。
    fn subgraph<N: Selection<Self::NodeId>>(self, nodes: N) -> Subgraph<Self, N, AllEdges> {
        adapt_ext_inner::subgraph(self, nodes)
    }
    /// nodes、edges：节点/边身份策略；可见边还要求双端点可见，不求值或复制图。
    fn subgraph_with_edges<N: Selection<Self::NodeId>, E: Selection<Self::EdgeId>>(
        self,
        nodes: N,
        edges: E,
    ) -> Subgraph<Self, N, SelectedEdges<E>> {
        adapt_ext_inner::subgraph_with_edges(self, nodes, edges)
    }
    /// predicate：稳定节点 ID 谓词，等同 subgraph(Predicate { predicate })。
    fn filter_nodes<F: Fn(Self::NodeId) -> bool>(
        self,
        predicate: F,
    ) -> Subgraph<Self, Predicate<F>, AllEdges> {
        adapt_ext_inner::filter_nodes(self, predicate)
    }
    /// predicate：稳定边 ID 谓词，须接受任意 ID（包括不存在的边）；true 不证明存在。
    /// 保留全部节点，不隐含 EdgeAccess 要求；权重谓词须将缺失访问处理为 false。
    fn filter_edges<F: Fn(Self::EdgeId) -> bool>(self, predicate: F) -> EdgeFiltered<Self, F> {
        adapt_ext_inner::filter_edges(self, predicate)
    }
    /// 反转当前视图方向，拥有中间层；数据与身份不变。
    fn reversed(self) -> Reversed<Self> {
        adapt_ext_inner::reversed(self)
    }
    /// mapper：借用节点数据计算独立值，不得借用本次输入逃逸；不提供可写能力。
    fn map_nodes<F: ValueMapper<Self::Node>>(self, mapper: F) -> MapNodes<Self, F>
    where
        Self: NodeAccess,
    {
        adapt_ext_inner::map_nodes(self, mapper)
    }
    /// mapper：借用边数据计算独立值；保留当前方向端点，Guard 只在计算期间持有。
    fn map_edges<F: ValueMapper<Self::Edge>>(self, mapper: F) -> MapEdges<Self, F>
    where
        Self: EdgeAccess,
    {
        adapt_ext_inner::map_edges(self, mapper)
    }
}
impl<G: GraphBase> GraphAdaptExt for G {}
