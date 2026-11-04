//! Adaptors：不复制底层图的组合视图，能力保留遵循各视图的契约。
//! EdgeFiltered 按边过滤，NodeFiltered 表达节点诱导子图（Subgraph 语义），
//! Reversed 改变方向。这里重导出真实定义，不重复定义包装或掩盖核心字段。
//! Subgraph 是通用惰性结构视图，MapNodes/MapEdges 按值改变数据；MapIds 双向翻译身份，IndexedSubgraph 固定一次结构选择；新增入口均为设计稿。
//! 通用 SnapshotView 未设计；快照 View 由 GraphSnapshot 的后端关联类型提供。
//! 重导出不表示所有方法已实现；节点过滤和新增索引转发仍处于可编译设计稿状态。
pub use crate::adapt_ext::GraphAdaptExt;
pub use crate::indexed_subgraph::{IndexedBuildResult, IndexedSubgraph, IndexedSubgraphError};
pub use crate::map_ids::{
    IdMapping, MapIds, TranslatedEdgeRef, TranslatedIds, TranslatedNodeRef, TranslatedVisitMap,
};
pub use crate::mapped::{MapEdges, MapNodes, MappedEdgeRef, MappedNodeRef, ValueMapper};
pub use crate::node_filtered::{NodeFiltered, VisibleEdges, VisibleNodes};
pub use crate::selection::{AllEdges, AllNodes, Predicate, SelectedEdges, Selection};
pub use crate::subgraph::Subgraph;
pub use crate::view::{EdgeFiltered, EdgeNeighbors, FilterEdges, Reversed, ReversedEdgeRef};
