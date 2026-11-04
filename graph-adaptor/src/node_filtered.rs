//! 节点谓词过滤是通用子图的便捷名称；真实核心结构单源定义在 subgraph.rs。
use crate::selection::{AllEdges, Predicate};
/// 仅节点谓词选择的诱导子图，保留原身份与最小邻居能力。
pub type NodeFiltered<G, F> = crate::subgraph::Subgraph<G, Predicate<F>, AllEdges>;
/// 旧边候选名称的兼容导出，泛型形状随通用子图迁移。
pub use crate::subgraph::SubgraphEdges as VisibleEdges;
/// 旧节点候选名称的兼容导出，真实字段在 subgraph.rs。
pub use crate::subgraph::SubgraphNodes as VisibleNodes;
