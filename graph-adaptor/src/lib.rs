//! 图适配器与组合视图。src 文件平铺，壳与配对芯同目录。
//! 依赖 graph-core 协议；算法只通过能力 Trait 使用视图，无需依赖本库。
//! 适配器业务与链式入口已有配对实现；验收状态见 design/graph-foundation/design.md。

// 供本库各壳使用协议，不将整个协议库再次公开重导出。
use graph_core::*;

pub mod adapt_ext;
pub mod adaptors;
pub mod indexed_subgraph;
pub mod map_ids;
pub mod mapped;
pub mod node_filtered;
pub mod selection;
pub mod subgraph;
pub mod view;

pub use adapt_ext::GraphAdaptExt;
pub use indexed_subgraph::{IndexedBuildResult, IndexedSubgraph, IndexedSubgraphError};
pub use map_ids::{IdMapping, MapIds};
pub use mapped::{MapEdges, MapNodes, MappedEdgeRef, MappedNodeRef, ValueMapper};
pub use node_filtered::NodeFiltered;
pub use selection::{AllEdges, AllNodes, Predicate, SelectedEdges, Selection};
pub use subgraph::Subgraph;
pub use view::{EdgeFiltered, Reversed};
