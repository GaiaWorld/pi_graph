//! 常用有向多重图存储，支持自环与平行边。
//! SparseGraph 使用代际键，DenseGraph 使用带代际的位置 ID；ID 仅属于创建它的图。
//! 普通分配失败沿用 Rust 容器行为；不捕获业务权重 Drop/查询闭包的 panic。
//!
//! ```
//! use graph_storage::SparseGraph;
//! use graph_core::{GraphInsert, Visitable};
//! use graph_query::{Bfs, Walker};
//!
//! let mut graph = SparseGraph::<&str, i32>::new();
//! let a = graph.insert_node("a")?;
//! let b = graph.insert_node("b")?;
//! graph.insert_edge(a, b, 1)?;
//! let order: Vec<_> = Bfs::with_visit_map(&graph, a, graph.visit_map())
//!     .iter(&graph).collect();
//! assert_eq!(order, vec![a, b]);
//! # Ok::<(), graph_storage::GraphError>(())
//! ```

#[macro_use]
mod common;
mod batch;
pub mod dense;
pub mod sparse;

pub use common::{GraphError, SlotVisitMap, VecVisitMap};
pub use dense::{DenseEdgeId, DenseGraph, DenseNodeId};
pub use sparse::{SparseEdgeId, SparseGraph, SparseNodeId};
