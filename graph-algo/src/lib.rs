//! Higher-Level：仅依赖最小图能力的算法及工作区，支持稀疏及自定义 ID。
//! 不将 ReadGraph 便捷门面作为算法的统一约束；SCC 与 ShortestPath 尚未设计。
pub mod bfs;
pub mod dfs;
pub mod topo;
pub use bfs::bfs;
pub use dfs::dfs;
pub use topo::{toposort, toposort_with_workspace, CycleError, TopoWorkspace};
