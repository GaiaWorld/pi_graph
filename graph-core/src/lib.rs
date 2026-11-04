//! 与存储无关的图协议。自定义后端只需实现所用的小能力 Trait。
//! ID 在一次算法或遍历期间须保持有效，拓扑保持稳定。
//! 邻接身份属于当前视图，平行边允许重复邻居；不存在节点返回空邻接。
//! 具体组合视图位于 graph-adaptor；本库不反向依赖适配器、查询或算法。
//! src 文件保持平铺，共享引用适配位于 borrowed.rs。

/// 可选编辑、索引、访问集合及快照能力导航。
pub mod capabilities;
/// 最小只读能力与 ReadGraph 便捷门面。
pub mod core;

pub mod access;
pub mod base;
pub mod batch;
mod borrowed;
pub mod edit;
pub mod index;
pub mod mutation;
pub mod project;
pub mod reference;
pub mod snapshot;
pub mod visit;

pub use access::*;
pub use base::*;
pub use batch::*;
pub use core::ReadGraph;
pub use edit::*;
pub use index::*;
pub use mutation::*;
pub use project::Project;
pub use reference::*;
pub use snapshot::GraphSnapshot;
pub use visit::*;
