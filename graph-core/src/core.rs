//! Core：最小只读图能力与便捷复合约束。
//! Neighbors 分为 OutNeighbors / InNeighbors，IncidentEdges 分为 OutEdges / InEdges，
//! 避免用未指定方向的统一名称掩盖有向语义。
//! 算法依赖实际所需的小 Trait，不以 ReadGraph 替代最小约束。
pub use crate::access::{EdgeAccess, EdgeEndpoints, NodeAccess};
pub use crate::base::GraphBase;
pub use crate::reference::{EdgeRef, EdgeReference, NodeRef, NodeReference};
pub use crate::visit::{EdgeIds, InEdges, InNeighbors, NodeIds, OutEdges, OutNeighbors};

/// 面向应用与工具层的只读有向图便捷门面，不增加行为或新的存储要求。
/// 组合节点/边数据访问、身份枚举及双向关联边；关联端点可由 EdgeRef 取得。
/// 不隐含邻居枚举、索引、快照、版本或可写能力，不是所有图必须实现的基础接口。
/// 满足小 Trait 的类型自动实现，包括动态大小类型，无需后端重复声明。
/// 算法应继续使用最小能力，例如拓扑排序只要求 NodeIds + OutNeighbors。
pub trait ReadGraph: NodeAccess + EdgeAccess + NodeIds + EdgeIds + OutEdges + InEdges {}
impl<G: ?Sized> ReadGraph for G where
    G: NodeAccess + EdgeAccess + NodeIds + EdgeIds + OutEdges + InEdges
{
}
