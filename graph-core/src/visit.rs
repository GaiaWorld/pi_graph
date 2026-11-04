use crate::GraphBase;

/// 枚举全部可见节点，包括孤立节点，每个身份恰好一次；枚举期间拓扑须稳定，顺序由后端决定。
pub trait NodeIds: GraphBase {
    /// 借用 self 的标准迭代器，返回身份而非数据引用，无需装箱。
    type NodeIds<'a>: Iterator<Item = Self::NodeId>
    where
        Self: 'a;
    /// 返回全部可见节点，包括孤立节点，每个身份恰好一次，不修改图。
    /// 返回的身份必须属于当前图或视图，过滤不得产生悬空端点。
    fn node_ids(&self) -> Self::NodeIds<'_>;
}

/// 枚举全部可见边，每个身份恰好一次；枚举期间拓扑须稳定，顺序由后端决定。
pub trait EdgeIds: GraphBase {
    /// 借用 self 的标准迭代器，返回身份而非数据引用，无需装箱。
    type EdgeIds<'a>: Iterator<Item = Self::EdgeId>
    where
        Self: 'a;
    /// 返回全部可见边，每个身份恰好一次，不修改图。
    /// 返回的身份必须属于当前图或视图，过滤不得产生悬空端点。
    fn edge_ids(&self) -> Self::EdgeIds<'_>;
}

/// 枚举出邻居，平行边允许重复；枚举期间拓扑须稳定，顺序由后端决定。
pub trait OutNeighbors: GraphBase {
    /// 借用 self 的标准迭代器，返回身份而非数据引用，无需装箱。
    type OutNeighbors<'a>: Iterator<Item = Self::NodeId>
    where
        Self: 'a;
    /// node：查询节点，不存在时返回空迭代器。返回出邻居，平行边允许重复，不修改图。
    /// 返回的身份必须属于当前图或视图，过滤不得产生悬空端点。
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_>;
}

/// 枚举入邻居，平行边允许重复；枚举期间拓扑须稳定，顺序由后端决定。
pub trait InNeighbors: GraphBase {
    /// 借用 self 的标准迭代器，返回身份而非数据引用，无需装箱。
    type InNeighbors<'a>: Iterator<Item = Self::NodeId>
    where
        Self: 'a;
    /// node：查询节点，不存在时返回空迭代器。返回入邻居，平行边允许重复，不修改图。
    /// 返回的身份必须属于当前图或视图，过滤不得产生悬空端点。
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_>;
}

/// 枚举可见出边，每条边恰好一次；枚举期间拓扑须稳定，顺序由后端决定。
pub trait OutEdges: GraphBase {
    /// 借用 self 的标准迭代器，返回身份而非数据引用，无需装箱。
    type OutEdges<'a>: Iterator<Item = Self::EdgeId>
    where
        Self: 'a;
    /// node：查询节点，不存在时返回空迭代器。返回可见出边，每条边恰好一次，不修改图。
    /// 返回的身份必须属于当前图或视图，过滤不得产生悬空端点。
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_>;
}

/// 枚举可见入边，每条边恰好一次；枚举期间拓扑须稳定，顺序由后端决定。
pub trait InEdges: GraphBase {
    /// 借用 self 的标准迭代器，返回身份而非数据引用，无需装箱。
    type InEdges<'a>: Iterator<Item = Self::EdgeId>
    where
        Self: 'a;
    /// node：查询节点，不存在时返回空迭代器。返回可见入边，每条边恰好一次，不修改图。
    /// 返回的身份必须属于当前图或视图，过滤不得产生悬空端点。
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_>;
}

/// 复用 gat-lending-iterator 的逐次借用迭代协议与组合器。
/// 仅供元素借用迭代器内部状态的后端使用，不替代标准 ID 迭代。
/// 返回元素的借用结束后才能再次推进；完整契约见原始 Trait 文档。
pub use gat_lending_iterator::LendingIterator;
