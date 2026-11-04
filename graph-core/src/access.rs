use crate::GraphBase;
use crate::{EdgeRef, EdgeRefMut, NodeRef, NodeRefMut};

/// 按节点身份取得引用对象；负责查找，NodeRef 负责取得后的读取，不规定存储与锁。
pub trait NodeAccess: GraphBase {
    /// 节点业务数据，允许动态大小类型。
    type Node: ?Sized;
    /// 借用 self 的带身份引用对象，可包装普通引用或 Guard，不要求 Copy/Clone。
    type NodeRef<'a>: NodeRef<NodeId = Self::NodeId, Weight = Self::Node>
    where
        Self: 'a;
    /// id：待查节点。存在则返回访问对象，不存在返回 None，不修改拓扑。
    /// 锁定、重入及借用冲突行为沿用后端；返回对象存活期间须遵守后端约束。
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>>;
}

/// 按边身份取得包含 ID、端点与数据的引用对象；无需额外实现 EdgeEndpoints。
/// 同时实现 EdgeEndpoints 时，两种访问得到的端点必须一致。
pub trait EdgeAccess: GraphBase {
    /// 边业务数据，允许动态大小类型。
    type Edge: ?Sized;
    /// 借用 self 的带身份引用对象，可包装普通引用或 Guard，不要求 Copy/Clone。
    type EdgeRef<'a>: EdgeRef<NodeId = Self::NodeId, EdgeId = Self::EdgeId, Weight = Self::Edge>
    where
        Self: 'a;
    /// id：待查边。存在则返回访问对象，不存在返回 None，不修改拓扑。
    /// 锁定、重入及借用冲突行为沿用后端；返回对象存活期间须遵守后端约束。
    fn edge(&self, id: Self::EdgeId) -> Option<Self::EdgeRef<'_>>;
}

/// 读取有向边的连接关系，不要求访问边业务数据。
pub trait EdgeEndpoints: GraphBase {
    /// edge：待查边。不存在返回 None；存在返回 (源节点 ID, 目标节点 ID)。
    /// 两个端点必须属于当前图或视图；不修改图。
    fn endpoints(&self, edge: Self::EdgeId) -> Option<(Self::NodeId, Self::NodeId)>;
}

/// 独占修改节点业务数据，不改变拓扑，不承诺自动维护属性索引。
/// 若后端同时保证属性索引查询正确，则必须在此写入路径维护索引，或在查询前
/// 使其失效并重建；不能以本能力不承诺自动维护为由暴露过期索引结果。
pub trait NodeMut: NodeAccess {
    /// 借用 self 的可写访问对象；释放后结束该次访问。
    type NodeRefMut<'a>: NodeRefMut<NodeId = Self::NodeId, Weight = Self::Node>
    where
        Self: 'a;
    /// id：待查节点。存在返回可写对象，否则返回 None；修改数据不得使拓扑或 ID 失效。
    /// 锁定、重入及借用冲突行为沿用后端。
    fn node_mut(&mut self, id: Self::NodeId) -> Option<Self::NodeRefMut<'_>>;
}

/// 独占修改边业务数据，不改变拓扑，不承诺自动维护属性索引。
pub trait EdgeMut: EdgeAccess {
    /// 借用 self 的可写访问对象；释放后结束该次访问。
    type EdgeRefMut<'a>: EdgeRefMut<
        NodeId = Self::NodeId,
        EdgeId = Self::EdgeId,
        Weight = Self::Edge,
    >
    where
        Self: 'a;
    /// id：待查边。存在返回可写对象，否则返回 None；修改数据不得使端点或 ID 失效。
    /// 锁定、重入及借用冲突行为沿用后端。
    fn edge_mut(&mut self, id: Self::EdgeId) -> Option<Self::EdgeRefMut<'_>>;
}
