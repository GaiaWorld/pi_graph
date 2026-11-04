//! 元素引用协议与普通借用、Guard 的通用包装；不拥有图存储。
use std::ops::{Deref, DerefMut};

/// 已取得节点的身份与业务数据，不要求 Copy、Clone 或 Deref。
pub trait NodeRef {
    /// 节点身份，与取得该引用时的查找 ID 一致。
    type NodeId: Copy;
    /// 节点业务数据，允许动态大小类型。
    type Weight: ?Sized;
    /// 返回节点身份，不改变状态。
    fn id(&self) -> Self::NodeId;
    /// 返回借自当前引用对象的数据；不得超过对象及其 Guard 的有效期。
    fn weight(&self) -> &Self::Weight;
}
/// 已取得有向边的身份、端点与数据；方向必须与取得它的视图一致。
pub trait EdgeRef {
    /// 端点节点身份。
    type NodeId: Copy;
    /// 当前边身份，与查找 ID 一致。
    type EdgeId: Copy;
    /// 边业务数据，允许动态大小类型。
    type Weight: ?Sized;
    /// 返回当前边身份，不改变状态。
    fn id(&self) -> Self::EdgeId;
    /// 返回当前视图中的源节点，不改变状态。
    fn source(&self) -> Self::NodeId;
    /// 返回当前视图中的目标节点，不改变状态。
    fn target(&self) -> Self::NodeId;
    /// 返回借自当前引用对象的数据，不延长 Guard 的生命周期。
    fn weight(&self) -> &Self::Weight;
}
/// 可写节点引用；只修改数据，身份与拓扑保持不变，不承诺属性索引自动维护。
pub trait NodeRefMut: NodeRef {
    /// 返回借自当前对象的可写数据；再次访问须结束当前独占借用。
    fn weight_mut(&mut self) -> &mut Self::Weight;
}
/// 可写边引用；只修改数据，身份和端点保持不变，不承诺属性索引自动维护。
pub trait EdgeRefMut: EdgeRef {
    /// 返回借自当前对象的可写数据；再次访问须结束当前独占借用。
    fn weight_mut(&mut self) -> &mut Self::Weight;
}

// 以下属于通用引用表示及原语适配，不承担算法主流程，无需人为拆芯。
/// 为普通引用或 Guard 附加节点身份；D 的析构负责释放访问资源。
pub struct NodeReference<N, D> {
    /// 当前节点身份，不可经可写数据访问改变。
    id: N,
    /// 拥有的借用或 Guard；数据借用不超过其存活期。
    data: D,
}
impl<N, D> NodeReference<N, D> {
    /// id：被访问节点；data：该节点的借用或 Guard，调用方保证二者对应。
    /// 转移 data 所有权，不访问数据、不分配存储。
    pub fn new(id: N, data: D) -> Self {
        Self { id, data }
    }
}
impl<N: Copy, D: Deref> NodeRef for NodeReference<N, D> {
    type NodeId = N;
    type Weight = D::Target;
    fn id(&self) -> N {
        self.id
    }
    fn weight(&self) -> &Self::Weight {
        &self.data
    }
}
impl<N: Copy, D: DerefMut> NodeRefMut for NodeReference<N, D> {
    fn weight_mut(&mut self) -> &mut Self::Weight {
        &mut self.data
    }
}
/// 为普通引用或 Guard 附加边身份与当前视图端点，不复制数据。
pub struct EdgeReference<N, E, D> {
    /// 当前边身份。
    id: E,
    /// 当前视图中的源节点。
    source: N,
    /// 当前视图中的目标节点。
    target: N,
    /// 拥有的借用或 Guard，负责数据生命周期与访问资源释放。
    data: D,
}
impl<N, E, D> EdgeReference<N, E, D> {
    /// id、source、target：当前视图的边及端点；data：对应数据借用或 Guard。
    /// 调用方保证身份与数据一致；构造只转移所有权，不访问数据或分配存储。
    pub fn new(id: E, source: N, target: N, data: D) -> Self {
        Self {
            id,
            source,
            target,
            data,
        }
    }
}
impl<N: Copy, E: Copy, D: Deref> EdgeRef for EdgeReference<N, E, D> {
    type NodeId = N;
    type EdgeId = E;
    type Weight = D::Target;
    fn id(&self) -> E {
        self.id
    }
    fn source(&self) -> N {
        self.source
    }
    fn target(&self) -> N {
        self.target
    }
    fn weight(&self) -> &Self::Weight {
        &self.data
    }
}
impl<N: Copy, E: Copy, D: DerefMut> EdgeRefMut for EdgeReference<N, E, D> {
    fn weight_mut(&mut self) -> &mut Self::Weight {
        &mut self.data
    }
}
