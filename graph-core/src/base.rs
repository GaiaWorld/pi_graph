use std::hash::Hash;

/// 定义节点和边的身份，不要求连续索引；遍历期间身份须稳定有效。
pub trait GraphBase {
    /// 节点身份；相等性与哈希在有效期内须一致，不同有效节点不共用身份。
    type NodeId: Copy + Eq + Hash;
    /// 边身份；平行边仍应拥有各自独立的身份。
    type EdgeId: Copy + Eq + Hash;
}

/// 可选的拓扑版本能力；每次拓扑变更都应改变版本。
/// 现有 Walker 不自动读取该能力，也不据此支持拓扑变更。
pub trait GraphVersion: GraphBase {
    /// 同一图的拓扑版本标识；只能用于该后端保证的比较区间。
    /// 不透明相等性令牌，不承诺连续整数、大小顺序或每批加一；不能跨图比较。
    type Version: Copy + Eq;
    /// 返回当前拓扑版本，不修改图；业务数据修改无需改变版本。
    fn version(&self) -> Self::Version;
}
