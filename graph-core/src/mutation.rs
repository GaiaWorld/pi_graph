//! 类型化节点属性修改及提交后的观察事件；后端负责数据和索引一致性。
use crate::GraphBase;

/// 后端定义的节点属性键；可以是零大小类型或携带字段标识的值。
pub trait NodeProperty {
    /// 该键对应的完整属性值类型；不强制 Clone、Copy、Eq 或统一 Value 枚举。
    type Value;
}

/// 单次成功节点属性修改的借用事件，不拥有图或值；仅在同步回调期间有效。
pub struct NodePropertyChanged<'a, N, K: NodeProperty> {
    /// 被修改节点身份，拓扑和身份均未改变。
    pub node: N,
    /// 此次修改的具体属性键，借用本次 setter 的键。
    pub key: &'a K,
    /// 修改前的属性值；后端须在提交前保留，不能读取已覆盖的数据冒充旧值。
    pub before: &'a K::Value,
    /// 已提交的新属性值，与更新后的数据及索引一致。
    pub after: &'a K::Value,
}

/// 成功修改后的同步观察者，不参与数据或索引的提交。
/// 后端负责监听器注册、持有与注销；本协议不引入全局注册表。
pub trait MutationListener<N, K: NodeProperty> {
    /// event：已成功提交的修改，借用不得逃出本次调用。
    /// 正常成功 setter 须通知当前适用监听器各一次；Err 不产生事件。
    /// 相同值的成功设置也通知一次，不要求属性值可比较；监听器之间顺序不保证。
    /// 不返回业务错误，不得重入修改同一图；可更新观察者自身状态。
    /// panic 向调用方传播，已提交数据和索引不回滚，后续监听器不保证收到事件。
    /// 不承诺持久化、跨线程排队或可靠投递。
    fn on_node_property_changed(&mut self, event: &NodePropertyChanged<'_, N, K>);
}

/// 可选受控属性修改能力，不要求完整节点 Weight，也不依赖 NodeMut。
/// 数据与索引的一致性必须由后端保证，不通过可失败外部监听器完成索引提交。
/// 若同时提供 NodeMut 或其他可写途径，该途径也必须维护索引，或在允许查询前
/// 可靠失效并重建；不能暴露过期索引结果，做不到则不应提供旁路写入能力。
pub trait IndexedNodeMut<K: NodeProperty>: GraphBase {
    /// 后端定义的缺失节点、无效属性、验证或索引更新错误。
    type Error;
    /// id：有效目标节点；key：后端支持的属性键；value：转移给后端的新值。
    /// 成功前完成校验、旧值保留以及节点数据与全部相关索引的提交，再同步发事件。
    /// 返回 Ok(()) 表示提交且通知正常结束；不存在节点、无效键、值校验或索引
    /// 更新失败返回 Err，节点数据及索引保持原状且不发事件，不承诺返还 value。
    /// 不改变拓扑、身份或 GraphVersion；不保证多次设置的事务原子性。
    /// 重复设置可重复发事件，因而观察副作用不幂等。
    /// # Panics
    /// Listener panic 时提交已经完成，不回滚、不转换为 Err；其他 panic 的恢复
    /// 由后端声明，但后端若允许继续查询，仍不得暴露数据与索引不一致的状态。
    fn set_node_property(
        &mut self,
        id: Self::NodeId,
        key: K,
        value: K::Value,
    ) -> Result<(), Self::Error>;
}
