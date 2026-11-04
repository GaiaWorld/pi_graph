use crate::GraphBase;

/// 插入节点与边；正常返回 Err 时不得留下部分拓扑更新。
pub trait GraphInsert: GraphBase {
    /// 插入节点时转移给后端的业务数据。
    type Node;
    /// 插入边时转移给后端的业务数据。
    type Edge;
    /// 后端定义的插入错误；不承诺在错误中返还已消费的数据。
    type Error;
    /// node：交给后端的新节点数据。成功返回新身份；失败返回后端错误且拓扑不变。
    /// 重复调用视为独立插入，不保证幂等。
    fn insert_node(&mut self, node: Self::Node) -> Result<Self::NodeId, Self::Error>;
    /// source、target：已存在的源与目标节点；edge：转移给后端的边数据。
    /// 成功返回新边身份；失败返回后端错误且拓扑不变，不保证返还 edge。
    /// 自环和平行边策略由后端决定，重复调用不保证幂等。
    fn insert_edge(
        &mut self,
        source: Self::NodeId,
        target: Self::NodeId,
        edge: Self::Edge,
    ) -> Result<Self::EdgeId, Self::Error>;
}

/// 删除拓扑元素；正常返回 Err 时不得留下部分拓扑更新。
pub trait GraphRemove: GraphBase {
    /// 后端定义的删除错误。
    type Error;
    /// node：待删除身份。存在则删除节点及所有关联边，返回 true；不存在返回 false。
    /// Err 时拓扑不变；不插入新节点的情况下，重复删除同一身份返回 false。
    fn remove_node(&mut self, node: Self::NodeId) -> Result<bool, Self::Error>;
    /// edge：待删除身份。存在则删除并返回 true，不存在返回 false。
    /// Err 时拓扑不变；不插入新边的情况下，重复删除同一身份返回 false。
    fn remove_edge(&mut self, edge: Self::EdgeId) -> Result<bool, Self::Error>;
}
