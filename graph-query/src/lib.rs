//! Higher-Level：Walker、只读 Project 与后端 Search 查询能力。
//! 默认 Walker 不持有图借用；索引优化入口借用图以映射 ID 到位图。
//! Project 在 graph-core 单源声明，这里提供高层使用入口。
//! 能力协议位于低层 crate 不代表消费者必须实现它，也不使 core 反向依赖查询层。
//! 同一 Walker 必须始终用于同一拓扑，起点必须是有效节点。
//! 两次 next 之间可在访问集合的借用约束内修改业务数据；拓扑变更后须重建 Walker。
pub use graph_core::Project;
pub mod query;
pub use query::{EdgeSearch, NodeSearch};
pub mod search;
pub mod walker;
pub use search::{find_node, reachable, reachable_indexed};
pub use walker::{Bfs, Dfs, Walker, WalkerIter};
