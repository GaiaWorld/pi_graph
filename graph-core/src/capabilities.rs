//! Capabilities：按后端真实能力选择实现，不构成必须全部满足的大图接口。
//! 拓扑编辑、普通可写访问、受控索引修改、位置映射和一致读取相互独立。
pub use crate::access::{EdgeMut, NodeMut};
pub use crate::base::GraphVersion;
pub use crate::batch::{
    AtomicBatchEditError, AtomicBatchOutcome, AtomicGraphBatchEdit, BatchEditError, BatchOperation,
    BatchOutcome, BatchResults, GraphBatchEdit, TopologyEdit, TopologyEditResult,
};
pub use crate::edit::{GraphInsert, GraphRemove};
pub use crate::index::{DenseNodeIndex, NodeIndexable, VisitMap, Visitable};
pub use crate::mutation::{IndexedNodeMut, MutationListener, NodeProperty, NodePropertyChanged};
pub use crate::reference::{EdgeRefMut, NodeRefMut};
pub use crate::snapshot::GraphSnapshot;
