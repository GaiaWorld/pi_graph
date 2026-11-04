//! 显式选择策略，不以重叠 blanket 实现同时匹配集合与闭包。
use std::collections::HashSet;
use std::hash::Hash;
/// 身份选择，不证明元素存在；遍历期间结果须稳定，须能接收无效 ID。
pub trait Selection<Id> {
    /// id：待查身份；选中 true，否则 false，不修改状态。
    fn contains(&self, id: Id) -> bool;
}
/// 保留全部底层节点。
pub struct AllNodes;
/// 保留全部底层边，允许纯诱导子图使用仅邻居能力的后端。
pub struct AllEdges;
/// 拥有显式身份谓词的策略。
pub struct Predicate<F> {
    /// 只读谓词，内部可变性不得破坏选择稳定性。
    pub predicate: F,
}
/// 显式边选择，区分 AllEdges，以无重叠 impl 提供边条件邻居枚举。
pub struct SelectedEdges<S> {
    /// 集合、借用集合或 Predicate 等边身份选择策略。
    pub selection: S,
}
impl<Id> Selection<Id> for AllNodes {
    fn contains(&self, _id: Id) -> bool {
        true
    }
}
impl<Id> Selection<Id> for AllEdges {
    fn contains(&self, _id: Id) -> bool {
        true
    }
}
impl<Id, F: Fn(Id) -> bool> Selection<Id> for Predicate<F> {
    fn contains(&self, id: Id) -> bool {
        (self.predicate)(id)
    }
}
impl<Id, S: Selection<Id>> Selection<Id> for SelectedEdges<S> {
    fn contains(&self, id: Id) -> bool {
        self.selection.contains(id)
    }
}
impl<Id: Eq + Hash> Selection<Id> for HashSet<Id> {
    fn contains(&self, id: Id) -> bool {
        HashSet::contains(self, &id)
    }
}
impl<Id, S: Selection<Id> + ?Sized> Selection<Id> for &S {
    fn contains(&self, id: Id) -> bool {
        (**self).contains(id)
    }
}
