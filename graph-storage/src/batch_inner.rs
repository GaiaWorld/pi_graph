//! 批次编排的配对实现；仅通过批次壳协议调用具体存储。
use super::*;
pub(super) fn edit_batch<G: BatchStorage>(
    graph: &mut G,
    edits: Vec<BatchOperation<G>>,
) -> BatchOutcome<G> {
    use graph_core::{TopologyEdit as Op, TopologyEditResult as Done};
    if edits.is_empty() {
        return Ok(Vec::new());
    }
    let mut ids = crate::common::BatchIds::new(graph, &edits);
    let mut completed = Vec::with_capacity(edits.len());
    for edit in edits {
        let result = match edit {
            Op::InsertNode { weight } => graph.add_node(weight).map(|id| Done::NodeInserted { id }),
            Op::InsertEdge {
                source,
                target,
                weight,
            } => {
                match (
                    ids.nodes.get(&source).copied(),
                    ids.nodes.get(&target).copied(),
                ) {
                    (Some(source), Some(target)) => graph
                        .add_edge(source, target, weight)
                        .map(|id| Done::EdgeInserted { id }),
                    _ => Err(crate::GraphError::MissingEndpoint),
                }
            }
            Op::RemoveNode { node } => match ids.nodes.get(&node).copied() {
                Some(id) => graph
                    .batch_delete_node(id, &mut ids)
                    .map(|removed| Done::NodeRemoved { removed }),
                None => Ok(Done::NodeRemoved { removed: false }),
            },
            Op::RemoveEdge { edge } => match ids.edges.get(&edge).copied() {
                Some(id) => graph
                    .batch_delete_edge(id, &mut ids)
                    .map(|removed| Done::EdgeRemoved { removed }),
                None => Ok(Done::EdgeRemoved { removed: false }),
            },
        };
        match result {
            Ok(value) => completed.push(value),
            Err(error) => {
                return Err(graph_core::BatchEditError {
                    failed_at: completed.len(),
                    completed,
                    error,
                })
            }
        }
    }
    Ok(completed)
}
pub(super) fn edit_batch_atomic<G: BatchStorage + Clone>(
    graph: &mut G,
    edits: Vec<BatchOperation<G>>,
) -> AtomicBatchOutcome<G> {
    if edits.is_empty() {
        return Ok(Vec::new());
    }
    let mut staged = graph.clone();
    match graph_core::GraphBatchEdit::edit_batch(&mut staged, edits) {
        Ok(results) => {
            // 全为无操作删除时不替换权重对象。
            if graph.version() != staged.version() {
                let old = std::mem::replace(graph, staged);
                drop(old);
            }
            Ok(results)
        }
        Err(error) => Err(graph_core::AtomicBatchEditError {
            failed_at: Some(error.failed_at),
            error: error.error,
        }),
    }
}
