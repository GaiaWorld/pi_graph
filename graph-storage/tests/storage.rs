use graph_core::*;
use graph_query::{reachable_indexed, Bfs, Walker};
use graph_storage::{DenseGraph, SparseGraph};

// 公共 seam：普通存储的能力协议与已有算法之间的集成。
fn basic_contract<G>(mut graph: G)
where
    G: GraphInsert<Node = i32, Edge = i32>
        + GraphRemove
        + NodeAccess<Node = i32>
        + EdgeAccess<Edge = i32>
        + NodeMut
        + EdgeMut
        + NodeIds
        + EdgeIds
        + OutNeighbors
        + InNeighbors
        + OutEdges
        + InEdges
        + EdgeEndpoints
        + NodeIndexable
        + DenseNodeIndex
        + Visitable
        + GraphVersion,
    <G as GraphInsert>::Error: std::fmt::Debug,
    <G as GraphRemove>::Error: std::fmt::Debug,
    G::NodeId: std::fmt::Debug,
    G::EdgeId: std::fmt::Debug,
{
    assert_eq!(graph.node_ids().count(), 0);
    assert_eq!(graph.edge_ids().count(), 0);
    let before = graph.version();
    let a = graph.insert_node(10).unwrap();
    let b = graph.insert_node(20).unwrap();
    let isolated = graph.insert_node(30).unwrap();
    assert!(before != graph.version());
    let edge = graph.insert_edge(a, b, 1).unwrap();
    graph.insert_edge(a, b, 2).unwrap();
    graph.insert_edge(a, a, 3).unwrap();
    assert_eq!(graph.out_neighbors(a).collect::<Vec<_>>(), vec![b, b, a]);
    assert_eq!(graph.in_neighbors(b).collect::<Vec<_>>(), vec![a, a]);
    assert_eq!(graph.out_edges(a).count(), 3);
    assert_eq!(graph.in_edges(a).count(), 1);
    assert_eq!(graph.endpoints(edge), Some((a, b)));
    assert_eq!(graph.node_count(), 3);
    for id in graph.node_ids() {
        assert_eq!(graph.node_at_index(graph.to_index(id)), Some(id));
        assert_eq!(graph.node_at_dense(graph.to_dense(id).unwrap()), Some(id));
        assert!(graph.to_index(id) < graph.node_bound());
    }
    let version = graph.version();
    *graph.node_mut(a).unwrap().weight_mut() = 11;
    *graph.edge_mut(edge).unwrap().weight_mut() = 12;
    assert!(version == graph.version());
    assert_eq!(*graph.node(a).unwrap().weight(), 11);
    assert_eq!(*graph.edge(edge).unwrap().weight(), 12);
    assert_eq!(
        Bfs::with_visit_map(&graph, a, graph.visit_map())
            .iter(&graph)
            .collect::<Vec<_>>(),
        vec![a, b]
    );
    assert!(reachable_indexed(&graph, a, b));
    assert!(!reachable_indexed(&graph, a, isolated));
    let reversed = graph_adaptor::Reversed::new(&graph);
    assert_eq!(graph_algo::bfs(&reversed, b), vec![b, a]);
    assert_eq!(graph_algo::toposort(&graph).unwrap_err().remaining.len(), 2);
    assert!(graph.remove_edge(edge).unwrap());
    assert!(!graph.remove_edge(edge).unwrap());
    assert_eq!(graph.edge_ids().count(), 2);
    assert_eq!(graph.in_neighbors(b).collect::<Vec<_>>(), vec![a]);
    assert!(graph.remove_node(a).unwrap());
    assert!(!graph.remove_node(a).unwrap());
    assert_eq!(graph.node_ids().count(), 2);
    assert_eq!(graph.edge_ids().count(), 0);
    for id in graph.node_ids() {
        assert_eq!(graph.out_neighbors(id).count(), 0);
        assert_eq!(graph.in_neighbors(id).count(), 0);
        assert_eq!(graph.node_at_dense(graph.to_dense(id).unwrap()), Some(id));
    }
}

#[test]
fn sparse_graph_implements_basic_capabilities() {
    basic_contract(SparseGraph::new());
}

#[test]
fn dense_graph_implements_basic_capabilities() {
    basic_contract(DenseGraph::new());
}

#[test]
fn sparse_generation_and_dense_compaction_preserve_topology() {
    let mut graph = SparseGraph::<i32, ()>::new();
    let a = graph.insert_node(1).unwrap();
    let b = graph.insert_node(2).unwrap();
    let c = graph.insert_node(3).unwrap();
    let ab = graph.insert_edge(a, b, ()).unwrap();
    let cb = graph.insert_edge(c, b, ()).unwrap();
    let slot = graph.to_index(a);
    graph.remove_node(a).unwrap();
    assert!(graph.node(a).is_none());
    assert!(graph.edge(ab).is_none());
    assert_eq!(graph.node_at_index(slot), None);
    let replacement = graph.insert_node(4).unwrap();
    assert_ne!(replacement, a);
    assert_eq!(graph.to_index(replacement), slot);
    assert_eq!(graph.endpoints(cb), Some((c, b)));
    let version = graph.version();
    assert_eq!(
        graph.insert_edge(a, b, ()),
        Err(graph_storage::GraphError::MissingEndpoint)
    );
    assert_eq!(graph.version(), version);
    assert!(!graph.remove_node(a).unwrap());
    assert!(!graph.remove_edge(ab).unwrap());
    assert_eq!(graph.out_neighbors(a).count(), 0);

    let mut graph = DenseGraph::<i32, i32>::new();
    let nodes: Vec<_> = (0..4)
        .map(|value| graph.insert_node(value).unwrap())
        .collect();
    graph.insert_edge(nodes[3], nodes[2], 10).unwrap();
    graph.insert_edge(nodes[0], nodes[1], 11).unwrap();
    graph.insert_edge(nodes[3], nodes[3], 12).unwrap();
    graph.insert_edge(nodes[2], nodes[3], 13).unwrap();
    graph.remove_node(nodes[1]).unwrap();
    let moved = graph.node_at_index(1).unwrap();
    assert_eq!(*graph.node(moved).unwrap().weight(), 3);
    assert!(graph.node(nodes[3]).is_none());
    assert_eq!(graph.edge_count(), 3);
    for edge in graph.edge_ids() {
        let (s, t) = graph.endpoints(edge).unwrap();
        assert!(graph.out_edges(s).any(|id| id == edge));
        assert!(graph.in_edges(t).any(|id| id == edge));
    }
    assert_eq!(
        graph.out_neighbors(moved).collect::<Vec<_>>(),
        vec![nodes[2], moved]
    );
    assert_eq!(
        graph.in_neighbors(moved).collect::<Vec<_>>(),
        vec![moved, nodes[2]]
    );
}

fn batch_and_query_contract<G>(mut graph: G)
where
    G: GraphInsert<Node = i32, Edge = i32>
        + AtomicGraphBatchEdit
        + GraphBatchEdit<Node = i32, Edge = i32>
        + NodeIds
        + EdgeIds
        + NodeAccess<Node = i32>
        + EdgeAccess<Edge = i32>
        + GraphSnapshot
        + Clone,
    G: graph_query::NodeSearch<fn(&i32) -> bool> + graph_query::EdgeSearch<fn(&i32) -> bool>,
    <G as GraphInsert>::Error: std::fmt::Debug,
    <G as graph_query::NodeSearch<fn(&i32) -> bool>>::Error: std::fmt::Debug,
    <G as graph_query::EdgeSearch<fn(&i32) -> bool>>::Error: std::fmt::Debug,
    G::NodeId: std::fmt::Debug,
{
    use graph_query::{EdgeSearch, NodeSearch};
    let a = graph.insert_node(10).unwrap();
    let b = graph.insert_node(20).unwrap();
    graph.insert_edge(a, b, 30).unwrap();
    let query: fn(&i32) -> bool = |weight| *weight >= 20;
    assert_eq!(
        NodeSearch::search_nodes(&graph, &query)
            .unwrap()
            .collect::<Vec<_>>(),
        vec![b]
    );
    assert_eq!(EdgeSearch::search_edges(&graph, &query).unwrap().count(), 1);
    assert!(!G::DATA_STABLE);
    assert!(graph.snapshot().is_ok());
    let version = graph.version();
    let edits = vec![
        TopologyEdit::RemoveNode { node: a },
        TopologyEdit::InsertEdge {
            source: a,
            target: b,
            weight: 1,
        },
    ];
    let failure = graph
        .edit_batch_atomic(edits)
        .err()
        .expect("invalid atomic batch");
    assert_eq!(failure.failed_at, Some(1));
    assert!(version == graph.version());
    assert_eq!(graph.node_ids().count(), 2);
    assert_eq!(graph.edge_ids().count(), 1);
    let edits = vec![
        TopologyEdit::RemoveNode { node: a },
        TopologyEdit::InsertEdge {
            source: a,
            target: b,
            weight: 1,
        },
    ];
    let failure = graph
        .edit_batch(edits)
        .err()
        .expect("invalid ordinary batch");
    assert_eq!(failure.failed_at, 1);
    assert_eq!(failure.completed.len(), 1);
    assert!(version != graph.version());
    assert_eq!(graph.node_ids().count(), 1);
    assert_eq!(graph.edge_ids().count(), 0);
    let version = graph.version();
    assert!(graph
        .edit_batch_atomic(vec![])
        .unwrap_or_else(|_| panic!("empty batch"))
        .is_empty());
    assert!(version == graph.version());
    let result = graph
        .edit_batch_atomic(vec![TopologyEdit::InsertNode { weight: 99 }])
        .unwrap_or_else(|_| panic!("valid atomic batch"));
    assert_eq!(result.len(), 1);
    assert_eq!(graph.node_ids().count(), 2);
}

#[test]
fn sparse_batches_queries_and_snapshots() {
    batch_and_query_contract(SparseGraph::new());
}

#[test]
fn dense_batches_queries_and_snapshots() {
    batch_and_query_contract(DenseGraph::new());
}

#[test]
fn dense_batch_tracks_moved_nodes_and_edges_by_original_identity() {
    let mut graph = DenseGraph::<i32, i32>::new();
    let nodes: Vec<_> = (0..4).map(|w| graph.insert_node(w).unwrap()).collect();
    let removed_edge = graph.insert_edge(nodes[0], nodes[1], 1).unwrap();
    let moved_edge = graph.insert_edge(nodes[3], nodes[2], 2).unwrap();
    let result = graph
        .edit_batch(vec![
            TopologyEdit::RemoveNode { node: nodes[1] },
            TopologyEdit::InsertEdge {
                source: nodes[3],
                target: nodes[2],
                weight: 3,
            },
            TopologyEdit::RemoveEdge { edge: moved_edge },
            TopologyEdit::RemoveEdge { edge: removed_edge },
            TopologyEdit::RemoveNode { node: nodes[1] },
        ])
        .unwrap_or_else(|_| panic!("valid original identities"));
    assert!(matches!(
        result[2],
        TopologyEditResult::EdgeRemoved { removed: true }
    ));
    assert!(matches!(
        result[3],
        TopologyEditResult::EdgeRemoved { removed: false }
    ));
    assert!(matches!(
        result[4],
        TopologyEditResult::NodeRemoved { removed: false }
    ));
    assert_eq!(graph.edge_count(), 1);
    let edge = graph.edge_ids().next().unwrap();
    let reference = graph.edge(edge).unwrap();
    assert_eq!(*reference.weight(), 3);
    let (source, target) = graph.endpoints(edge).unwrap();
    assert_eq!(*graph.node(source).unwrap().weight(), 3);
    assert_eq!(*graph.node(target).unwrap().weight(), 2);
}

#[test]
fn sparse_clone_has_independent_deque_links_and_preserves_neighbor_order() {
    let mut graph = SparseGraph::<i32, i32>::new();
    let a = graph.insert_node(1).unwrap();
    let b = graph.insert_node(2).unwrap();
    let first = graph.insert_edge(a, b, 10).unwrap();
    graph.insert_edge(a, a, 11).unwrap();
    graph.insert_edge(a, b, 12).unwrap();
    let mut copy = graph.clone();
    copy.remove_edge(first).unwrap();
    assert_eq!(graph.out_neighbors(a).collect::<Vec<_>>(), vec![b, a, b]);
    assert_eq!(copy.out_neighbors(a).collect::<Vec<_>>(), vec![a, b]);
    copy.remove_node(a).unwrap();
    assert_eq!(copy.edge_count(), 0);
    assert_eq!(graph.edge_count(), 3);
}
