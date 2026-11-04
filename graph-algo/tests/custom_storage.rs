use graph_adaptor::{EdgeFiltered, Reversed};
use graph_algo::{bfs, dfs, toposort};
use graph_core::{
    EdgeAccess, EdgeEndpoints, EdgeIds, EdgeRef, EdgeReference, GraphBase, InEdges, InNeighbors,
    NodeAccess, NodeIds, NodeRef, NodeReference, OutEdges, OutNeighbors,
};
use graph_query::{find_node, reachable, Bfs, Walker};
use std::cell::{Ref, RefCell};

#[test]
fn reference_wrappers_support_mutable_and_unsized_weights() {
    use graph_core::{EdgeRefMut, NodeRefMut};
    let mut value = 7;
    let mut node = NodeReference::new("node", &mut value);
    *node.weight_mut() = 9;
    assert_eq!(node.id(), "node");
    assert_eq!(*node.weight(), 9);
    let mut value = 3;
    let mut edge = EdgeReference::new(42, "a", "b", &mut value);
    *edge.weight_mut() = 5;
    assert_eq!((edge.id(), edge.source(), edge.target()), (42, "a", "b"));
    assert_eq!(*edge.weight(), 5);
    let node = NodeReference::new(1, "unsized");
    assert_eq!(node.weight(), "unsized");
}

// 公共 seam：索引访问集合、BFS/DFS 索引构造和索引可达性查询。
#[test]
fn indexed_queries_support_sparse_slots_cycles_and_parallel_edges() {
    use graph_core::{IndexedVisitMap, VisitMap};
    use graph_query::{reachable_indexed, Dfs};
    let graph = Storage::new(vec![
        ("alpha", "alpha", 0),
        ("alpha", "omega", 1),
        ("alpha", "omega", 2),
        ("alpha", "leaf", 3),
        ("leaf", "omega", 4),
        ("omega", "alpha", 5),
    ]);
    let mut visited = IndexedVisitMap::new(&graph);
    assert!(!visited.is_visited("leaf"));
    assert!(visited.visit("leaf"));
    assert!(!visited.visit("leaf"));
    assert!(visited.is_visited("leaf"));
    assert!(!IndexedVisitMap::new(&graph).is_visited("leaf"));
    assert_eq!(
        Bfs::new_indexed(&graph, "alpha")
            .iter(&graph)
            .collect::<Vec<_>>(),
        vec!["alpha", "omega", "leaf"]
    );
    assert_eq!(
        Dfs::new_indexed(&graph, "alpha")
            .iter(&graph)
            .collect::<Vec<_>>(),
        vec!["alpha", "leaf", "omega"]
    );
    assert!(reachable_indexed(&graph, "alpha", "leaf"));
    assert!(!reachable_indexed(&graph, "alpha", "isolated"));
    assert!(reachable_indexed(&graph, "isolated", "isolated"));
}

impl graph_core::NodeIndexable for Storage {
    fn node_bound(&self) -> usize {
        200
    }
    fn to_index(&self, node: Self::NodeId) -> usize {
        match node {
            "alpha" => 2,
            "omega" => 65,
            "leaf" => 130,
            "isolated" => 199,
            _ => panic!("invalid node"),
        }
    }
    fn node_at_index(&self, index: usize) -> Option<Self::NodeId> {
        match index {
            2 => Some("alpha"),
            65 => Some("omega"),
            130 => Some("leaf"),
            199 => Some("isolated"),
            _ => None,
        }
    }
}

// 测试专用存储：稀疏字符串 ID、自环、平行边以及 RefCell Guard。
// 不属于库提供的生产存储后端。
struct Storage {
    nodes: Vec<(&'static str, RefCell<i32>)>,
    edges: Vec<(&'static str, &'static str, i32)>,
}
impl Storage {
    fn new(edges: Vec<(&'static str, &'static str, i32)>) -> Self {
        Self {
            nodes: vec![
                ("alpha", RefCell::new(10)),
                ("omega", RefCell::new(20)),
                ("leaf", RefCell::new(30)),
                ("isolated", RefCell::new(40)),
            ],
            edges,
        }
    }
}
impl GraphBase for Storage {
    type NodeId = &'static str;
    type EdgeId = usize;
}
impl NodeIds for Storage {
    type NodeIds<'a> = std::vec::IntoIter<&'static str>;
    fn node_ids(&self) -> Self::NodeIds<'_> {
        self.nodes
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>()
            .into_iter()
    }
}
impl NodeAccess for Storage {
    type Node = i32;
    type NodeRef<'a> = NodeReference<Self::NodeId, Ref<'a, i32>>;
    fn node(&self, id: Self::NodeId) -> Option<Self::NodeRef<'_>> {
        self.nodes
            .iter()
            .find(|(key, _)| *key == id)
            .map(|(_, value)| NodeReference::new(id, value.borrow()))
    }
}
impl EdgeAccess for Storage {
    type Edge = i32;
    type EdgeRef<'a> = EdgeReference<Self::NodeId, usize, &'a i32>;
    fn edge(&self, id: usize) -> Option<Self::EdgeRef<'_>> {
        self.edges
            .get(id)
            .map(|edge| EdgeReference::new(id, edge.0, edge.1, &edge.2))
    }
}
impl EdgeEndpoints for Storage {
    fn endpoints(&self, id: usize) -> Option<(Self::NodeId, Self::NodeId)> {
        self.edges.get(id).map(|edge| (edge.0, edge.1))
    }
}
impl EdgeIds for Storage {
    type EdgeIds<'a> = std::ops::Range<usize>;
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        0..self.edges.len()
    }
}
impl OutEdges for Storage {
    type OutEdges<'a> = std::vec::IntoIter<usize>;
    fn out_edges(&self, node: Self::NodeId) -> Self::OutEdges<'_> {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.0 == node)
            .map(|(id, _)| id)
            .collect::<Vec<_>>()
            .into_iter()
    }
}
impl InEdges for Storage {
    type InEdges<'a> = std::vec::IntoIter<usize>;
    fn in_edges(&self, node: Self::NodeId) -> Self::InEdges<'_> {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.1 == node)
            .map(|(id, _)| id)
            .collect::<Vec<_>>()
            .into_iter()
    }
}
impl OutNeighbors for Storage {
    type OutNeighbors<'a> = std::vec::IntoIter<Self::NodeId>;
    fn out_neighbors(&self, node: Self::NodeId) -> Self::OutNeighbors<'_> {
        self.edges
            .iter()
            .filter(|edge| edge.0 == node)
            .map(|edge| edge.1)
            .collect::<Vec<_>>()
            .into_iter()
    }
}
impl InNeighbors for Storage {
    type InNeighbors<'a> = std::vec::IntoIter<Self::NodeId>;
    fn in_neighbors(&self, node: Self::NodeId) -> Self::InNeighbors<'_> {
        self.edges
            .iter()
            .filter(|edge| edge.1 == node)
            .map(|edge| edge.0)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

#[test]
fn sparse_ids_parallel_edges_and_isolated_nodes() {
    let graph = Storage::new(vec![
        ("alpha", "omega", 1),
        ("alpha", "omega", 2),
        ("omega", "leaf", 3),
    ]);
    let order = toposort(&graph).unwrap();
    assert_eq!(order.len(), 4);
    let pos = |id| order.iter().position(|node| *node == id).unwrap();
    assert!(pos("alpha") < pos("omega"));
    assert!(pos("omega") < pos("leaf"));
    assert_eq!(bfs(&graph, "alpha"), vec!["alpha", "omega", "leaf"]);
    assert_eq!(dfs(&graph, "alpha"), vec!["alpha", "omega", "leaf"]);
    assert!(reachable(&graph, "alpha", "leaf"));
    assert!(!reachable(&graph, "leaf", "alpha"));
    assert!(reachable(&graph, "isolated", "isolated"));
}

#[test]
fn cycles_self_loops_and_empty_graph() {
    let graph = Storage::new(vec![
        ("alpha", "omega", 1),
        ("omega", "alpha", 1),
        ("omega", "leaf", 1),
    ]);
    assert_eq!(
        toposort(&graph).unwrap_err().remaining,
        vec!["alpha", "omega", "leaf"]
    );
    assert_eq!(bfs(&graph, "alpha"), vec!["alpha", "omega", "leaf"]);
    assert_eq!(dfs(&graph, "alpha"), vec!["alpha", "omega", "leaf"]);
    let graph = Storage::new(vec![("alpha", "alpha", 1)]);
    assert_eq!(toposort(&graph).unwrap_err().remaining, vec!["alpha"]);
    assert_eq!(bfs(&graph, "alpha"), vec!["alpha"]);
    let graph = Storage {
        nodes: vec![],
        edges: vec![],
    };
    assert_eq!(toposort(&graph), Ok(vec![]));
}

#[test]
fn dfs_descends_before_visiting_a_pending_sibling() {
    let graph = Storage::new(vec![
        ("alpha", "omega", 1),
        ("alpha", "leaf", 1),
        ("leaf", "omega", 1),
        ("leaf", "isolated", 1),
    ]);
    assert_eq!(
        dfs(&graph, "alpha"),
        vec!["alpha", "leaf", "isolated", "omega"]
    );
    assert_eq!(
        bfs(&graph, "alpha"),
        vec!["alpha", "omega", "leaf", "isolated"]
    );
}

#[test]
fn views_are_consistent_and_composable() {
    let graph = Storage::new(vec![
        ("alpha", "omega", 1),
        ("omega", "alpha", 0),
        ("omega", "leaf", 1),
    ]);
    let filtered = EdgeFiltered::new(&graph, |id| {
        graph.edge(id).is_some_and(|edge| *edge.weight() > 0)
    });
    assert!(filtered.edge(usize::MAX).is_none());
    assert_eq!(filtered.endpoints(usize::MAX), None);
    assert_eq!(filtered.node_ids().count(), 4);
    assert_eq!(filtered.edge_ids().collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(filtered.out_edges("omega").collect::<Vec<_>>(), vec![2]);
    assert_eq!(filtered.in_edges("alpha").count(), 0);
    assert_eq!(
        filtered.in_neighbors("leaf").collect::<Vec<_>>(),
        vec!["omega"]
    );
    assert_eq!(filtered.endpoints(1), None);
    assert!(filtered.edge(1).is_none());
    assert_eq!(*filtered.node("alpha").unwrap().weight(), 10);
    assert!(toposort(&filtered).is_ok());
    let reversed = Reversed::new(&filtered);
    assert_eq!(bfs(&reversed, "leaf"), vec!["leaf", "omega", "alpha"]);
    assert_eq!(reversed.endpoints(0), Some(("omega", "alpha")));
    let edge = reversed.edge(0).unwrap();
    assert_eq!(edge.id(), 0);
    assert_eq!(
        (edge.source(), edge.target()),
        reversed.endpoints(0).unwrap()
    );
    let restored = Reversed::new(&reversed);
    let edge = restored.edge(0).unwrap();
    assert_eq!((edge.source(), edge.target()), graph.endpoints(0).unwrap());
    assert_eq!(reversed.out_edges("leaf").collect::<Vec<_>>(), vec![2]);
    assert_eq!(
        reversed.in_neighbors("omega").collect::<Vec<_>>(),
        vec!["leaf"]
    );
    assert_eq!(reversed.edge_ids().count(), 2);
    assert_eq!(*reversed.edge(2).unwrap().weight(), 1);
    assert_eq!(*reversed.node("leaf").unwrap().weight(), 30);
    let none = EdgeFiltered::new(&graph, |_| false);
    assert_eq!(toposort(&none).unwrap().len(), 4);
    assert_eq!(bfs(&none, "alpha"), vec!["alpha"]);
}

#[test]
fn walker_releases_graph_borrow_and_access_supports_guards() {
    let mut graph = Storage::new(vec![("alpha", "omega", 1)]);
    assert_eq!(find_node(&graph, |value| *value == 20), Some("omega"));
    {
        let node = graph.node("omega").unwrap();
        assert_eq!(node.id(), "omega");
        assert_eq!(*node.weight(), 20);
        assert!(graph.nodes[1].1.try_borrow_mut().is_err());
    }
    assert!(graph.nodes[1].1.try_borrow_mut().is_ok());
    let mut walker = Bfs::new(&graph, "alpha");
    assert_eq!(walker.next(&graph), Some("alpha"));
    // Walker 本身不持有图借用，可以在 next 之间独占访问并修改业务数据。
    *graph.nodes[1].1.get_mut() = 99;
    assert_eq!(find_node(&graph, |value| *value == 99), Some("omega"));
    assert_eq!(walker.next(&graph), Some("omega"));
    assert_eq!(walker.next(&graph), None);
    assert_eq!(walker.next(&graph), None);
    assert_eq!(
        Bfs::new(&graph, "alpha").iter(&graph).collect::<Vec<_>>(),
        vec!["alpha", "omega"]
    );
}
// 公共 seam：工作区排序与便捷排序；工作区可以跨图、环失败及空图复用。
#[test]
fn topo_workspace_reinitializes_after_cycles_and_graph_changes() {
    use graph_algo::topo::{toposort_with_workspace, TopoWorkspace};
    let mut workspace = TopoWorkspace::new();
    let dag = Storage::new(vec![
        ("alpha", "omega", 1),
        ("alpha", "omega", 2),
        ("omega", "leaf", 3),
    ]);
    assert_eq!(
        toposort_with_workspace(&dag, &mut workspace),
        Ok(vec!["alpha", "isolated", "omega", "leaf"])
    );
    assert_eq!(
        toposort_with_workspace(&dag, &mut workspace),
        toposort(&dag)
    );
    let cycle = Storage::new(vec![("alpha", "alpha", 0), ("alpha", "leaf", 0)]);
    assert_eq!(
        toposort_with_workspace(&cycle, &mut workspace)
            .unwrap_err()
            .remaining,
        vec!["alpha", "leaf"]
    );
    let empty = Storage {
        nodes: vec![],
        edges: vec![],
    };
    assert_eq!(toposort_with_workspace(&empty, &mut workspace), Ok(vec![]));
    let small = Storage {
        nodes: vec![("leaf", RefCell::new(1))],
        edges: vec![],
    };
    assert_eq!(
        toposort_with_workspace(&small, &mut workspace),
        Ok(vec!["leaf"])
    );
    assert_eq!(
        toposort_with_workspace(&dag, &mut workspace),
        Ok(vec!["alpha", "isolated", "omega", "leaf"])
    );
}
#[test]
fn mapped_and_translated_guards_release_on_success_and_panic() {
    use graph_adaptor::{IdMapping, MapIds, MapNodes};
    struct Names;
    impl IdMapping<&'static str> for Names {
        type External = &'static str;
        fn to_external(&self, id: &'static str) -> Option<&'static str> {
            Some(id)
        }
        fn to_internal(&self, id: &'static str) -> Option<&'static str> {
            Some(id)
        }
    }
    struct Edges;
    impl IdMapping<usize> for Edges {
        type External = usize;
        fn to_external(&self, id: usize) -> Option<usize> {
            Some(id)
        }
        fn to_internal(&self, id: usize) -> Option<usize> {
            Some(id)
        }
    }
    let graph = Storage::new(vec![]);
    let mapped = MapNodes::new(&graph, |v: &i32| v.to_string());
    let node = mapped.node("alpha").unwrap();
    assert_eq!(node.weight(), "10");
    assert!(graph.nodes[0].1.try_borrow_mut().is_ok());
    let panic_map = MapNodes::new(&graph, |_: &i32| -> String { panic!("mapper") });
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| panic_map.node("alpha"))).is_err()
    );
    assert!(graph.nodes[0].1.try_borrow_mut().is_ok());
    let translated = MapIds::new(&graph, Names, Edges);
    let node = translated.node("alpha").unwrap();
    assert_eq!(node.id(), "alpha");
    assert!(graph.nodes[0].1.try_borrow_mut().is_err());
    drop(node);
    assert!(graph.nodes[0].1.try_borrow_mut().is_ok());
}
