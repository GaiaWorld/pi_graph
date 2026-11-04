use graph_adaptor::{EdgeFiltered, Reversed};
use graph_core::{DenseNodeIndex, GraphBase, NodeIndexable, VisitMap, Visitable};
use graph_core::{EdgeAccess, EdgeRef, NodeAccess, NodeRef};
use graph_core::{EdgeEndpoints, EdgeIds, InEdges, InNeighbors, NodeIds, OutEdges, OutNeighbors};
use std::collections::HashSet;

struct Slots;
// 公共 seam：最小邻居能力、未知起点短路与映射完整性。
struct NeighborOnly;
impl GraphBase for NeighborOnly {
    type NodeId = u32;
    type EdgeId = u32;
}
impl OutNeighbors for NeighborOnly {
    type OutNeighbors<'a> = std::array::IntoIter<u32, 4>;
    fn out_neighbors(&self, node: u32) -> Self::OutNeighbors<'_> {
        assert_eq!(node, 10, "hidden or unmapped start must not reach backend");
        [20, 30, 10, 20].into_iter()
    }
}
impl InNeighbors for NeighborOnly {
    type InNeighbors<'a> = std::array::IntoIter<u32, 4>;
    fn in_neighbors(&self, node: u32) -> Self::InNeighbors<'_> {
        assert_eq!(node, 10, "hidden or unmapped start must not reach backend");
        [30, 20, 10, 20].into_iter()
    }
}
#[test]
fn induced_neighbors_need_no_edge_capabilities_and_short_circuit_hidden_starts() {
    let graph = graph_adaptor::Subgraph::new(NeighborOnly, |id| id != 30);
    assert_eq!(
        graph.out_neighbors(10).collect::<Vec<_>>(),
        vec![20, 10, 20]
    );
    assert_eq!(graph.in_neighbors(10).collect::<Vec<_>>(), vec![20, 10, 20]);
    assert_eq!(graph.out_neighbors(30).count(), 0);
    assert_eq!(graph.in_neighbors(30).count(), 0);
}
// 公共 seam：六类翻译迭代器的 Iterator::next；起点映射始终合法。
#[derive(Clone, Copy)]
enum BrokenCandidateKind {
    Collision,
    MissingReverse,
    WrongReverse,
}
struct BrokenCandidate {
    id: u32,
    kind: BrokenCandidateKind,
}
impl graph_adaptor::IdMapping<u32> for BrokenCandidate {
    type External = u32;
    fn to_external(&self, id: u32) -> Option<u32> {
        if id == self.id && matches!(self.kind, BrokenCandidateKind::Collision) {
            Some(if id == 10 {
                1020
            } else if id == 20 {
                1010
            } else {
                1007
            })
        } else {
            Some(id + 1000)
        }
    }
    fn to_internal(&self, external: u32) -> Option<u32> {
        if external == self.id + 1000 {
            match self.kind {
                BrokenCandidateKind::MissingReverse => None,
                BrokenCandidateKind::WrongReverse => Some(999),
                BrokenCandidateKind::Collision => external.checked_sub(1000),
            }
        } else {
            external.checked_sub(1000)
        }
    }
}
const BROKEN_CANDIDATE_KINDS: [BrokenCandidateKind; 3] = [
    BrokenCandidateKind::Collision,
    BrokenCandidateKind::MissingReverse,
    BrokenCandidateKind::WrongReverse,
];
fn assert_candidate_panics(mut iter: impl Iterator<Item = u32>) {
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| iter.next()))
        .expect_err("invalid candidate must panic before yielding an external identity");
    let message = panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some("identity mapping must round trip"));
}
#[test]
fn translated_node_ids_reject_non_round_tripping_candidates() {
    for kind in BROKEN_CANDIDATE_KINDS {
        let graph = graph_adaptor::MapIds::new(Slots, BrokenCandidate { id: 20, kind }, Offset);
        let mut iter = graph.node_ids();
        assert_eq!(iter.next(), Some(1010));
        assert_candidate_panics(iter);
    }
}
#[test]
fn translated_edge_ids_reject_non_round_tripping_candidates() {
    for kind in BROKEN_CANDIDATE_KINDS {
        let graph = graph_adaptor::MapIds::new(Slots, Offset, BrokenCandidate { id: 8, kind });
        let mut iter = graph.edge_ids();
        assert_eq!(iter.next(), Some(1007));
        assert_candidate_panics(iter);
    }
}
#[test]
fn translated_out_neighbors_reject_non_round_tripping_candidates() {
    for kind in BROKEN_CANDIDATE_KINDS {
        let graph = graph_adaptor::MapIds::new(Slots, BrokenCandidate { id: 20, kind }, Offset);
        assert_candidate_panics(graph.out_neighbors(1010));
    }
}
#[test]
fn translated_in_neighbors_reject_non_round_tripping_candidates() {
    for kind in BROKEN_CANDIDATE_KINDS {
        let graph = graph_adaptor::MapIds::new(Slots, BrokenCandidate { id: 10, kind }, Offset);
        assert_candidate_panics(graph.in_neighbors(1020));
    }
}
#[test]
fn translated_out_edges_reject_non_round_tripping_candidates() {
    for kind in BROKEN_CANDIDATE_KINDS {
        let graph = graph_adaptor::MapIds::new(Slots, Offset, BrokenCandidate { id: 8, kind });
        let mut iter = graph.out_edges(1010);
        assert_eq!(iter.next(), Some(1007));
        assert_candidate_panics(iter);
    }
}
#[test]
fn translated_in_edges_reject_non_round_tripping_candidates() {
    for kind in BROKEN_CANDIDATE_KINDS {
        let graph = graph_adaptor::MapIds::new(Slots, Offset, BrokenCandidate { id: 8, kind });
        let mut iter = graph.in_edges(1020);
        assert_eq!(iter.next(), Some(1007));
        assert_candidate_panics(iter);
    }
}

struct PartialMapping;
impl graph_adaptor::IdMapping<u32> for PartialMapping {
    type External = u32;
    fn to_external(&self, id: u32) -> Option<u32> {
        (id != 30).then_some(id + 1000)
    }
    fn to_internal(&self, id: u32) -> Option<u32> {
        id.checked_sub(1000)
    }
}
#[test]
fn translated_neighbors_short_circuit_unknown_starts_and_panic_on_missing_candidates() {
    let graph = graph_adaptor::MapIds::new(NeighborOnly, PartialMapping, Offset);
    assert_eq!(graph.out_neighbors(999).count(), 0);
    assert_eq!(graph.in_neighbors(999).count(), 0);
    let mut outgoing = graph.out_neighbors(1010);
    assert_eq!(outgoing.next(), Some(1020));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| outgoing.next())).is_err());
    let mut incoming = graph.in_neighbors(1010);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| incoming.next())).is_err());
}
#[test]
fn translated_adjacency_rejects_non_round_tripping_starts() {
    struct Broken;
    impl graph_adaptor::IdMapping<u32> for Broken {
        type External = u32;
        fn to_external(&self, _: u32) -> Option<u32> {
            Some(9999)
        }
        fn to_internal(&self, _: u32) -> Option<u32> {
            Some(10)
        }
    }
    let graph = graph_adaptor::MapIds::new(Slots, Broken, Offset);
    assert!(std::panic::catch_unwind(|| graph.out_neighbors(1010)).is_err());
    assert!(std::panic::catch_unwind(|| graph.in_neighbors(1020)).is_err());
    assert!(std::panic::catch_unwind(|| graph.out_edges(1010)).is_err());
    assert!(std::panic::catch_unwind(|| graph.in_edges(1020)).is_err());
}
#[test]
fn translated_edges_use_the_edge_mapping_in_both_directions() {
    struct EdgeOffset;
    impl graph_adaptor::IdMapping<u32> for EdgeOffset {
        type External = u32;
        fn to_external(&self, id: u32) -> Option<u32> {
            id.checked_add(2000)
        }
        fn to_internal(&self, id: u32) -> Option<u32> {
            id.checked_sub(2000)
        }
    }
    let graph = graph_adaptor::MapIds::new(Slots, Offset, EdgeOffset);
    assert_eq!(graph.edge_ids().collect::<Vec<_>>(), vec![2007, 2008]);
    assert_eq!(graph.out_edges(1010).collect::<Vec<_>>(), vec![2007, 2008]);
    assert_eq!(graph.in_edges(1020).collect::<Vec<_>>(), vec![2007, 2008]);
    let reversed = Reversed::new(graph);
    assert_eq!(
        reversed.out_neighbors(1020).collect::<Vec<_>>(),
        vec![1010, 1010]
    );
    assert_eq!(
        reversed.out_edges(1020).collect::<Vec<_>>(),
        vec![2007, 2008]
    );
}
#[test]
fn t109_subgraph_enumeration() {
    let graph = graph_adaptor::Subgraph::new(Slots, |id| id == 10 || id == 20);
    assert_eq!(graph.in_neighbors(20).collect::<Vec<_>>(), vec![10, 10]);
    assert_eq!(graph.in_neighbors(99).count(), 0);
    let hidden = graph_adaptor::Subgraph::new(Slots, |id| id != 20);
    assert_eq!(hidden.in_neighbors(20).count(), 0);
    let hidden_neighbor = graph_adaptor::Subgraph::new(Slots, |id| id == 20);
    assert_eq!(hidden_neighbor.in_neighbors(20).count(), 0);
}
#[test]
fn t107_subgraph_enumeration() {
    let graph = graph_adaptor::Subgraph::new(Slots, |id| id == 10 || id == 20);
    assert_eq!(graph.out_neighbors(10).collect::<Vec<_>>(), vec![20, 20]);
    assert_eq!(graph.out_neighbors(99).count(), 0);
    let hidden = graph_adaptor::Subgraph::new(Slots, |id| id != 10);
    assert_eq!(hidden.out_neighbors(10).count(), 0);
    let hidden_neighbor = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    assert_eq!(hidden_neighbor.out_neighbors(10).count(), 0);
}
#[test]
fn t100_subgraph_enumeration() {
    let graph = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    assert_eq!(graph.node_ids().collect::<Vec<_>>(), vec![10]);
    let empty = graph_adaptor::Subgraph::new(Slots, |_: u32| false);
    assert_eq!(empty.node_ids().count(), 0);
}
#[test]
fn t40_translated_enumeration() {
    let graph = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(graph.in_edges(1020).collect::<Vec<_>>(), vec![1007, 1008]);
    assert_eq!(graph.in_edges(999).count(), 0);
    assert_eq!(graph.in_edges(1099).count(), 0);
}
#[test]
fn t39_translated_enumeration() {
    let graph = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(graph.out_edges(1010).collect::<Vec<_>>(), vec![1007, 1008]);
    assert_eq!(graph.out_edges(999).count(), 0);
    assert_eq!(graph.out_edges(1099).count(), 0);
}
#[test]
fn t38_translated_enumeration() {
    let graph = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(
        graph.in_neighbors(1020).collect::<Vec<_>>(),
        vec![1010, 1010]
    );
    assert_eq!(graph.in_neighbors(999).count(), 0);
    assert_eq!(graph.in_neighbors(1099).count(), 0);
}
#[test]
fn t37_translated_enumeration() {
    let graph = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(
        graph.out_neighbors(1010).collect::<Vec<_>>(),
        vec![1020, 1020]
    );
    assert_eq!(graph.out_neighbors(999).count(), 0);
    assert_eq!(graph.out_neighbors(1099).count(), 0);
}
#[test]
fn t36_translated_enumeration() {
    let graph = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(graph.edge_ids().collect::<Vec<_>>(), vec![1007, 1008]);
}
#[test]
fn t35_translated_enumeration() {
    let graph = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(graph.node_ids().collect::<Vec<_>>(), vec![1010, 1020]);
}
impl GraphBase for Slots {
    type NodeId = u32;
    type EdgeId = u32;
}
impl NodeIndexable for Slots {
    fn node_bound(&self) -> usize {
        100
    }
    fn to_index(&self, node: u32) -> usize {
        match node {
            10 => 2,
            20 => 99,
            _ => panic!("invalid node"),
        }
    }
    fn node_at_index(&self, index: usize) -> Option<u32> {
        match index {
            2 => Some(10),
            99 => Some(20),
            _ => None,
        }
    }
}
impl DenseNodeIndex for Slots {
    fn node_count(&self) -> usize {
        2
    }
    fn to_dense(&self, node: u32) -> Option<usize> {
        match node {
            10 => Some(0),
            20 => Some(1),
            _ => None,
        }
    }
    fn node_at_dense(&self, index: usize) -> Option<u32> {
        [10, 20].get(index).copied()
    }
}
impl Visitable for Slots {
    type Map = HashSet<u32>;
    fn visit_map(&self) -> Self::Map {
        HashSet::new()
    }
}

// 公共 seam：视图的原始索引、稠密索引与独立访问集合。
#[test]
fn t111() {
    let g = Reversed::new(Slots);
    assert_eq!(g.node_bound(), 100);
}
#[test]
fn t112() {
    let g = Reversed::new(Slots);
    assert_eq!(g.to_index(20), 99);
}
#[test]
fn t113() {
    let g = Reversed::new(Slots);
    assert_eq!(g.node_at_index(2), Some(10));
    assert_eq!(g.node_at_index(0), None);
    assert_eq!(g.node_at_index(100), None);
}
#[test]
fn t114() {
    let g = Reversed::new(Slots);
    let mut a = g.visit_map();
    assert!(!a.is_visited(10));
    assert!(a.visit(10));
    assert!(!a.visit(10));
    assert!(!g.visit_map().is_visited(10));
}
#[test]
fn t115() {
    let g = Reversed::new(Slots);
    assert_eq!(g.node_count(), 2);
}
#[test]
fn t116() {
    let g = Reversed::new(Slots);
    assert_eq!(g.to_dense(20), Some(1));
    assert_eq!(g.to_dense(99), None);
}
#[test]
fn t117() {
    let g = Reversed::new(Slots);
    assert_eq!(g.node_at_dense(0), Some(10));
    assert_eq!(g.node_at_dense(2), None);
}
#[test]
fn t118() {
    let g = EdgeFiltered::new(Slots, |_: u32| false);
    assert_eq!(g.node_bound(), 100);
}
#[test]
fn t119() {
    let g = EdgeFiltered::new(Slots, |_: u32| false);
    assert_eq!(g.to_index(20), 99);
}
#[test]
fn t120() {
    let g = EdgeFiltered::new(Slots, |_: u32| false);
    assert_eq!(g.node_at_index(2), Some(10));
    assert_eq!(g.node_at_index(0), None);
}
#[test]
fn t121() {
    let g = EdgeFiltered::new(Slots, |_: u32| false);
    let mut a = g.visit_map();
    assert!(!a.is_visited(10));
    assert!(a.visit(10));
    assert!(!a.visit(10));
    assert!(!g.visit_map().is_visited(10));
}
#[test]
fn t122() {
    let g = EdgeFiltered::new(Slots, |_: u32| false);
    assert_eq!(g.node_count(), 2);
}
#[test]
fn t123() {
    let g = EdgeFiltered::new(Slots, |_: u32| false);
    assert_eq!(g.to_dense(20), Some(1));
    assert_eq!(g.to_dense(99), None);
}
#[test]
fn t124() {
    let g = EdgeFiltered::new(Slots, |_: u32| false);
    assert_eq!(g.node_at_dense(0), Some(10));
    assert_eq!(g.node_at_dense(2), None);
}
#[test]
fn t97_constructor_does_not_evaluate_selection() {
    use graph_adaptor::Subgraph;
    let _ = Subgraph::new(Slots, |_: u32| -> bool { panic!("must stay lazy") });
}
#[test]
fn t27_constructor() {
    let _ = graph_adaptor::MapIds::new((), (), ());
}
#[test]
fn t87_constructor() {
    let _ = graph_adaptor::MapNodes::new((), |_: &i32| -> i32 { panic!("lazy") });
}
#[test]
fn t90_constructor() {
    let _ = graph_adaptor::MapEdges::new((), |_: &i32| -> i32 { panic!("lazy") });
}
impl graph_core::NodeIds for Slots {
    type NodeIds<'a> = std::array::IntoIter<u32, 2>;
    fn node_ids(&self) -> Self::NodeIds<'_> {
        [10, 20].into_iter()
    }
}
impl graph_core::EdgeIds for Slots {
    type EdgeIds<'a> = std::array::IntoIter<u32, 2>;
    fn edge_ids(&self) -> Self::EdgeIds<'_> {
        [7, 8].into_iter()
    }
}
impl graph_core::OutNeighbors for Slots {
    type OutNeighbors<'a> = std::vec::IntoIter<u32>;
    fn out_neighbors(&self, node: u32) -> Self::OutNeighbors<'_> {
        if node == 10 { vec![20, 20] } else { vec![] }.into_iter()
    }
}
impl graph_core::InNeighbors for Slots {
    type InNeighbors<'a> = std::vec::IntoIter<u32>;
    fn in_neighbors(&self, node: u32) -> Self::InNeighbors<'_> {
        if node == 20 { vec![10, 10] } else { vec![] }.into_iter()
    }
}
impl graph_core::OutEdges for Slots {
    type OutEdges<'a> = std::vec::IntoIter<u32>;
    fn out_edges(&self, node: u32) -> Self::OutEdges<'_> {
        if node == 10 { vec![7, 8] } else { vec![] }.into_iter()
    }
}
impl graph_core::InEdges for Slots {
    type InEdges<'a> = std::vec::IntoIter<u32>;
    fn in_edges(&self, node: u32) -> Self::InEdges<'_> {
        if node == 20 { vec![7, 8] } else { vec![] }.into_iter()
    }
}
impl graph_core::EdgeEndpoints for Slots {
    fn endpoints(&self, edge: u32) -> Option<(u32, u32)> {
        matches!(edge, 7 | 8).then_some((10, 20))
    }
}
#[test]
fn t53() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.node_ids().collect::<Vec<_>>(), vec![10, 20]);
}
#[test]
fn t54() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.edge_ids().collect::<Vec<_>>(), vec![7, 8]);
}
#[test]
fn t55() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.out_neighbors(10).collect::<Vec<_>>(), vec![20, 20]);
}
#[test]
fn t56() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.in_neighbors(20).collect::<Vec<_>>(), vec![10, 10]);
}
#[test]
fn t57() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.out_edges(10).collect::<Vec<_>>(), vec![7, 8]);
}
#[test]
fn t58() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.in_edges(20).collect::<Vec<_>>(), vec![7, 8]);
}
#[test]
fn t59() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.node_ids().collect::<Vec<_>>(), vec![10, 20]);
}
#[test]
fn t60() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.edge_ids().collect::<Vec<_>>(), vec![7, 8]);
}
#[test]
fn t61() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.out_neighbors(10).collect::<Vec<_>>(), vec![20, 20]);
}
#[test]
fn t62() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.in_neighbors(20).collect::<Vec<_>>(), vec![10, 10]);
}
#[test]
fn t63() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.out_edges(10).collect::<Vec<_>>(), vec![7, 8]);
}
#[test]
fn t64() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.in_edges(20).collect::<Vec<_>>(), vec![7, 8]);
}
#[test]
fn t65() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.endpoints(7), Some((10, 20)));
    assert_eq!(g.endpoints(99), None);
}
#[test]
fn t66() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.node_bound(), 100);
}
#[test]
fn t67() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.to_index(20), 99);
}
#[test]
fn t68() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.node_at_index(99), Some(20));
    assert_eq!(g.node_at_index(0), None);
}
#[test]
fn t69() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.node_count(), 2);
}
#[test]
fn t70() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.to_dense(20), Some(1));
    assert_eq!(g.to_dense(99), None);
}
#[test]
fn t71() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(g.node_at_dense(0), Some(10));
    assert_eq!(g.node_at_dense(2), None);
}
#[test]
fn t72() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    let mut a = g.visit_map();
    assert!(a.visit(10));
    assert!(!a.visit(10));
    assert!(!g.visit_map().is_visited(10));
}
#[test]
fn t73() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.endpoints(7), Some((10, 20)));
    assert_eq!(g.endpoints(99), None);
}
#[test]
fn t74() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.node_bound(), 100);
}
#[test]
fn t75() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.to_index(20), 99);
}
#[test]
fn t76() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.node_at_index(99), Some(20));
    assert_eq!(g.node_at_index(0), None);
}
#[test]
fn t77() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.node_count(), 2);
}
#[test]
fn t78() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.to_dense(20), Some(1));
    assert_eq!(g.to_dense(99), None);
}
#[test]
fn t79() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(g.node_at_dense(0), Some(10));
    assert_eq!(g.node_at_dense(2), None);
}
#[test]
fn t80() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    let mut a = g.visit_map();
    assert!(a.visit(10));
    assert!(!a.visit(10));
    assert!(!g.visit_map().is_visited(10));
}
#[test]
fn t93() {
    let g = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    assert_eq!(g.node_bound(), 100);
}
#[test]
fn t94() {
    let g = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    assert_eq!(g.to_index(10), 2);
}
#[test]
fn t95() {
    let g = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    assert_eq!(g.node_at_index(2), Some(10));
    assert_eq!(g.node_at_index(99), None);
    assert_eq!(g.node_at_index(0), None);
}
#[test]
fn t96() {
    let g = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    let mut a = g.visit_map();
    assert!(a.visit(10));
    assert!(!g.visit_map().is_visited(10));
}
#[test]
fn t10_build_validates_duplicates_and_missing_endpoints() {
    use graph_adaptor::{AllEdges, AllNodes, IndexedSubgraph, IndexedSubgraphError};
    struct Invalid {
        nodes: Vec<u32>,
        edges: Vec<u32>,
        endpoints: Option<(u32, u32)>,
    }
    impl GraphBase for Invalid {
        type NodeId = u32;
        type EdgeId = u32;
    }
    impl NodeIds for Invalid {
        type NodeIds<'a> = std::iter::Copied<std::slice::Iter<'a, u32>>;
        fn node_ids(&self) -> Self::NodeIds<'_> {
            self.nodes.iter().copied()
        }
    }
    impl EdgeIds for Invalid {
        type EdgeIds<'a> = std::iter::Copied<std::slice::Iter<'a, u32>>;
        fn edge_ids(&self) -> Self::EdgeIds<'_> {
            self.edges.iter().copied()
        }
    }
    impl EdgeEndpoints for Invalid {
        fn endpoints(&self, _: u32) -> Option<(u32, u32)> {
            self.endpoints
        }
    }
    assert!(matches!(
        IndexedSubgraph::build(
            Invalid {
                nodes: vec![10, 10],
                edges: vec![],
                endpoints: None
            },
            AllNodes,
            AllEdges
        ),
        Err(IndexedSubgraphError::DuplicateNode { node: 10 })
    ));
    assert!(matches!(
        IndexedSubgraph::build(
            Invalid {
                nodes: vec![10],
                edges: vec![7, 7],
                endpoints: Some((10, 10))
            },
            AllNodes,
            AllEdges
        ),
        Err(IndexedSubgraphError::DuplicateEdge { edge: 7 })
    ));
    assert!(matches!(
        IndexedSubgraph::build(
            Invalid {
                nodes: vec![10],
                edges: vec![7],
                endpoints: None
            },
            AllNodes,
            AllEdges
        ),
        Err(IndexedSubgraphError::MissingEndpoints { edge: 7 })
    ));
    assert!(matches!(
        IndexedSubgraph::build(
            Invalid {
                nodes: vec![10],
                edges: vec![7],
                endpoints: Some((10, 20))
            },
            AllNodes,
            AllEdges
        ),
        Err(IndexedSubgraphError::UnknownEndpoint { edge: 7, node: 20 })
    ));
    let nodes = std::cell::Cell::new(0);
    let edges = std::cell::Cell::new(0);
    let _ = IndexedSubgraph::build(
        Slots,
        graph_adaptor::Predicate {
            predicate: |_: u32| {
                nodes.set(nodes.get() + 1);
                false
            },
        },
        graph_adaptor::Predicate {
            predicate: |_: u32| {
                edges.set(edges.get() + 1);
                false
            },
        },
    )
    .unwrap();
    assert_eq!((nodes.get(), edges.get()), (2, 2));
}
#[test]
fn t11() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.node_ids().collect::<Vec<_>>(), vec![10, 20]);
}
#[test]
fn t12() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.edge_ids().collect::<Vec<_>>(), vec![7, 8]);
}
#[test]
fn t13() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.out_neighbors(10).collect::<Vec<_>>(), vec![20, 20]);
    assert_eq!(g.out_neighbors(99).count(), 0);
}
#[test]
fn t14() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.in_neighbors(20).collect::<Vec<_>>(), vec![10, 10]);
    assert_eq!(g.in_neighbors(99).count(), 0);
}
#[test]
fn t15() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.out_edges(10).collect::<Vec<_>>(), vec![7, 8]);
    assert_eq!(g.out_edges(99).count(), 0);
}
#[test]
fn t16() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.in_edges(20).collect::<Vec<_>>(), vec![7, 8]);
    assert_eq!(g.in_edges(99).count(), 0);
}
#[test]
fn t19() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.endpoints(7), Some((10, 20)));
    assert_eq!(g.endpoints(99), None);
}
#[test]
fn t20() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.node_bound(), 2);
}
#[test]
fn t21() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.to_index(20), 1);
}
#[test]
fn t22() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.node_at_index(1), Some(20));
    assert_eq!(g.node_at_index(2), None);
}
#[test]
fn t23() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.node_count(), 2);
}
#[test]
fn t24() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.to_dense(20), Some(1));
    assert_eq!(g.to_dense(99), None);
}
#[test]
fn t25() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(g.node_at_dense(1), Some(20));
    assert_eq!(g.node_at_dense(2), None);
}
#[test]
fn t26() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    let mut a = g.visit_map();
    assert!(a.visit(10));
    assert!(!g.visit_map().is_visited(10));
}
#[test]
fn t3() {
    use graph_adaptor::GraphAdaptExt;
    let _ = Slots.subgraph(graph_adaptor::AllNodes);
}
#[test]
fn t4() {
    use graph_adaptor::GraphAdaptExt;
    let _ = Slots.subgraph_with_edges(graph_adaptor::AllNodes, graph_adaptor::AllEdges);
}
#[test]
fn t5() {
    use graph_adaptor::GraphAdaptExt;
    let _ = Slots.filter_nodes(|_: u32| panic!("lazy"));
}
#[test]
fn t6() {
    use graph_adaptor::GraphAdaptExt;
    let _ = Slots.filter_edges(|_: u32| panic!("lazy"));
}
#[test]
fn t7() {
    use graph_adaptor::GraphAdaptExt;
    let g = Slots.reversed();
    assert_eq!(g.out_neighbors(20).collect::<Vec<_>>(), vec![10, 10]);
}
#[test]
fn t2() {
    use graph_adaptor::GraphAdaptExt;
    let g = Slots
        .indexed_subgraph(graph_adaptor::AllNodes, graph_adaptor::AllEdges)
        .unwrap();
    assert_eq!(g.node_ids().collect::<Vec<_>>(), vec![10, 20]);
}
impl graph_core::NodeAccess for Slots {
    type Node = i32;
    type NodeRef<'a> = graph_core::NodeReference<u32, &'a i32>;
    fn node(&self, id: u32) -> Option<Self::NodeRef<'_>> {
        matches!(id, 10 | 20).then(|| graph_core::NodeReference::new(id, &42))
    }
}
impl graph_core::EdgeAccess for Slots {
    type Edge = i32;
    type EdgeRef<'a> = graph_core::EdgeReference<u32, u32, &'a i32>;
    fn edge(&self, id: u32) -> Option<Self::EdgeRef<'_>> {
        matches!(id, 7 | 8).then(|| graph_core::EdgeReference::new(id, 10, 20, &9))
    }
}
#[test]
fn t17() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(*g.node(10).unwrap().weight(), 42);
    assert!(g.node(99).is_none());
}
#[test]
fn t18() {
    let g = graph_adaptor::IndexedSubgraph::build(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::AllEdges,
    )
    .unwrap();
    assert_eq!(*g.edge(7).unwrap().weight(), 9);
    assert!(g.edge(99).is_none());
}
#[test]
fn t104() {
    let g = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    assert_eq!(*g.node(10).unwrap().weight(), 42);
    assert!(g.node(20).is_none());
    assert!(g.node(99).is_none());
}
#[test]
fn t105() {
    let g = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    assert!(g.edge(7).is_none());
    assert!(g.edge(99).is_none());
    let all = graph_adaptor::Subgraph::new(Slots, |_: u32| true);
    assert_eq!(*all.edge(8).unwrap().weight(), 9);
}
#[test]
fn t106() {
    let g = graph_adaptor::Subgraph::new(Slots, |id| id == 10);
    assert_eq!(g.endpoints(7), None);
    assert_eq!(g.endpoints(99), None);
    let all = graph_adaptor::Subgraph::new(Slots, |_: u32| true);
    assert_eq!(all.endpoints(8), Some((10, 20)));
}
#[test]
fn t89() {
    let g = graph_adaptor::MapNodes::new(Slots, ());
    assert_eq!(*g.edge(7).unwrap().weight(), 9);
    assert!(g.edge(99).is_none());
}
#[test]
fn t92() {
    let g = graph_adaptor::MapEdges::new(Slots, ());
    assert_eq!(*g.node(10).unwrap().weight(), 42);
    assert!(g.node(99).is_none());
}
#[test]
fn t88() {
    struct Value(i32);
    let calls = std::cell::Cell::new(0);
    let g = graph_adaptor::MapNodes::new(Slots, |v: &i32| {
        calls.set(calls.get() + 1);
        Value(v + 1)
    });
    assert!(g.node(99).is_none());
    let n = g.node(10).unwrap();
    assert_eq!(n.id(), 10);
    assert_eq!(n.weight().0, 43);
    assert_eq!(g.node(10).unwrap().weight().0, 43);
    assert_eq!(calls.get(), 2);
}
#[test]
fn t91() {
    let g = graph_adaptor::MapEdges::new(Reversed::new(Slots), |v: &i32| v.to_string());
    let e = g.edge(7).unwrap();
    assert_eq!((e.id(), e.source(), e.target()), (7, 20, 10));
    assert_eq!(e.weight(), "9");
    assert!(g.edge(99).is_none());
}
#[test]
fn t41() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    let n = g.node(1010).unwrap();
    assert_eq!(n.id(), 1010);
    assert_eq!(*n.weight(), 42);
    assert!(g.node(999).is_none());
    assert!(g.node(1099).is_none());
}
#[test]
fn t42() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    let e = g.edge(1007).unwrap();
    assert_eq!((e.id(), e.source(), e.target()), (1007, 1010, 1020));
    assert_eq!(*e.weight(), 9);
    assert!(g.edge(999).is_none());
    assert!(g.edge(1099).is_none());
}
#[test]
fn t50() {
    let map = {
        let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
        g.visit_map()
    };
    let mut map = map;
    assert!(!map.is_visited(1010));
    assert!(map.visit(1010));
    assert!(!map.visit(1010));
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert!(!g.visit_map().is_visited(1010));
}
#[test]
fn t101() {
    let g = graph_adaptor::Subgraph::with_edges(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::Predicate {
            predicate: |id| id == 7,
        },
    );
    assert_eq!(g.edge_ids().collect::<Vec<_>>(), vec![7]);
}
#[test]
fn t102() {
    let g = graph_adaptor::Subgraph::with_edges(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::Predicate {
            predicate: |id| id == 7,
        },
    );
    assert_eq!(g.out_edges(10).collect::<Vec<_>>(), vec![7]);
    assert_eq!(g.out_edges(99).count(), 0);
}
#[test]
fn t103() {
    let g = graph_adaptor::Subgraph::with_edges(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::Predicate {
            predicate: |id| id == 7,
        },
    );
    assert_eq!(g.in_edges(20).collect::<Vec<_>>(), vec![7]);
    assert_eq!(g.in_edges(99).count(), 0);
}
#[test]
fn t108() {
    let g = graph_adaptor::Subgraph::with_edges(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::Predicate {
            predicate: |id| id == 7,
        },
    );
    assert_eq!(g.out_neighbors(10).collect::<Vec<_>>(), vec![20]);
    assert_eq!(g.out_neighbors(99).count(), 0);
}
#[test]
fn t110() {
    let g = graph_adaptor::Subgraph::with_edges(
        Slots,
        graph_adaptor::AllNodes,
        graph_adaptor::Predicate {
            predicate: |id| id == 7,
        },
    );
    assert_eq!(g.in_neighbors(20).collect::<Vec<_>>(), vec![10]);
    assert_eq!(g.in_neighbors(99).count(), 0);
}
#[derive(Clone)]
struct Offset;
impl graph_adaptor::IdMapping<u32> for Offset {
    type External = u32;
    fn to_external(&self, id: u32) -> Option<u32> {
        id.checked_add(1000)
    }
    fn to_internal(&self, id: u32) -> Option<u32> {
        id.checked_sub(1000)
    }
}
#[test]
fn t1() {
    use graph_adaptor::GraphAdaptExt;
    let _ = Slots.map_ids(Offset, Offset);
}
#[test]
fn t8() {
    use graph_adaptor::GraphAdaptExt;
    let _ = Slots.map_nodes(|_: &i32| panic!("lazy"));
}
#[test]
fn t9() {
    use graph_adaptor::GraphAdaptExt;
    let _ = Slots.map_edges(|_: &i32| panic!("lazy"));
}
#[test]
fn t44() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(g.node_bound(), 100);
}
#[test]
fn t45() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(g.to_index(1020), 99);
}
#[test]
fn t46() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(g.node_at_index(99), Some(1020));
    assert_eq!(g.node_at_index(0), None);
}
#[test]
fn t47() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(g.node_count(), 2);
}
#[test]
fn t48() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(g.to_dense(1020), Some(1));
    assert_eq!(g.to_dense(999), None);
    assert_eq!(g.to_dense(1099), None);
}
#[test]
fn t49() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(g.node_at_dense(1), Some(1020));
    assert_eq!(g.node_at_dense(2), None);
}
#[test]
fn t43() {
    let g = graph_adaptor::MapIds::new(Slots, Offset, Offset);
    assert_eq!(g.endpoints(1007), Some((1010, 1020)));
    assert_eq!(g.endpoints(999), None);
    assert_eq!(g.endpoints(1099), None);
}
#[test]
fn mapping_contract_violations_panic_in_position_and_endpoint_queries() {
    struct Broken;
    impl graph_adaptor::IdMapping<u32> for Broken {
        type External = u32;
        fn to_external(&self, id: u32) -> Option<u32> {
            Some(id + 1000)
        }
        fn to_internal(&self, _: u32) -> Option<u32> {
            Some(10)
        }
    }
    let graph = graph_adaptor::MapIds::new(Slots, Broken, Offset);
    assert!(std::panic::catch_unwind(|| graph.node_at_index(99)).is_err());
    assert!(std::panic::catch_unwind(|| graph.node_at_dense(1)).is_err());
    assert!(std::panic::catch_unwind(|| graph.endpoints(1007)).is_err());
}
#[test]
fn cached_selection_is_fixed_and_hidden_endpoints_are_excluded() {
    use graph_adaptor::{AllEdges, IndexedSubgraph, Predicate};
    let selected = std::cell::Cell::new(true);
    let graph = IndexedSubgraph::build(
        Slots,
        Predicate {
            predicate: |_: u32| selected.get(),
        },
        AllEdges,
    )
    .unwrap();
    selected.set(false);
    assert_eq!(graph.node_ids().collect::<Vec<_>>(), vec![10, 20]);
    assert_eq!(graph.edge_ids().collect::<Vec<_>>(), vec![7, 8]);
    let hidden = IndexedSubgraph::build(
        Slots,
        Predicate {
            predicate: |id| id == 10,
        },
        AllEdges,
    )
    .unwrap();
    assert_eq!(hidden.node_ids().collect::<Vec<_>>(), vec![10]);
    assert_eq!(hidden.edge_ids().count(), 0);
    assert_eq!(hidden.to_dense(10), Some(0));
    assert!(hidden.node(20).is_none());
    assert!(hidden.edge(7).is_none());
    let empty = IndexedSubgraph::build(
        Slots,
        Predicate {
            predicate: |_: u32| false,
        },
        AllEdges,
    )
    .unwrap();
    assert_eq!(empty.node_count(), 0);
    assert_eq!(empty.node_at_index(0), None);
}
