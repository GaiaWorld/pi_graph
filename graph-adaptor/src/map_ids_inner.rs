//! 身份翻译配对实现；已有身份转换须满足完整映射与往返契约。
use super::*;
#[cfg(test)]
#[test]
fn t28_translates_lazily_without_skipping_missing_ids() {
    struct Mapping;
    impl IdMapping<u32> for Mapping {
        type External = u32;
        fn to_external(&self, id: u32) -> Option<u32> {
            (id != 30).then_some(id + 1000)
        }
        fn to_internal(&self, id: u32) -> Option<u32> {
            id.checked_sub(1000)
        }
    }
    let mut ids = TranslatedIds {
        _mapping: &Mapping,
        _iter: [20, 10, 20, 30].into_iter(),
    };
    assert_eq!(ids.next(), Some(1020));
    assert_eq!(ids.next(), Some(1010));
    assert_eq!(ids.next(), Some(1020));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ids.next())).is_err());
    assert_eq!(ids.next(), None);
}
// 公共 seam：Iterator::next；字段只用于构造原始候选夹具。
#[cfg(test)]
#[test]
fn t28_rejects_non_round_tripping_candidates_before_yielding() {
    struct Mapping(Option<u32>);
    impl IdMapping<u32> for Mapping {
        type External = u32;
        fn to_external(&self, _: u32) -> Option<u32> {
            Some(1010)
        }
        fn to_internal(&self, _: u32) -> Option<u32> {
            self.0
        }
    }
    for reverse in [None, Some(10), Some(999)] {
        let mapping = Mapping(reverse);
        let mut ids = TranslatedIds {
            _mapping: &mapping,
            _iter: [20].into_iter(),
        };
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ids.next())).is_err());
        assert_eq!(ids.next(), None);
    }
}
#[cfg(test)]
#[test]
fn t52() {
    #[derive(Clone)]
    struct Mapping;
    impl IdMapping<u32> for Mapping {
        type External = u32;
        fn to_external(&self, id: u32) -> Option<u32> {
            id.checked_add(1000)
        }
        fn to_internal(&self, id: u32) -> Option<u32> {
            id.checked_sub(1000)
        }
    }
    let mut map = TranslatedVisitMap {
        _mapping: Mapping,
        _identity: std::marker::PhantomData,
        _visited: std::collections::HashSet::<u32>::new(),
    };
    assert!(!VisitMap::is_visited(&map, 1010));
    VisitMap::visit(&mut map, 1010);
    assert!(VisitMap::is_visited(&map, 1010));
    assert!(std::panic::catch_unwind(|| VisitMap::is_visited(&map, 999)).is_err());
}
#[cfg(test)]
#[test]
fn t51() {
    #[derive(Clone)]
    struct Mapping;
    impl IdMapping<u32> for Mapping {
        type External = u32;
        fn to_external(&self, id: u32) -> Option<u32> {
            id.checked_add(1000)
        }
        fn to_internal(&self, id: u32) -> Option<u32> {
            id.checked_sub(1000)
        }
    }
    let mut map = TranslatedVisitMap {
        _mapping: Mapping,
        _identity: std::marker::PhantomData,
        _visited: std::collections::HashSet::<u32>::new(),
    };
    assert!(VisitMap::visit(&mut map, 1010));
    assert!(!VisitMap::visit(&mut map, 1010));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| VisitMap::visit(
            &mut map, 999
        )))
        .is_err()
    );
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅建立引用协议测试夹具。
#[cfg(test)]
#[test]
fn t34() {
    let r = TranslatedEdgeRef {
        _reference: graph_core::EdgeReference::new(7, 10, 20, &9),
        _id: 1007,
        _source: 1010,
        _target: 1020,
    };
    assert_eq!(*EdgeRef::weight(&r), 9);
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅建立引用协议测试夹具。
#[cfg(test)]
#[test]
fn t33() {
    let r = TranslatedEdgeRef {
        _reference: graph_core::EdgeReference::new(7, 10, 20, &9),
        _id: 1007,
        _source: 1010,
        _target: 1020,
    };
    assert_eq!(EdgeRef::target(&r), 1020);
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅建立引用协议测试夹具。
#[cfg(test)]
#[test]
fn t32() {
    let r = TranslatedEdgeRef {
        _reference: graph_core::EdgeReference::new(7, 10, 20, &9),
        _id: 1007,
        _source: 1010,
        _target: 1020,
    };
    assert_eq!(EdgeRef::source(&r), 1010);
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅建立引用协议测试夹具。
#[cfg(test)]
#[test]
fn t31() {
    let r = TranslatedEdgeRef {
        _reference: graph_core::EdgeReference::new(7, 10, 20, &9),
        _id: 1007,
        _source: 1010,
        _target: 1020,
    };
    assert_eq!(EdgeRef::id(&r), 1007);
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅建立引用协议测试夹具。
#[cfg(test)]
#[test]
fn t30() {
    let r = TranslatedNodeRef {
        _reference: graph_core::NodeReference::new(10, &42),
        _id: 1010,
    };
    assert_eq!(*NodeRef::weight(&r), 42);
}
// 公共 seam：NodeRef / EdgeRef；私有字段仅建立引用协议测试夹具。
#[cfg(test)]
#[test]
fn t29() {
    let r = TranslatedNodeRef {
        _reference: graph_core::NodeReference::new(10, &42),
        _id: 1010,
    };
    assert_eq!(NodeRef::id(&r), 1010);
}
impl<G, NM, EM> MapIds<G, NM, EM> {
    pub(super) fn inner_new(_graph: G, _nodes: NM, _edges: EM) -> Self {
        // 仅保存底层与映射，不验证全图或分配身份。
        Self {
            _graph,
            _nodes,
            _edges,
        }
    }
}
impl<I: Iterator, M: IdMapping<I::Item>> TranslatedIds<'_, M, I>
where
    I::Item: Copy + Eq,
{
    pub(super) fn inner_next(&mut self) -> Option<M::External> {
        // 逐个转换已有身份并校验反查回到原候选；缺失或往返不一致时显式 panic。
        self._iter.next().map(|id| {
            let external = self
                ._mapping
                .to_external(id)
                .expect("existing identity must have a mapping");
            assert!(
                self._mapping.to_internal(external) == Some(id),
                "identity mapping must round trip"
            );
            external
        })
    }
}

impl<R: NodeRef, N: Copy> TranslatedNodeRef<R, N> {
    pub(super) fn inner_id(&self) -> N {
        // 读壳侧已翻译字段或借用底层 weight，不重新申请 Guard。
        self._id
    }
    pub(super) fn inner_weight(&self) -> &R::Weight {
        // 读壳侧已翻译字段或借用底层 weight，不重新申请 Guard。
        self._reference.weight()
    }
}

impl<R: EdgeRef, N: Copy, E: Copy> TranslatedEdgeRef<R, N, E> {
    pub(super) fn inner_id(&self) -> E {
        // 读壳侧已翻译字段或借用底层 weight，不重新申请 Guard。
        self._id
    }
    pub(super) fn inner_source(&self) -> N {
        // 读壳侧已翻译字段或借用底层 weight，不重新申请 Guard。
        self._source
    }
    pub(super) fn inner_target(&self) -> N {
        // 读壳侧已翻译字段或借用底层 weight，不重新申请 Guard。
        self._target
    }
    pub(super) fn inner_weight(&self) -> &R::Weight {
        // 读壳侧已翻译字段或借用底层 weight，不重新申请 Guard。
        self._reference.weight()
    }
}

impl<G: NodeIds, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_node_ids(&self) -> <Self as NodeIds>::NodeIds<'_> {
        // 按壳侧无效起点和映射完整性规则构建迭代器。
        TranslatedIds {
            _mapping: &self._nodes,
            _iter: Some(self._graph.node_ids()).into_iter().flatten(),
        }
    }
}

impl<G: EdgeIds, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_edge_ids(&self) -> <Self as EdgeIds>::EdgeIds<'_> {
        // 按壳侧无效起点和映射完整性规则构建迭代器。
        TranslatedIds {
            _mapping: &self._edges,
            _iter: Some(self._graph.edge_ids()).into_iter().flatten(),
        }
    }
}

impl<G: OutNeighbors, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_out_neighbors(
        &self,
        _node: NM::External,
    ) -> <Self as OutNeighbors>::OutNeighbors<'_> {
        // 按壳侧无效起点和映射完整性规则构建迭代器。
        TranslatedIds {
            _mapping: &self._nodes,
            _iter: self
                ._nodes
                .to_internal(_node)
                .map(|node| {
                    assert!(
                        self._nodes.to_external(node) == Some(_node),
                        "node mapping must round trip"
                    );
                    self._graph.out_neighbors(node)
                })
                .into_iter()
                .flatten(),
        }
    }
}

impl<G: InNeighbors, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_in_neighbors(
        &self,
        _node: NM::External,
    ) -> <Self as InNeighbors>::InNeighbors<'_> {
        // 按壳侧无效起点和映射完整性规则构建迭代器。
        TranslatedIds {
            _mapping: &self._nodes,
            _iter: self
                ._nodes
                .to_internal(_node)
                .map(|node| {
                    assert!(
                        self._nodes.to_external(node) == Some(_node),
                        "node mapping must round trip"
                    );
                    self._graph.in_neighbors(node)
                })
                .into_iter()
                .flatten(),
        }
    }
}

impl<G: OutEdges, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_out_edges(&self, _node: NM::External) -> <Self as OutEdges>::OutEdges<'_> {
        // 按壳侧无效起点和映射完整性规则构建迭代器。
        TranslatedIds {
            _mapping: &self._edges,
            _iter: self
                ._nodes
                .to_internal(_node)
                .map(|node| {
                    assert!(
                        self._nodes.to_external(node) == Some(_node),
                        "node mapping must round trip"
                    );
                    self._graph.out_edges(node)
                })
                .into_iter()
                .flatten(),
        }
    }
}

impl<G: InEdges, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_in_edges(&self, _node: NM::External) -> <Self as InEdges>::InEdges<'_> {
        // 按壳侧无效起点和映射完整性规则构建迭代器。
        TranslatedIds {
            _mapping: &self._edges,
            _iter: self
                ._nodes
                .to_internal(_node)
                .map(|node| {
                    assert!(
                        self._nodes.to_external(node) == Some(_node),
                        "node mapping must round trip"
                    );
                    self._graph.in_edges(node)
                })
                .into_iter()
                .flatten(),
        }
    }
}

impl<G: NodeAccess, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_node(
        &self,
        _id: NM::External,
    ) -> Option<<Self as NodeAccess>::NodeRef<'_>> {
        // 保留底层引用所有权，只翻译元数据并校验往返。
        let internal = self._nodes.to_internal(_id)?;
        let reference = self._graph.node(internal)?;
        let id = self
            ._nodes
            .to_external(reference.id())
            .expect("existing node must have a mapping");
        assert!(
            id == _id && self._nodes.to_internal(id) == Some(reference.id()),
            "node mapping must round trip"
        );
        Some(TranslatedNodeRef {
            _reference: reference,
            _id: id,
        })
    }
}

impl<G: EdgeAccess, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_edge(
        &self,
        _id: EM::External,
    ) -> Option<<Self as EdgeAccess>::EdgeRef<'_>> {
        // 保留底层引用所有权，只翻译元数据并校验往返。
        let internal = self._edges.to_internal(_id)?;
        let reference = self._graph.edge(internal)?;
        let id = self
            ._edges
            .to_external(reference.id())
            .expect("existing edge must have a mapping");
        assert!(
            id == _id && self._edges.to_internal(id) == Some(reference.id()),
            "edge mapping must round trip"
        );
        let translate = |node| {
            let external = self
                ._nodes
                .to_external(node)
                .expect("existing endpoint must have a mapping");
            assert!(
                self._nodes.to_internal(external) == Some(node),
                "endpoint mapping must round trip"
            );
            external
        };
        let source = translate(reference.source());
        let target = translate(reference.target());
        Some(TranslatedEdgeRef {
            _reference: reference,
            _id: id,
            _source: source,
            _target: target,
        })
    }
}

impl<G: EdgeEndpoints, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_endpoints(
        &self,
        _edge: EM::External,
    ) -> Option<(NM::External, NM::External)> {
        // 按方向或位置保留语义，转换身份并检查映射，不把未知 ID 当索引。
        let edge = self._edges.to_internal(_edge)?;
        let (source, target) = self._graph.endpoints(edge)?;
        assert!(
            self._edges.to_external(edge) == Some(_edge),
            "edge mapping must round trip"
        );
        let translate = |node| {
            let external = self
                ._nodes
                .to_external(node)
                .expect("existing endpoint must have a mapping");
            assert!(
                self._nodes.to_internal(external) == Some(node),
                "endpoint mapping must round trip"
            );
            external
        };
        Some((translate(source), translate(target)))
    }
}

impl<G: NodeIndexable, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_node_bound(&self) -> usize {
        // 按方向或位置保留语义，转换身份并检查映射，不把未知 ID 当索引。
        self._graph.node_bound()
    }
    pub(super) fn inner_to_index(&self, _node: NM::External) -> usize {
        // 按方向或位置保留语义，转换身份并检查映射，不把未知 ID 当索引。
        let node = self
            ._nodes
            .to_internal(_node)
            .expect("external node must have a mapping");
        assert!(
            self._nodes.to_external(node) == Some(_node),
            "node mapping must round trip"
        );
        self._graph.to_index(node)
    }
    pub(super) fn inner_node_at_index(&self, _index: usize) -> Option<NM::External> {
        // 按方向或位置保留语义，转换身份并检查映射，不把未知 ID 当索引。
        self._graph.node_at_index(_index).map(|node| {
            let external = self
                ._nodes
                .to_external(node)
                .expect("existing node must have a mapping");
            assert!(
                self._nodes.to_internal(external) == Some(node),
                "node mapping must round trip"
            );
            external
        })
    }
}

impl<G: DenseNodeIndex, NM: IdMapping<G::NodeId>, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_node_count(&self) -> usize {
        // 按方向或位置保留语义，转换身份并检查映射，不把未知 ID 当索引。
        self._graph.node_count()
    }
    pub(super) fn inner_to_dense(&self, _id: NM::External) -> Option<usize> {
        // 按方向或位置保留语义，转换身份并检查映射，不把未知 ID 当索引。
        let node = self._nodes.to_internal(_id)?;
        let position = self._graph.to_dense(node)?;
        assert!(
            self._nodes.to_external(node) == Some(_id),
            "node mapping must round trip"
        );
        Some(position)
    }
    pub(super) fn inner_node_at_dense(&self, _index: usize) -> Option<NM::External> {
        // 按方向或位置保留语义，转换身份并检查映射，不把未知 ID 当索引。
        self._graph.node_at_dense(_index).map(|node| {
            let external = self
                ._nodes
                .to_external(node)
                .expect("existing node must have a mapping");
            assert!(
                self._nodes.to_internal(external) == Some(node),
                "node mapping must round trip"
            );
            external
        })
    }
}

impl<G: Visitable, NM: IdMapping<G::NodeId> + Clone, EM: IdMapping<G::EdgeId>> MapIds<G, NM, EM> {
    pub(super) fn inner_visit_map(&self) -> TranslatedVisitMap<G::NodeId, NM, G::Map> {
        // 复制稳定节点映射，持有底层全新集合，不借用适配器。
        TranslatedVisitMap {
            _mapping: self._nodes.clone(),
            _identity: std::marker::PhantomData,
            _visited: self._graph.visit_map(),
        }
    }
}
impl<I: Copy, M: IdMapping<I>, V: VisitMap<I>> TranslatedVisitMap<I, M, V> {
    pub(super) fn inner_visit(&mut self, _node: M::External) -> bool {
        // 反查并校验往返后标记底层集合，未知 ID 明确 panic。
        let node = self
            ._mapping
            .to_internal(_node)
            .expect("external node must have a mapping");
        assert!(
            self._mapping.to_external(node) == Some(_node),
            "node mapping must round trip"
        );
        self._visited.visit(node)
    }
    pub(super) fn inner_is_visited(&self, _node: M::External) -> bool {
        // 反查并校验往返后查询底层集合。
        let node = self
            ._mapping
            .to_internal(_node)
            .expect("external node must have a mapping");
        assert!(
            self._mapping.to_external(node) == Some(_node),
            "node mapping must round trip"
        );
        self._visited.is_visited(node)
    }
}
