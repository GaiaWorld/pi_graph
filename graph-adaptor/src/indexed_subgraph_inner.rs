//! 缓存子图配对实现；构建时验证完整性，读取时使用固定缓存。
use super::*;
impl<G: NodeIds + EdgeIds + EdgeEndpoints> IndexedSubgraph<G> {
    pub(super) fn inner_build<N: Selection<G::NodeId>, E: Selection<G::EdgeId>>(
        _graph: G,
        _nodes: N,
        _edges: E,
    ) -> IndexedBuildResult<G> {
        // 按壳侧唯一扫描流程构建缓存；完整性检查需临时记录所有底层节点和边。
        // 持久缓存 O(V'+E')，构建临时验证空间允许 O(V+E)，不得误称全程仅 O(V'+E')。
        let mut all_nodes = HashSet::new();
        let mut visible_nodes = Vec::new();
        let mut positions = HashMap::new();
        for node in _graph.node_ids() {
            if !all_nodes.insert(node) {
                return Err(IndexedSubgraphError::DuplicateNode { node });
            }
            if _nodes.contains(node) {
                positions.insert(node, visible_nodes.len());
                visible_nodes.push(node);
            }
        }
        let count = visible_nodes.len();
        let mut visible_edges = Vec::new();
        let mut endpoints = HashMap::new();
        let mut out_edges = vec![Vec::new(); count];
        let mut in_edges = vec![Vec::new(); count];
        let mut out_neighbors = vec![Vec::new(); count];
        let mut in_neighbors = vec![Vec::new(); count];
        let mut all_edges = HashSet::new();
        for edge in _graph.edge_ids() {
            if !all_edges.insert(edge) {
                return Err(IndexedSubgraphError::DuplicateEdge { edge });
            }
            let (source, target) = _graph
                .endpoints(edge)
                .ok_or(IndexedSubgraphError::MissingEndpoints { edge })?;
            for node in [source, target] {
                if !all_nodes.contains(&node) {
                    return Err(IndexedSubgraphError::UnknownEndpoint { edge, node });
                }
            }
            let selected = _edges.contains(edge);
            if let (true, Some(&s), Some(&t)) =
                (selected, positions.get(&source), positions.get(&target))
            {
                visible_edges.push(edge);
                endpoints.insert(edge, (source, target));
                out_edges[s].push(edge);
                in_edges[t].push(edge);
                out_neighbors[s].push(target);
                in_neighbors[t].push(source);
            }
        }
        Ok(Self {
            _graph,
            _nodes: visible_nodes,
            _positions: positions,
            _edges: visible_edges,
            _endpoints: endpoints,
            _out_edges: out_edges,
            _in_edges: in_edges,
            _out_neighbors: out_neighbors,
            _in_neighbors: in_neighbors,
        })
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_node_ids(&self) -> <Self as NodeIds>::NodeIds<'_> {
        // 返回缓存切片迭代器，未知节点返回空切片。
        self._nodes.iter().copied()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_edge_ids(&self) -> <Self as EdgeIds>::EdgeIds<'_> {
        // 返回缓存切片迭代器，未知节点返回空切片。
        self._edges.iter().copied()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_out_neighbors(
        &self,
        _node: G::NodeId,
    ) -> <Self as OutNeighbors>::OutNeighbors<'_> {
        // 返回缓存切片迭代器，未知节点返回空切片。
        self._positions
            .get(&_node)
            .map_or(&[][..], |&i| self._out_neighbors[i].as_slice())
            .iter()
            .copied()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_in_neighbors(
        &self,
        _node: G::NodeId,
    ) -> <Self as InNeighbors>::InNeighbors<'_> {
        // 返回缓存切片迭代器，未知节点返回空切片。
        self._positions
            .get(&_node)
            .map_or(&[][..], |&i| self._in_neighbors[i].as_slice())
            .iter()
            .copied()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_out_edges(&self, _node: G::NodeId) -> <Self as OutEdges>::OutEdges<'_> {
        // 返回缓存切片迭代器，未知节点返回空切片。
        self._positions
            .get(&_node)
            .map_or(&[][..], |&i| self._out_edges[i].as_slice())
            .iter()
            .copied()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_in_edges(&self, _node: G::NodeId) -> <Self as InEdges>::InEdges<'_> {
        // 返回缓存切片迭代器，未知节点返回空切片。
        self._positions
            .get(&_node)
            .map_or(&[][..], |&i| self._in_edges[i].as_slice())
            .iter()
            .copied()
    }
}

impl<G: NodeAccess> IndexedSubgraph<G> {
    pub(super) fn inner_node(&self, _id: G::NodeId) -> Option<G::NodeRef<'_>> {
        // 先缓存成员判断，再访问底层数据；不重新求值选择。
        self._positions
            .contains_key(&_id)
            .then(|| self._graph.node(_id))
            .flatten()
    }
}

impl<G: EdgeAccess> IndexedSubgraph<G> {
    pub(super) fn inner_edge(&self, _id: G::EdgeId) -> Option<G::EdgeRef<'_>> {
        // 先缓存成员判断，再访问底层数据；不重新求值选择。
        self._endpoints
            .contains_key(&_id)
            .then(|| self._graph.edge(_id))
            .flatten()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_endpoints(&self, _edge: G::EdgeId) -> Option<(G::NodeId, G::NodeId)> {
        // 读取缓存身份、端点或紧凑映射，或创建全新空集合，不访问底层索引。
        self._endpoints.get(&_edge).copied()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_node_bound(&self) -> usize {
        // 读取缓存身份、端点或紧凑映射，或创建全新空集合，不访问底层索引。
        self._nodes.len()
    }
    pub(super) fn inner_to_index(&self, _node: G::NodeId) -> usize {
        // 读取缓存身份、端点或紧凑映射，或创建全新空集合，不访问底层索引。
        *self
            ._positions
            .get(&_node)
            .expect("node must belong to cached subgraph")
    }
    pub(super) fn inner_node_at_index(&self, _index: usize) -> Option<G::NodeId> {
        // 读取缓存身份、端点或紧凑映射，或创建全新空集合，不访问底层索引。
        self._nodes.get(_index).copied()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_node_count(&self) -> usize {
        // 读取缓存身份、端点或紧凑映射，或创建全新空集合，不访问底层索引。
        self._nodes.len()
    }
    pub(super) fn inner_to_dense(&self, _id: G::NodeId) -> Option<usize> {
        // 读取缓存身份、端点或紧凑映射，或创建全新空集合，不访问底层索引。
        self._positions.get(&_id).copied()
    }
    pub(super) fn inner_node_at_dense(&self, _index: usize) -> Option<G::NodeId> {
        // 读取缓存身份、端点或紧凑映射，或创建全新空集合，不访问底层索引。
        self._nodes.get(_index).copied()
    }
}

impl<G: GraphBase> IndexedSubgraph<G> {
    pub(super) fn inner_visit_map(&self) -> HashSet<G::NodeId> {
        // 读取缓存身份、端点或紧凑映射，或创建全新空集合，不访问底层索引。
        HashSet::new()
    }
}
