//! 有向无环图

use log::{debug, error};
use std::{fmt::Debug, mem::replace, option::Iter, collections::VecDeque};
use pi_slotmap::{Key, SecondaryMap};
use pi_map::vecmap::VecMap;

/// 有向无环图
/// K 节点的键
/// T 节点的值
// #[delegate]
pub trait DirectedGraph<K: Key, T> {
    /// 节点
    type Node: DirectedGraphNode<K, T>;

    // /// 迭代器的关联类型，指定了迭代器`Item`为`K`
    // type NodeIter: Iterator<Item = &'a K>;

    /// 根据 key 取 节点
    fn get(&self, key: K) -> Option<&Self::Node>;

    /// 根据 key 取 节点
    fn get_mut(&mut self, key: K) -> Option<&mut Self::Node>;

    /// 取节点的数量
    fn node_count(&self) -> usize;

    /// 取 有输入节点 的 数量
    fn from_len(&self) -> usize;

    /// 取 有输出节点 的 数量
    fn to_len(&self) -> usize;

    /// 取 输入节点 切片
    fn from(&self) -> &[K];

    /// 取 输出节点 切片
    fn to(&self) -> &[K];

    /// 拓扑排序
    fn topological_sort(&self) -> &[K];

    // /// 全部节点的迭代器
    // fn nodes('a self) -> Self::NodeIter;

    // /// 获取图的froms节点的迭代器
    // fn from(&'a self) -> Self::NodeIter;

    // /// 获取图的froms节点的迭代器
    // fn to(&'a self) -> Self::NodeIter;

    // /// 从from节点开始的深度遍历迭代器
    // fn from_dfs('a self) -> Self::NodeIter;

    // /// 从from节点开始的深度遍历迭代器
    // fn to_dfs('a self) -> Self::NodeIter;
}

/// 有向无环图 节点
pub trait DirectedGraphNode<K: Key, T> {
    // /// 迭代器的关联类型，指定了迭代器`Item`为`K`
    // type NodeIter: Iterator<Item = &'a K>;

    /// 取 from节点 的 数量
    fn from_len(&self) -> usize;

    /// 取 to节点 的 数量
    fn to_len(&self) -> usize;

    /// 取 入点 的 切片
    fn from(&self) -> &[K];

    /// 取 出点 的 切片
    fn to(&self) -> &[K];

    /// 取键的引用
    fn key(&self) -> &K;

    /// 取值的引用
    fn value(&self) -> &T;

    /// 取值的可变引用
    fn value_mut(&mut self) -> &mut T;

    // /// 读取计数器
    // fn load_count(&self) -> usize;

    // /// 增加计数器的值
    // fn add_count(&mut self, add: usize) -> usize;

    // /// 设置计数器的值
    // fn set_count(&mut self, count: usize);

    // /// 获取from节点的迭代器
    // fn from(&self) -> Self::NodeIter;

    // /// 获取to节点的迭代器
    // fn to(&self) -> Self::NodeIter;
}

// 遍历邻居的迭代器 TODO 不好实现
/// 节点 迭代器
pub struct NodeIterator<'a, K>(Iter<'a, K>);

impl<'a, K> Iterator for NodeIterator<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

/// 图
#[derive(Default, Debug)]
pub struct NGraph<K: Key, T> {
    // 所有节点
    map: SecondaryMap<K, NGraphNode<K, T>>,

    // 入度为0 的 节点
    from: Vec<K>,

    // 出度为0 的 节点
    to: Vec<K>,

    // 拓扑排序后的结果
    topological: Vec<K>,
}

/// 图节点
#[derive(Debug)]
pub struct NGraphNode<K: Key, T> {
    // 该节点的 入度节点
    from: Vec<K>,

    // 该节点的 出度节点
    to: Vec<K>,
    // 键
    key: K,
    // 值
    value: T,
}

impl<K: Key, T> NGraphNode<K, T>{
	// 到 from 节点删掉 to
	#[inline]
	fn remove_edge_from(&mut self, from: K) -> bool {
        if let Some(index) = self.from.iter().position(|v| *v == from) {
            self.from.swap_remove(index);
			true
        } else {
			false
		}
    }

	// 到 to 节点删掉 from
	#[inline]
	fn remove_edge_to(&mut self, to: K) -> bool {
        if let Some(index) = self.to.iter().position(|v| *v == to) {
            self.to.swap_remove(index);
			true
        } else {
			false
		}
    }

	// 添加to
	#[inline]
    fn add_edge_to(&mut self, to: K) -> bool {
        match self.to.iter().position(|s| *s == to) {
            Some(_) => false,
            None => {
                self.to.push(to);
                true
            }
        }
    }

    // // 添加from
    // #[inline]
    // fn add_edge_from(&mut self, from: K) -> bool {
    //     match self.from.iter().position(|s| *s == from) {
    //         Some(_) => false,
    //         None => {
    //             self.from.push(from);
	// 			true
    //         }
    //     }
    // }
}

impl<K: Key, T: Clone> Clone for NGraphNode<K, T> {
    fn clone(&self) -> Self {
        Self {
            from: self.from.clone(),
            to: self.to.clone(),
            key: self.key.clone(),
            value: self.value.clone(),
        }
    }
}

impl<K: Key, T> DirectedGraphNode<K, T> for NGraphNode<K, T> {
    #[inline]
	fn from_len(&self) -> usize {
        self.from.len()
    }

	#[inline]
    fn to_len(&self) -> usize {
        self.to.len()
    }

	#[inline]
    fn from(&self) -> &[K] {
        &self.from[..]
    }

    fn to(&self) -> &[K] {
        &self.to[..]
    }

	#[inline]
    fn key(&self) -> &K {
        &self.key
    }

	#[inline]
    fn value(&self) -> &T {
        &self.value
    }

	#[inline]
    fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }

	// #[inline]
    // fn load_count(&self) -> usize {
	// 	self.count
    //     // self.count.load(Ordering::Relaxed)
    // }

	// #[inline]
    // fn add_count(&mut self, add: usize) -> usize {
	// 	self.count += add;
	// 	self.count
    //     // self.count.fetch_add(add, Ordering::SeqCst)
    // }

	// #[inline]
    // fn set_count(&mut self, count: usize) {
	// 	self.count = count;
    //     // self.count.store(count, Ordering::SeqCst)
    // }
}

impl<K: Key, T> NGraph<K, T> {
    pub(crate) fn new() -> Self {
        Self {
            map: Default::default(),
            from: Default::default(),
            to: Default::default(),
            topological: Default::default(),
        }
    }

    // /// 重置 图
    // pub fn reset(&self) {
    //     for n in self.map.values() {
    //         n.set_count(0)
    //     }
    // }
}

impl<K: Key, T> NGraph<K, T> {
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.values().into_iter().map(|v| v.value())
    }
}

impl<K: Key, T> DirectedGraph<K, T> for NGraph<K, T> {
    // /// 迭代器的关联类型，指定了迭代器`Item`为`K`
    // type NodeIter = NodeIterator<'a, K>;

    type Node = NGraphNode<K, T>;

    fn get(&self, key: K) -> Option<&Self::Node> {
        self.map.get(key)
    }

    fn get_mut(&mut self, key: K) -> Option<&mut Self::Node> {
        self.map.get_mut(key)
    }

    fn node_count(&self) -> usize {
        self.map.len()
    }
    fn from_len(&self) -> usize {
        self.from.len()
    }

    fn to_len(&self) -> usize {
        self.to.len()
    }
    fn from(&self) -> &[K] {
        &self.from[..]
    }

    fn to(&self) -> &[K] {
        &self.to[..]
    }
    fn topological_sort(&self) -> &[K] {
        &self.topological[..]
    }
    // fn check_loop(&self) -> Option<&K> {
    //     let mut stack = Vec::new();
    //     let mut arr = (0, self.from());
    //     loop {
    //         while arr.0 < arr.1.len() {
    //             let k = &arr.1[arr.0];
    //             arr.0 += 1;
    //             let n = self.get(k).unwrap();
    //             if n.to_len() > 0 {
    //                 if n.from_len() < n.load_count() {
    //                     self.reset();
    //                     return Some(k);
    //                 }
    //                 // 进入次数加1
    //                 n.add_count(1);
    //                 // 将当前的节点切片放入栈
    //                 stack.push(arr);
    //                 // 切换成检查下一层的节点切片
    //                 arr = (0, n.to());
    //             }
    //         }
    //         match stack.pop() {
    //             Some(r) => arr = r,
    //             _ => {
    //                 self.reset();
    //                 return None;
    //             }
    //         }
    //     }
    // }
}

impl<K: Key, T: Clone> NGraph<K, T> {
    /// 遍历 局部图
    pub fn gen_graph_from_keys(&self, finish: &[K]) -> Self {
        let mut builder = NGraphBuilder::new();

        debug!("gen_graph_from_keys, param keys = {:?}", finish);

        let mut current_keys = vec![];
        for k in finish {
            current_keys.push(k.clone());

            let n = self.map.get(k.clone()).unwrap();

            // 防止 keys的 重复 元素
            if !builder.has_node(*k) {
                debug!("gen_graph_from_keys, add node k = {:?}", k);
                builder.node(*k, n.value.clone());
            }
        }

        while !current_keys.is_empty() {
            debug!("gen_graph_from_keys, current_keys = {:?}", current_keys);

            let mut from_keys = vec![];

            for curr in current_keys.iter() {
                let curr_node = self.map.get(curr.clone()).unwrap();

                // 下一轮 出点
                for from in curr_node.from() {
                    let from_node = self.map.get(from.clone()).unwrap();

                    if !builder.has_node(*from) {
                        debug!("gen_graph_from_keys, add node next = {:?}", from);
                        builder.node(from.clone(), from_node.value.clone());
                    }

                    debug!("gen_graph_from_keys, add edge = ({:?}, {:?})", from, curr);
                    builder.edge(from.clone(), curr.clone());
                    from_keys.push(from.clone());
                }
            }

            debug!("gen_graph_from_keys, from_keys = {:?}", from_keys);

            let _ = replace(&mut current_keys, from_keys);
        }

        builder.build().unwrap()
    }
}

/// 图 构建器
pub struct NGraphBuilder<K: Key, T> {
    graph: NGraph<K, T>,
}

impl<K: Key, T> NGraphBuilder<K, T> {
    /// 创建 默认
    pub fn new() -> Self {
        NGraphBuilder {
            graph: NGraph::new(),
        }
    }

    /// 用已有的图 重新 构建
    pub fn new_with_graph(graph: NGraph<K, T>) -> Self {
        // graph.from.clear();
        // graph.to.clear();
        // graph.topological.clear();

        NGraphBuilder { graph }
    }

	pub fn graph(&self) -> &NGraph<K, T> {
        &self.graph
    }

    /// 对应节点是否存在
    pub fn has_node(&self, key: K) -> bool {
        self.graph.map.contains_key(key)
    }

    /// 添加 节点
    pub fn node(&mut self, key: K, value: T) -> &mut Self {
        self.graph.map.insert(
            key.clone(),
            NGraphNode {
                from: Default::default(),
                to: Default::default(),
                key,
                value,
            },
        );

        self
    }

    /// 添加 边
    pub fn edge(&mut self, from: K, to: K) -> &mut Self {
		if let Some([from_node, to_node]) = self.graph.map.get_disjoint_mut([from, to]) {
			if from_node.add_edge_to(to) {
				to_node.from.push(from);
			}
		}
        self
    }

    /// 移除 节点
    pub fn remove_node(&mut self, key: K) -> &mut Self {
		if let Some(node) = self.graph.map.remove(key) {
			for f in node.from.iter() {
				self.graph.map.get_mut(*f).unwrap().remove_edge_to(key);
			}
	
			for t in node.to.iter() {
				self.graph.map.get_mut(*t).unwrap().remove_edge_from(key);
			}
		}
        self
    }

    /// 移除 边
    pub fn remove_edge(&mut self, from: K, to: K) -> &mut Self {
		if let Some(from_node) = self.graph.map.get_mut(from) {
			if from_node.remove_edge_to(to) {
				self.graph.map.get_mut(to).unwrap().remove_edge_from(from);
			}
		}
        self
    }

    /// 构建图
    /// 返回Graph，或者 回环的节点
    pub fn build(mut self) -> Result<NGraph<K, T>, Vec<K>> {
		let mut counts = VecMap::with_capacity(self.graph.map.len());
		// 已经处理过的节点Key
        let mut topos: Vec<K> = Vec::with_capacity(self.graph.map.len());

        // 计算开头 和 结尾的 节点
        for (k, v) in self.graph.map.iter() {
            // 开头：没有入边的点
            if v.from.is_empty() {
                self.graph.from.push(k);
            }

            // 结尾：没有出边的点
            if v.to.is_empty() {
                self.graph.to.push(k);
            }

			counts.insert(key_index(k), v.from.len());
        }

        debug!("graph's from = {:?}", self.graph.from());
		let mut queue = self.graph.from.iter().copied().collect::<VecDeque<K>>();
        while let Some(k) = queue.pop_front() {
            topos.push(k);
			
            // 处理 from 的 下一层
            let node = self.graph.get(k).unwrap();
			debug!("from = {:?}, to: {:?}", k, node.to());
            // 遍历节点的后续节点
            for to in node.to()  {
				counts[key_index(*to)] -= 1;
				debug!("graph's each = {:?}, count = {:?}", to, counts[key_index(*to)]);
                // handle_set.insert(*to, ());
				if counts[key_index(*to)] == 0 {
					queue.push_back(*to);
				}
                // // 即将处理：将节点的计数加1
                // let n = self.graph.map.get_mut(*to).unwrap();
                // n.add_count(1);

                // debug!(
                //     "add n: k: {:?}, from:{} count:{}",
                //     to,
                //     n.from_len(),
                //     n.load_count()
                // );
            }
        }

		// 如果拓扑排序列表的节点数等于图中的总节点数，则返回拓扑排序列表，否则返回空列表（说明图中存在环路）
		if topos.len() == self.graph.map.len() {
			self.graph.topological = topos;
			return Ok(self.graph);
		}

		let keys = self.graph.map.keys().map(|k|{k.clone()}).filter(|r| {!topos.contains(r)}).collect::<Vec<K>>();
		let mut iter = keys.into_iter();
		while let Some(n) = iter.next() {
			let mut cycle_keys = Vec::new();
			self.find_cycle(n, &mut cycle_keys, Vec::new());

			if cycle_keys.len() > 0 {
				let cycle: Vec<(K, T)> = cycle_keys.iter().map(|k| {(k.clone(), self.graph.map.remove(*k).unwrap().value)}).collect();
				pi_print_any::out_any!(error, "graph build error, no from node, they make cycle: {:?}", cycle);
				return Result::Err(cycle_keys);
			}
		}
		return Result::Err(Vec::new());
    }
    /// 寻找循环依赖
    fn find_cycle(&mut self, node: K, nodes: &mut Vec<K>, mut indexs: Vec<usize>) {
		nodes.push(node.clone());
        indexs.push(0);
        while nodes.len() > 0 {
            let index = nodes.len() - 1;
            let k = &nodes[index];
            let n = self.graph.get(*k).unwrap();
            let to = n.to();
            let child_index = indexs[index];
            if child_index >= to.len() {
                nodes.pop();
                indexs.pop();
                continue
            }
            let child = to[child_index].clone();
            if child == node {
                break;
            }
            indexs[index] += 1;
            nodes.push(child);
            indexs.push(0);
        }
    }
}

#[inline]
pub fn key_index<K: Key>(k: K) -> usize{
	k.data().as_ffi() as u32 as usize
}

#[cfg(test)]
mod tests {
    use pi_slotmap::{DefaultKey, SlotMap};

    use crate::*;

    use std::sync::Once;

    static INIT: Once = Once::new();
    fn setup_logger() {
        use env_logger::{Builder, Env};

        INIT.call_once(|| {
            Builder::from_env(Env::default().default_filter_or("debug")).init();
        });
    }

    // 测试 无节点 的 图
    #[test]
    fn test_empty() {
        setup_logger();

        let graph = NGraphBuilder::<DefaultKey, u32>::new().build();

        assert_eq!(graph.is_ok(), true);

        let graph = graph.unwrap();
        assert_eq!(graph.node_count(), 0);

        assert_eq!(graph.from_len(), 0);
        assert_eq!(graph.from(), &[]);

        assert_eq!(graph.to_len(), 0);
        assert_eq!(graph.to(), &[]);

        assert_eq!(graph.topological_sort(), &[]);
    }

    // 测试 1个节点的 图
    #[test]
    fn test_one_node() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = map.insert(());
        let mut graph = NGraphBuilder::new();
		graph.node(node, 111);
		let graph = graph.build();

        assert_eq!(graph.is_ok(), true);

        let graph = graph.unwrap();
        assert_eq!(graph.node_count(), 1);

        assert_eq!(graph.from_len(), 1);
        assert_eq!(graph.from(), &[node]);

        assert_eq!(graph.to_len(), 1);
        assert_eq!(graph.to(), &[node]);

        assert_eq!(graph.topological_sort(), &[node]);
    }

    // 测试 无边 的 图
    #[test]
    fn test_no_edge() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [map.insert(()), map.insert(()), map.insert(())];
        // 1 2 3
        let mut graph = NGraphBuilder::new();
		graph.node(node[0], 1)
            .node(node[1], 2)
            .node(node[2], 3);
		let graph = graph.build();

        assert_eq!(graph.is_ok(), true);

        let graph = graph.unwrap();
        assert_eq!(graph.node_count(), 3);

        assert_eq!(graph.from_len(), 3);
        assert_eq!(graph.from(), node.as_slice());

        assert_eq!(graph.to_len(), 3);
        assert_eq!(graph.to(), node.as_slice());

        assert_eq!(graph.topological_sort(), node.as_slice());
    }

	// 测试 无边 的 图
    #[test]
    fn test_no_edge1() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [map.insert(()), map.insert(()), map.insert(()), map.insert(())];
        // 1 2 3
        let mut graph = NGraphBuilder::new();
        graph.node(node[0], 1)
            .node(node[1], 2)
            .node(node[2], 3)
			.node(node[3], 4)
			.edge(node[0], node[1])
            .edge(node[0], node[2])
			.edge(node[2], node[1])
			.edge(node[1], node[3]);
        let _graph = graph.build();

        // assert_eq!(graph.is_ok(), true);

        // let graph = graph.unwrap();
        // assert_eq!(graph.node_count(), 3);

        // assert_eq!(graph.from_len(), 3);
        // assert_eq!(graph.from(), &[1, 2, 3]);

        // assert_eq!(graph.to_len(), 3);
        // assert_eq!(graph.to(), &[1, 2, 3]);

        // assert_eq!(graph.topological_sort(), &[1, 2, 3]);
    }

    // 测试 简单的 图
    #[test]
    fn test_simple() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [map.insert(()), map.insert(()), map.insert(()), map.insert(())];
        // 1 --> 2 --> 3
        let mut graph = NGraphBuilder::new();
        graph.node(node[1], 1)
            .node(node[2], 2)
            .node(node[3], 3)
            .edge(node[1], node[2])
            .edge(node[2], node[3]);
        let graph = graph.build();

        assert_eq!(graph.is_ok(), true);

        let graph = graph.unwrap();
        assert_eq!(graph.node_count(), 3);

        assert_eq!(graph.from_len(), 1);
        assert_eq!(graph.from(), &[node[1]]);

        assert_eq!(graph.to_len(), 1);
        assert_eq!(graph.to(), &[node[3]]);

        assert_eq!(graph.topological_sort(), &[node[1], node[2], node[3]]);
    }

    // 测试 循环图
    #[test]
    fn test_cycle_graph() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [map.insert(()), map.insert(()), map.insert(()), map.insert(())];
        // 1 --> 2 --> 3 --> 1
        let mut graph = NGraphBuilder::new();
        graph.node(node[1], 1)
            .node(node[2], 2)
            .node(node[3], 3)
            .edge(node[1], node[2])
            .edge(node[2], node[3])
            .edge(node[3], node[1]);
        let graph = graph.build();

        assert_eq!(graph.is_err(), true);
        if let Err(r) = graph {
            assert_eq!(&r, &[node[1], node[2], node[3]]);
        }
    }

    // 测试 局部 循环
    #[test]
    fn test_cycle_local() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [map.insert(()), map.insert(()), map.insert(()), map.insert(())];
        // 1 --> 2 <--> 3
        let mut graph = NGraphBuilder::new();
		graph.node(node[1], 1)
            .node(node[2], 2)
            .node(node[3], 3)
            .edge(node[1], node[2])
            .edge(node[2], node[3])
            .edge(node[3], node[2]);
        let graph = graph.build();

        assert_eq!(graph.is_err(), true);
        if let Err(r) = graph {
            assert_eq!(&r, &[node[2], node[3]]);
        }
    }

    // 生成局部图
    #[test]
    fn test_gen_graph() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [
			map.insert(()), map.insert(()), map.insert(()), map.insert(()),
			map.insert(()), map.insert(()), map.insert(()), map.insert(())
		];
        // 7 --> 2, 6
        // 2, 3 --> 1
        // 5, 6 --> 4
        let mut graph = NGraphBuilder::new();
        graph.node(node[1], 1)
            .node(node[2], 2)
            .node(node[3], 3)
            .node(node[4], 4)
            .node(node[5], 5)
            .node(node[6], 6)
            .node(node[7], 7)
            .edge(node[7], node[2])
            .edge(node[7], node[6])
            .edge(node[2], node[1])
            .edge(node[3], node[1])
            .edge(node[5], node[4])
            .edge(node[6], node[4]);
        let graph = graph.build();

        assert_eq!(graph.is_ok(), true);

        let graph = graph.unwrap();
        let g2 = graph.gen_graph_from_keys(&[node[7]]);
        assert_eq!(g2.node_count(), 1);
    }

    // 复杂
    #[test]
    fn test_complex() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [
			map.insert(()), map.insert(()), map.insert(()), map.insert(()),
			map.insert(()), map.insert(()),
		];
        // 1 --> 3
        // 2 --> 3, 4, 5
        // 4 --> 5
        let mut graph = NGraphBuilder::new();
        graph.node(node[1], 1)
            .node(node[2], 2)
            .node(node[3], 3)
            .node(node[4], 4)
            .node(node[5], 5)
            .edge(node[1], node[3])
            .edge(node[2], node[3])
            .edge(node[2], node[4])
            .edge(node[2], node[5])
            .edge(node[4], node[5]);
        let graph = graph.build();

        assert_eq!(graph.is_ok(), true);

        let graph = graph.unwrap();
        assert_eq!(graph.node_count(), 5);

        assert_eq!(graph.from_len(), 2);

        let mut v: Vec<DefaultKey> = graph.from().iter().cloned().collect();
        v.sort();
        assert_eq!(&v, &[node[1], node[2]]);

        assert_eq!(graph.to_len(), 2);
        let mut v: Vec<DefaultKey> = graph.to().iter().cloned().collect();
        v.sort();
        assert_eq!(&v, &[node[3], node[5]]);

        assert_eq!(graph.topological_sort(), &[node[1], node[2], node[3], node[4], node[5]]);
    }

    #[test]
    fn test_graph() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [
			map.insert(()), map.insert(()), map.insert(()), map.insert(()),
			map.insert(()), map.insert(()), map.insert(()), map.insert(()),
			map.insert(()), map.insert(()), map.insert(()), map.insert(()),
		];
        let mut graph = NGraphBuilder::new();
        graph.node(node[1], 1)
            .node(node[2], 2)
            .node(node[3], 3)
            .node(node[4], 4)
            .node(node[5], 5)
            .node(node[6], 6)
            .node(node[7], 7)
            .node(node[8], 8)
            .node(node[9], 9)
            .node(node[10], 10)
            .node(node[11], 11)
            .edge(node[1], node[4])
            .edge(node[2], node[4])
            .edge(node[2], node[5])
            .edge(node[3], node[5])
            .edge(node[4], node[6])
            .edge(node[4], node[7])
            .edge(node[5], node[8])
            .edge(node[9], node[10])
            .edge(node[10], node[11])
            .edge(node[11], node[5])
            .edge(node[5], node[10]);
        let graph = graph.build();

        assert_eq!(graph.is_err(), true);
        if let Err(v) = graph {
            assert_eq!(&v, &[node[5], node[10], node[11]]);
        }
    }

	// 添加重复边、不存在节点的边、相同节点但是反向的边（循环依赖）
    #[test]
    fn test_add_repeat_edge() {
        setup_logger();
		let mut map = SlotMap::<DefaultKey, ()>::default();
		let node = [
			map.insert(()), map.insert(()), map.insert(()), map.insert(()),
			map.insert(()), map.insert(()),
		];
        // 1 --> 3
        // 2 --> 3, 4, 5
        // 4 --> 5
        let mut graph = NGraphBuilder::new();
        graph.node(node[1], 1)
            .node(node[2], 2)
            .edge(node[1], node[2])
            .edge(node[1], node[2])
            // .edge(node[2], node[1])
            .edge(node[2], node[3]);
        let graph = graph.build();

        assert_eq!(graph.is_ok(), true);

        let graph = graph.unwrap();

        let v: Vec<DefaultKey> = graph.from().iter().cloned().collect();
        assert_eq!(&v, &[node[1]]);

        let v: Vec<DefaultKey> = graph.get(node[1]).unwrap().to().iter().cloned().collect();
        assert_eq!(&v, &[node[2]]);

    }
}
