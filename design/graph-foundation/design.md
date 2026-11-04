# 图基础库设计

实现根：graph-core/、graph-adaptor/、graph-query/、graph-algo/。四库 src 均保持平铺，不建立子目录；壳与配对芯同目录。
壳芯入口：graph-adaptor/src/view.rs、graph-query/src/walker.rs、graph-query/src/search.rs、graph-algo/src/{bfs,dfs,topo}.rs；各自由同目录同名 _inner.rs 承接既有实现。
检查工作目录：`D:/dev/pi_graph`。本次 F1 修复检查（2026-11-04）：`cargo check --workspace --all-targets`、`cargo test --workspace`、`cargo clippy --workspace --all-targets -- -D warnings` 均 exit 0。实际通过 18 个单元测试、136 个集成测试（adaptor_capabilities 119、custom_storage 9、topo_workspace 1、storage 7）及 1 个文档测试，共 155 个，失败/忽略均为 0。独立实现复审已通过，初始 T1–T126 全部闭合，详情与覆盖边界见文末。

## 问题与范围

提供能接入用户自定义存储的图能力、查询与算法，不引入 petgraph。协议四库不内置存储；workspace 已有独立 `graph-storage` 后端，其设计与实现边界单源见 [普通图存储设计](../graph-storage/design.md)，本文只引用，不复制存储内部设计。原始讨论保留于 design/设计草案.md；当前契约以源码声明为准。

当前范围包括壳芯整理及已确认的带身份引用模型升级；保留查找入口路径，调整返回对象协议，不新增图算法。SCC、最短路径和拓扑变更安全遍历不属于当前已实现范围。节点过滤与视图索引转发已实现；受控属性修改仍为协议声明，具体后端由调用方提供。

## 可观察需求与验收

- 自定义、不连续的节点身份可用于遍历和拓扑排序；不要求整数索引。
- 数据访问返回带身份的 NodeRef / EdgeRef，可包装普通借用或 Guard，不要求 Copy/Clone；标准身份迭代与 LendingIterator 独立。
- 反向与边过滤视图可组合，不复制底层图；边过滤保留全部节点。
- Walker 支持暂停后恢复，恢复期间只允许改变业务数据，拓扑与谓词保持稳定。
- 既有跨库集成测试及引用对象验收保持通过，全部目标可编译，Clippy 无警告。
- 读者仅看壳即可定位真实核心状态、字段不变量、方法契约及主流程；除已确认的引用对象协议迁移外，保持既有算法行为与公开路径。

验收用例：graph-algo/tests/custom_storage.rs。节点、边、视图与错误的精确语义只在对应源码壳声明，不在文档复制算法步骤。

## 四层职责与公共导航

用户确认按职责分层，通过公共导航模块组织已有符号；真实声明保持单一来源，原模块与 crate 根导出继续可用，不以重导出代替核心状态定义。

| 层 | 推荐入口 | 内容 |
| --- | --- | --- |
| Core | graph_core::core | GraphBase、引用对象、按 ID 访问、ID 枚举、方向明确的邻接与关联边、ReadGraph |
| Capabilities | graph_core::capabilities | 插入/删除、可写及受控修改、Listener、批次、位置与稠密映射、访问集合、版本、快照 |
| Adaptors | graph_adaptor | EdgeFiltered、NodeFiltered、Reversed 及关联引用/迭代表示 |
| Higher-Level | graph_query、graph_algo | Walker、Project、Search、DFS/BFS、Topo 及工作区 |

Neighbors 用 OutNeighbors / InNeighbors 表达，IncidentEdges 用 OutEdges / InEdges 表达，避免方向歧义。Subgraph 已成为节点/边选择的统一真实结构，NodeFiltered 是带 Predicate 与 AllEdges 策略的便捷名称。MapNodes / MapEdges、通用 Subgraph、MapIds 与 IndexedSubgraph 均已实现；通用 SnapshotView、SCC、ShortestPath 尚未设计或实现，不以空类型冒充能力已存在；Snapshot 的具体视图由后端关联类型提供。

ReadGraph 的唯一声明见 core.rs，组合 NodeAccess、EdgeAccess、NodeIds、EdgeIds、OutEdges、InEdges，并为满足条件的类型自动实现（包括 ?Sized）。它仅是应用便捷约束，不额外承诺邻居枚举、索引、快照或一致性；算法继续使用原来的最小 Trait。

Project 概念属于 Higher-Level，单源声明仍在 graph-core/src/project.rs，graph_query::Project 直接重导出，避免破坏既有使用路径及引入反向依赖。四层是职责边界，目前具体适配器已独立为第四个 crate graph-adaptor，core 不反向重导出适配器。导航重组保持能力边界；已确认实现项的当前状态见文末。

## 模块协作

依赖方向：graph-adaptor → graph-core；graph-query → graph-core；graph-algo → graph-query，并直接使用 graph-core。graph-algo 仅在集成测试 dev-dependencies 引入 graph-adaptor，算法生产代码不依赖适配器。

| 边界 | 壳与职责 | 配对芯 |
| --- | --- | --- |
| 图视图 | graph-adaptor/src/view.rs：真实借用、谓词、迭代状态与能力映射 | view_inner.rs |
| 遍历状态 | graph-query/src/walker.rs：Bfs、Dfs、WalkerIter 状态及主流程 | walker_inner.rs |
| 查询 | graph-query/src/search.rs：可达性及属性搜索契约 | search_inner.rs |
| 批量遍历 | graph-algo/src/bfs.rs、dfs.rs：结果收集入口，引用 Walker 流程 | bfs_inner.rs、dfs_inner.rs |
| 拓扑排序 | graph-algo/src/topo.rs：排序契约、CycleError 与核心流程 | topo_inner.rs |

壳方法只委托配对芯，或在协议适配入口引用唯一壳主体。芯跨模块调用对方壳，不导入或调用其他芯。配对模块私有，实现入口最多 pub(super)，不经 crate 根导出。

## 数据与独立定义

- 核心字段直接定义在壳中：Bfs.queue/visited、Dfs.stack/visited、WalkerIter.graph/walker，以及视图与惰性迭代器字段；字段仍按既有可见性封装。
- CycleError.remaining 保留在 topo.rs，表达失败结果，不隐藏到芯中。
- graph-core/src/{base,access,visit,edit,index}.rs 是能力与共享定义的单一来源。无体 Trait 不人为补默认流程，也不强拆实现芯。
- index.rs 的 HashSet 访问集合适配属于通用集合定义，不承担图遍历主流程，直接委托集合原语。
- 既有拓扑排序临时状态仍在芯中；新增 TopoWorkspace 在 topo.rs 壳侧真实定义节点表、入度表与队列，新增入口已实现。结果缓冲独立返回，不隐藏进跨模块实现类型。

## 引用模型

reference.rs 是 NodeRef / EdgeRef、可写引用协议及通用 NodeReference / EdgeReference 的单一来源。它们属于独立引用定义与原语适配，不承担算法流程；核心身份、端点和数据所有权直接显示在结构中。

NodeAccess / EdgeAccess 仍表示按 ID 从图取得引用的能力，不与已取得引用对象的读取协议混淆。保留关联数据类型 Node / Edge，与引用的 Weight 约束相等；不强制 Deref。调用方从解引用数据迁移为 weight()，写入通过 weight_mut()。Guard 由引用对象拥有，不延长数据借用生命周期。

反向视图的边引用采用 view.rs 中 ReversedEdgeRef，方向必须与 endpoints 一致；双重反向恢复原方向。边过滤保留底层引用，仅在访问前判定可见性。只使用 ID 的算法不增加数据访问约束。

可写引用不允许修改身份和端点，也不保证属性索引维护。受控索引修改已在 mutation.rs 声明协议；结构化查询与 Snapshot 的协议独立声明，既有普通后端由 `graph-storage` 提供，也允许用户存储实现；具体边界见其单源设计。

本轮公共协议调整已获用户确认；现有自定义存储访问实现需迁移为引用对象，源码声明与测试已对齐。

## 只读投影

用户已确认采用最小独立协议，单一声明位于 graph-core/src/project.rs 的 Project<P>，从模块与 crate 根导出。它是无体能力定义，不人为建立实现芯。

P 表示投影类别，输出通过 GAT 表达，可为引用或组合字段 View。协议不依赖 NodeRef / EdgeRef / Weight，也不自动提供包装转发、全局 blanket impl 或泛型便捷扩展方法。调用形式与生命周期契约以源码为准。

本轮只新增协议与文档，不增加业务投影实现。只读类型不匹配用 None；访问错误由取得源对象的边界承担。可写投影仍需独立设计；自动索引维护的能力边界见 mutation.rs，具体后端实现未提供。

独立投影允许后端针对非完整数据对象实现，但当前 NodeAccess / EdgeAccess 仍要求完整 Weight 访问；纯列式后端的查找入口尚未设计，不能将本轮协议视为已完成列式存储接入。

## 既有取舍与限制

优先静态泛型与标准 Iterator + GAT。仅需要逐次借用内部状态时使用 gat-lending-iterator 0.1.8 的 LendingIterator，通过 graph_core::visit 与 crate 根直接重导出，不维护本地同名协议。Walker 默认使用 HashSet，M: VisitMap 的显式集合注入与 with_visit_map 已在 walker.rs / walker_inner.rs 实现，默认 new 也复用同一初始化主体。Visitable 可由调用方显式用于创建集合，GraphVersion 不由 Walker 自动消费。

后端负责身份有效性、邻接与枚举一致性，以及锁和 Guard 的具体限制。拓扑排序对能力契约违例可能 panic；这是既有行为，详见 topo::toposort。跨调用拓扑修改语义仍属于后续独立设计议题。

## Walker 访问集合

Bfs<N, M = HashSet<N>> 与 Dfs<N, M = HashSet<N>> 的真实 visited: M 字段及全部状态仍在 walker.rs 中声明。默认 new 保持 HashSet 行为；with_visit_map(&graph, start, map) 明确消费调用方提供的空集合，可传 graph.visit_map()，不要求所有图实现 Visitable。

自定义集合须精确区分当前图节点身份，索引映射和拓扑在遍历期间稳定。预标记跳过、清空复用、自动后端选择、自动版本校验不在当前设计范围；传入空集合是调用方前提，不执行运行时检测。

next 的唯一主流程继续保留在壳，配对芯通过 VisitMap 原语访问集合；两个 with_visit_map 构造已实现 BFS 起点入队即标记与 DFS 压栈后延迟标记。`graph-storage/tests/storage.rs` 与 `graph-storage/src/lib.rs` 文档示例已消费后端 visit_map；自定义集合的环、平行边及顺序等价仍须按相关验收场景核对，不能由构造已实现推断全覆盖。

`graph-core/src/index.rs::IndexedVisitMap` 按创建时 node_bound 建立清零位图，保留图共享借用以调用 to_index，允许索引空洞。真实入口为 `graph-query/src/walker.rs::{Bfs,Dfs}::new_indexed`，真实查询消费者为 `graph-query/src/search.rs::reachable_indexed`（通过 Bfs::new_indexed）；`graph-algo/tests/custom_storage.rs` 验证稀疏索引与遍历。与默认 HashSet 不同，索引集合在存活期间持有共享借用，暂停时不能取得图独占借用；业务数据修改只能在该借用约束内通过内部可变性等途径进行，拓扑及索引映射仍须稳定。

## 节点过滤与视图索引

节点诱导子图已统一到 subgraph.rs / subgraph_inner.rs；node_filtered.rs 只保留指向真实壳类型的便捷别名和兼容迭代器名称，旧的未实现 node_filtered_inner.rs 已删除；没有芯间依赖。下划线私有字段仍承载实际运行状态，当前配对芯已使用这些字段。

NodeFiltered 沿用底层身份，节点可见性由稳定 ID 谓词决定；可见边的两个端点均须可见。精确的 node / edge / endpoints / 枚举及邻接语义单源保留在壳中。对无效 ID，底层访问能力仍须遵守返回空或 None 的契约；谓词应能处理传入的 ID，不得把谓词返回 true 等同于节点存在。

| 可选能力 | Reversed | EdgeFiltered | NodeFiltered |
| --- | --- | --- | --- |
| NodeIndexable | 保持原映射 | 保持原映射 | 保留上界与位置，隐藏位置返回 None |
| DenseNodeIndex | 保持原映射 | 保持原映射 | 不自动提供 |
| Visitable | 底层全新空集合 | 底层全新空集合 | 底层空集合，可容纳可见节点超集 |

DenseNodeIndex 独立于 NodeIndexable，声明见 index.rs；使用 node_at_dense 命名避免 from_* 的 Rust 方法惯例歧义。节点过滤不拥有重编号缓存；紧凑映射由独立 IndexedSubgraph 设计承接，当前已实现。单次遍历内图、谓词与索引映射均须稳定，GraphVersion 不能单独检测谓词变化。

新增视图及转发能力已由配对芯实现，反向、边过滤和默认遍历入口保持可运行。验收边界包括隐藏起点、隐藏任一端点、全隐藏、孤立节点、自环、平行边、节点/边过滤与反向嵌套，以及各访问入口、原始索引空洞和集合创建一致性；须验证节点过滤不被误声明为 DenseNodeIndex。

## 受控节点属性与监听事件

mutation.rs 定义 NodeProperty、IndexedNodeMut<K>、NodePropertyChanged 与 MutationListener，无体能力和事件结构按独立定义正常组织。属性键及值不强制 Clone/Copy/Eq，不强制拥有完整 Weight；其实际后端、注册管理和索引数据结构由用户存储提供。

数据及相关索引的一致性属于 setter 后端，Listener 是已提交事件的同步观察者；不存在节点、无效属性、验证或索引错误返回 Err 时数据与索引不变且不通知。事件借用的旧值必须在覆盖前由后端保留，不能用新值替代；新值与索引先提交再通知。值相等也按成功设置通知一次，不依赖 Eq。

事件字段、顺序、失败与 panic 契约在源码单源声明。回调注册与注销方式不属于当前统一协议，不把监听器当作索引提交插件，不承诺跨操作事务或可靠消息。共存 NodeMut 的后端仍须保证其索引查询不返回过期结果；维护、失效及查询前重建的具体方式由后端选择。

没有提供生产索引后端或 Listener 实现。本轮协议不代表已完成自动索引功能。后续后端验收至少覆盖成功数据/索引/事件一致、Err 零变更零事件、同值重复通知、非 Clone 旧值、监听器 panic 后已提交状态、直接写入旁路及禁止重入。

## Snapshot 与 Search

已确认采用拓扑一致的快照基线，声明见 graph-core/src/snapshot.rs 的 GraphSnapshot。View 拥有或保留一致读取资源，实际后端提供访问能力；DATA_STABLE 明确是否还固定业务数据。快照获取可失败，不提供默认共享引用实现。仅拓扑稳定不能保证属性查询或过滤的可重复结果，调用方捕获的可变谓词状态也不由快照固定。

独立查询位于 graph-query/src/query.rs 的 NodeSearch<Q> / EdgeSearch<Q>。查询描述由后端类型化定义，执行可选索引或扫描，但匹配语义必须相同。图和查询共用返回迭代器的借用区间；此公共能力不支持逐项 I/O 错误，不提供统一 DSL 或视图自动转发。原 search.rs 中 find_node 仍是普通扫描便捷查询，与后端规划能力独立。

后续后端验收应覆盖快照获取失败释放资源、原图变更下快照拓扑固定、DATA_STABLE 的数据保证、查询借用约束、空结果与执行错误、索引扫描结果等价，以及视图适配不泄漏隐藏元素。协议四库只声明这些能力；workspace 的 `graph-storage` 已提供共享借用 Snapshot（DATA_STABLE=false）及惰性谓词扫描 Search。具体实现和保证见 [普通图存储设计](../graph-storage/design.md) 与 `graph-storage/src/common.rs::impl_graph!`；不由此推断属性索引查询或数据稳定快照已实现。

## 批量拓扑编辑与版本

声明及事件式操作数据在 graph-core/src/batch.rs：TopologyEdit、TopologyEditResult、BatchEditError、AtomicBatchEditError、GraphBatchEdit、AtomicGraphBatchEdit。公开别名 BatchOperation / BatchResults / BatchOutcome / AtomicBatchOutcome 按后端展开关联类型，不隐藏核心结构。

用户已确认首版批次仅引用批前已有 ID，不支持批内新建节点句柄；按序执行时须重新检查被前项删除的元素。普通批次失败保留前缀，原子入口正常失败保持数据、拓扑、索引和版本不变。两个能力均不承诺持久化事务或 panic 回滚。原子能力的提交对读取者不暴露中间前缀，但不替代多次读取所需的快照。

版本为同图、后端定义比较区间内的不透明相等性令牌，不是递增计数器或 Snapshot。普通批次前缀发生实际拓扑变化即须在返回时改变版本；原子成功变更时改变版本；空批次和无操作删除不变。具体边界与失败数据只在源码声明，不依赖属性修改 Listener 表达拓扑事件。

后续后端验收须覆盖空批、级联删除、删除后再引用、任意失败位置的结果前缀、无操作删除、非 Clone 数据、原子提交失败零变更及版本规则。协议四库不提供批次默认执行实现；`graph-storage` 已实现普通批与要求权重 Clone 的原子批，具体保证及尚待补充的异常验收见 [普通图存储设计](../graph-storage/design.md)，真实流程主体为 `graph-storage/src/batch.rs::{edit_batch,edit_batch_atomic}`。

## 拓扑排序 Workspace

新增 TopoWorkspace<N> 和 toposort_with_workspace，声明与真实状态位于 graph-algo/src/topo.rs，配对芯仍为 topo_inner.rs。每次初始化当前图逻辑状态并复用容量，不缓存图身份、版本或跨调用入度；成功和环失败后都允许换图复用。返回独立结果，不承诺结果 Vec 零分配。

new / Default 及工作区排序入口已实现；便捷 toposort 复用同一工作区算法主体。验收边界包括连续运行、换图、节点数增减、空图、环失败后再次运行、稀疏 ID 和无权重能力图。

## 可拥有的组合适配器与数据映射

本轮用户确认迁移固定借用结构：Reversed<'g, G> → Reversed<G>，EdgeFiltered<'g, G, F> → EdgeFiltered<G, F>，NodeFiltered<'g, G, F> → NodeFiltered<G, F>。G 可为 &Storage 或前一适配器；new(&graph, ...) 调用习惯保留，显式类型注解需要迁移。view.rs 的真实 graph 字段仍在壳侧，不通过不透明芯句柄隐藏所有权。

borrowed.rs 逐项为 &G 适配已存在的只读能力，是通用引用表示适配；不通过 GraphView blanket 推断能力，也不自动转发修改。GraphAdaptExt 位于 adapt_ext.rs，配对芯 adapt_ext_inner.rs；消费 self 持有中间层，支持 subgraph、subgraph_with_edges、filter_nodes、filter_edges、reversed、map_nodes、map_edges。组合方法已薄委托对应适配器壳入口。

selection.rs 单源定义 Selection、AllNodes、AllEdges、Predicate、SelectedEdges；集合与闭包通过显式包装避免 coherence 重叠。HashSet 可直接充当策略，借用策略可用 &S；BitSet 等自定义策略由后端实现。成员条件不证明底层存在性，谓词须可处理任意传入身份并在遍历期间稳定。

通用 Subgraph<G, N, E> 的真实字段与惰性候选位于 subgraph.rs。induced 使用 AllEdges，with_edges 包装 SelectedEdges；纯诱导子图直接筛邻居，不提升到底层边能力。显式选边子图用关联边及端点枚举邻居。NodeFiltered 是此真实结构的便捷别名，未隐藏核心数据表示。EdgeFiltered 保留已实现的专用结构与较弱能力约束，语义对应全节点选边子图；不为形式统一而扩大纯边过滤的访问要求。

MapNodes / MapEdges、ValueMapper 及拥有值的 MappedNodeRef / MappedEdgeRef 在 mapped.rs，配对芯 mapped_inner.rs。map 借用完整 Weight 计算独立值，不强制 Clone，不允许借用当前输入逃逸；计算时 Guard 存活，完成后释放，映射引用拥有独立结果。Project 保留借用式表示，不自动成为图级数据映射。映射每次访问重新计算，不自动缓存或获得 Snapshot 属性一致性。

数据适配器保留身份、拓扑、方向、原始和稠密索引及 Visitable，另一类元素数据访问直接透传；只读适配器不提供 GraphInsert/Remove、NodeMut/EdgeMut 或受控 setter，也不自动传播 Search/GraphSnapshot/GraphVersion。任何后续转发均须单独核对闭包捕获状态及可见性。

MapIds 与 IndexedSubgraph 已按下节边界定稿声明，均无自动修改与缓存维护实现。当前没有强制 GraphAdaptor / GraphView 父 Trait。

### 实现边界与验收场景

Reversed 与 EdgeFiltered 已适配所有权结构；Subgraph、Mapped、GraphAdaptExt 及新增索引转发均已有配对芯实现。已删除的旧节点过滤芯全为过期桩，没有丢弃有效实现。

后续验收须包含：仅 OutNeighbors 后端上的诱导子图、显式选边的隐藏双端点、共享引用与拥有中间层的链式类型、非 Clone 按值结果、Guard 释放及映射 panic、反向后映射的端点一致性、多次映射组合、原始/稠密索引与集合能力保持，以及负向编译约束（临时 Guard 引用不得逃逸、只读适配器不暴露修改）。

## MapIds 与 IndexedSubgraph

身份翻译壳芯为 map_ids.rs / map_ids_inner.rs。IdMapping<I> 以关联 External 类型表达双向身份翻译；MapIds 拥有图及节点、边两类映射，不新建存储身份，不隐式过滤。调用方保证可见元素上的完整双射与稳定性；不要求构造时枚举全图验证。无效外部查询正常为空，已有元素无法转换或正反不一致为契约违例，显式 panic，不静默跳过。

翻译引用对象拥有底层引用或 Guard，真实外部身份和方向端点在壳侧字段中表示，weight 不复制。位置索引保持底层位置，反查转换外部 ID。Visitable 仅在节点映射可 Clone 时提供，TranslatedVisitMap 拥有语义等价稳定副本及底层标记，避免访问集合借用图；此 Clone 不得改变映射语义。普通读取无需 Clone。集合操作只接受当前有效外部节点，无法转换时 panic。

用户已明确批准两处迭代泛型约束：`SubgraphNodes` 的 `Iterator` 实现要求 `I::Item: Copy`，用于按值判定选择后保留候选；`TranslatedIds` 的 `Iterator` 实现要求 `I::Item: Copy + Eq`，用于翻译后比较 `mapping.to_internal(external) == Some(原内部ID)`。两者壳芯约束一致；GraphBase 的节点与边身份原本已满足这些条件。T28 与 T35–40 共用翻译入口，每项候选输出前正向缺失、反查缺失或反查不等于原身份均显式 panic，保持惰性、顺序与重复计数，不在构造时遍历验证。

缓存子图壳芯为 indexed_subgraph.rs / indexed_subgraph_inner.rs。IndexedSubgraph::build 以 NodeIds + EdgeIds + EdgeEndpoints 构建固定选择，节点、边谓词各判定一次，之后不保留也不重新调用策略。持久缓存包含可见 ID、端点、双向边及邻居，空间 O(V'+E')；全图完整性验证的临时空间允许 O(V+E)，期望构建时间 O(V+E)（不含策略本身成本）。

构建检测重复节点/边、缺失端点和未知端点，Err 不发布半成品、不修改底层，不保证返还已消费的 G。边及邻接按底层 edge_ids 顺序建立，不保证和原后端专用邻接顺序相同。可见孤立节点保留，全隐藏为合法空缓存。

IndexedSubgraph 使用自有 0..V' 紧凑位置，同时实现 NodeIndexable 与 DenseNodeIndex，不承诺保留底层槽位。Visitable 新建 HashSet，不依赖底层集合。数据访问仍来自底层，结构缓存不等于数据快照；底层拓扑、方向及身份在构建和使用期间均须稳定。内部并发后端应先取得 Snapshot View；无自动版本检查、重建或增量维护。更新选择或结构须重新 build，旧缓存不更新。缓存与原始视图的索引表不能混用。

GraphAdaptExt 新增 map_ids 和 indexed_subgraph 组合入口，仅薄委托各自壳。两种适配器不自动转发写入、Search、Snapshot 或 GraphVersion，不用底层版本推断映射有效性。

MapIds 与 IndexedSubgraph 及组合入口均已实现。验收边界包括：映射双射及无效查询、缺失映射 panic、外部端点一致、Guard 生命周期、索引反查、翻译访问集合与非 Clone 读取；缓存全隐藏/孤立/自环/平行边、选择仅构建时求值、完整性构建错误、自有稠密映射、与反向/映射组合及权重不复制。稳定性前提违例不承诺缓存失效自动检测。

## 四库平铺布局

用户确认将具体适配器迁入 graph-adaptor，并明确全部 src 目录不允许子目录。四库仅使用同目录 .rs 模块，壳与 _inner.rs 配对位置不变。

graph-core 保留身份、引用、访问、枚举、可选编辑/索引/快照、Project、ReadGraph 及共享引用能力适配。graph-adaptor 承载 view、subgraph、selection、mapped、map_ids、indexed_subgraph、adapt_ext 及各配对芯；上述文件的真实路径均为 graph-adaptor/src/<文件名>，此前设计章节的裸文件名按此归属定位。

graph_query::Project 仍指向 core 的单源协议。适配器导入从 graph_core 模块或根迁移为 graph_adaptor；这是导入路径变更，core 不做反向兼容重导出以避免循环依赖。拆库阶段保持现有类型、字段、方法与行为；后续 126 项实现状态见文末。跨库集成测试更新导入并仅在 dev-dependencies 引入适配器。

## 当前验收边界

原实现清单 T1–T126 共 126 项已实现，包括原 graph-adaptor 的 124 项与 graph-algo 工作区的 2 项；四库 src 当前无 `todo!()`。本轮清理已实现芯中的过期实现 TODO 和待实现模块头，保留实际能力限制及后端待验收事项。

本轮仅修复实现审查 F1：T28 原始翻译身份迭代与 T35–40 六类翻译枚举输出前补齐候选往返校验，并同步获准的泛型前提与中文契约。新增 1 个原始迭代器测试和 6 个公共图枚举测试；先在未修复实现上观察 7 个失败，再最小修复。覆盖非单射碰撞、反查缺失和反查指向其他身份，邻接测试起点保持合法；已有未知起点、缺失候选与 Guard 测试保留。

本次检查的实际命令、数量与结果统一见开头。独立实现复审已对照影子基线和初始 T1–T126 全部通过，F1 已关闭，无剩余功能阻塞；实测全目标编译、155 项测试（18 单元、136 集成、1 文档）及严格 Clippy 全部通过。清理复审提示的 topo.rs 过期说明；两处获准泛型前提之外未改变公开契约。该结论不代表发布放行或范围外异常验收完成。

各节功能、Guard/异常路径、索引与组合场景仍为验收边界，不由测试总数推断逐项穷尽覆盖；graph-storage 设计中尚待补充的异常验收及未提供的生产索引后端保持原范围。
