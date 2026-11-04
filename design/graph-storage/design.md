# 普通图存储设计

实现根：`graph-storage/`。壳芯入口：`src/sparse.rs` ↔ `sparse_inner.rs`、`src/dense.rs` ↔ `dense_inner.rs`、`src/batch.rs` ↔ `batch_inner.rs`；共享独立定义和宏协议入口为 `src/common.rs`。
检查工作目录：`D:/dev/pi_graph`。转换前实际运行 `cargo check --workspace --all-targets`、`cargo test --workspace`、`cargo clippy --workspace --all-targets -- -D warnings` 均退出 0；14 个集成测试、1 个文档测试通过。转换后实际执行同三条命令均退出 0，仍为 14 个集成测试与 1 个文档测试通过、Clippy 零警告。独立只读 convert 轻审已通过，范围为本库与本文档，核对六份转换前快照；无阻塞发现。此结论为保持行为的转换审查，不代表用户已确认设计满意。
转换前六份只读快照：`C:/Users/zmax/AppData/Local/Temp/opencode/graph-storage-zcode-baseline`；转换开始时均与原文件字节一致，保留作为行为比对证据。

## 问题与范围

为 graph-core 的能力协议提供两种已有普通有向多重图存储。此次只整理现有实现的设计表达和壳芯归属，允许自环、平行边；不增加算法、业务或新公共能力，不修改其他 crate。基础分层参考 `design/graph-foundation/design.md`，生产存储的精确契约在本库源码。

## 可观察需求与验收

- 保持 crate 根和 dense/sparse 模块全部既有公共符号、签名、可见性及 trait 关联类型。
- SparseGraph 的 SlotMap 代际身份在其他元素删除时不变；紧凑枚举表末项填洞，槽位反查保存完整代际身份。
- DenseGraph 的位置和代际共同组成身份；swap_remove 搬移使末项旧位置身份失效，修复双向邻接和端点；未搬移元素身份不变。
- 两者保留邻接插入顺序、平行边重复和自环关联；权重修改不递增拓扑版本，失效删除幂等，业务错误发生前不修改图。
- 普通批保留成功前缀，原子批仅全部成功后发布；本批引用按批前身份解释，双向映射仅跟踪输入引用项，不扫描整图。
- Snapshot 为共享借用，固定拓扑而 DATA_STABLE=false；Search 为惰性谓词扫描，Visitable 为 FixedBitSet。
- 保持权重 Clone/Drop、闭包 panic、分配行为和操作/销毁顺序；原子批不提供 panic 回滚或内部共享权重隔离。

已有验收单源：`graph-storage/tests/storage.rs` 的 7 个测试，加上工作区跨存储集成与 lib.rs 文档示例。此次保留全部原断言与测试文件。

## 壳、核心状态与依赖

| 壳/独立定义 | 真实状态和职责 | 配对实现 |
| --- | --- | --- |
| sparse.rs | SparseGraph、Node、Edge、代际键与 SparseEdges；共享 DequeStorage、紧凑身份表及 slots 反查 | sparse_inner.rs |
| dense.rs | DenseGraph、Node、Edge、位置代际 ID、DenseNodes/DenseEdges；Vec 与有序双向邻接 | dense_inner.rs |
| batch.rs | 普通/原子批核心编排契约；BatchStorage 为内部最小存储协议，不新增公共接口 | batch_inner.rs |
| common.rs | GraphError、位图、邻居/查询迭代、BatchIds 四张真实双向映射、impl_graph! 公共协议实现 | 无需强拆通用结构 |

所有生产字段保持原类型和可见性，主结构不包装 inner，不用别名或重导出隐藏状态。配对芯只接收真实壳对象；batch_inner 通过 BatchStorage 的真实存储壳调用两种实现，其他芯只调用本类型壳及 common 独立定义；芯间无依赖。索引 trait 仍在各存储壳真实实现，算法体迁配对芯。

核心伪码只写在真实壳方法：各存储 add_node/add_edge/unlink_edge/delete_edge 和主要删节点入口、稀疏 Clone、批次 edit_batch/edit_batch_atomic。公共协议适配引用这些主体，文档不再复制算法。

## 既有取舍与边界

稀疏插入和单边删除为 O(1)，删节点为 O(关联边数)；稠密删除需要保序扫描端点邻接并修复搬移节点的关联端点。稀疏克隆重建独立链接池，保持完整代际身份和邻接顺序。两种克隆的精确成本分别见源码 SparseGraph、DenseGraph 的类型契约：稀疏成本包含历史空槽，不能只按活节点/边数计算；原子批另外包含准备与执行操作的成本，权重 Clone 成本另计。全无操作删除不替换原权重对象，但非空批仍需克隆。ID 只属于创建它的图，跨图误用不提供额外防护；访问位图仅在同一稳定拓扑内有效。稀疏键继承 pi_slotmap 有限代际周期，生命周期边界见 sparse.rs::SparseGraph，不承诺回绕后旧键仍失效。

行为比对：54 个迁移后的存储实现体（稠密 28、稀疏 26）与快照在接收者/构造类型映射及空白注释归一化后逐体一致；两个批次实现保持原控制流，仅适配内部协议和版本访问。Cargo.toml 与 tests/storage.rs 与快照字节一致。三个芯无其他芯引用，无未实现桩。

## 异常与生命周期验收设计

以下是设计审查补齐的待实现验收，不表示现有 7 个存储测试已覆盖。既有公共正常路径测试继续保留；不为每个私有委托方法建立镜像测试。

| 风险 | 验收落点与预期观察 |
| --- | --- |
| 版本耗尽 | 在 dense.rs、sparse.rs 的模块内测试设置私有 version=u64::MAX，再调用公开 GraphInsert/GraphRemove/GraphBatchEdit：有效插入/删除返回 VersionExhausted，拓扑、权重和版本不变；失效删除仍为 Ok(false)；无效端点优先 MissingEndpoint；普通批保留成功前缀，原子批错误不提交。耗尽边界由这些模块内状态测试补充，不暴露生产测试开关。 |
| 非空全无操作原子批 | tests/storage.rs 经 AtomicGraphBatchEdit，使用可辨识 Clone 的权重，证明调用可克隆但不替换源权重对象，版本及拓扑不变。 |
| 删除后 Drop panic | tests/storage.rs 经 GraphRemove 及普通批，权重只单次 panic；catch_unwind 成功展开后验证删除已提交、版本改变、全部剩余端点及双向关联自洽，能继续读写；普通批不返回成功前缀结果。 |
| 原子批 Clone panic | tests/storage.rs 使用受控 Clone panic 权重；成功展开后源图拓扑与版本未变、可继续读写；不把业务共享状态副作用当回滚。 |
| 发布后旧图 Drop panic | tests/storage.rs 经原子批，旧权重单次析构 panic；成功展开后验证新图及版本已提交、邻接完整且能继续使用。二次 panic/进程终止无恢复承诺。 |
| 非 Clone 权重 | tests/storage.rs 的编译及运行用例以不实现 Clone 的权重执行单项增删、普通批，证明只有原子批要求 Clone。 |

panic 的分阶段结构保证以 batch.rs 的两个真实批次入口为准；分配失败、不变量违例或其他未承诺恢复的阶段不允许假定图可继续使用。上述验收由后续实现/测试工作承接，本轮只落契约与验收设计。

本轮设计审查清理 BatchIds::remove_node/remove_edge 的无消费者搬移参数；搬移使用 relocate_node/relocate_edge 完整身份接口。未更改公共 API，无阻塞设计待定。独立 design 复审已通过，四项原阻塞均闭合；复审后工作区 cargo check --workspace --all-targets、cargo test --workspace（14 集成测试、1 文档测试）、cargo clippy --workspace --all-targets -- -D warnings 均退出 0。上述新增验收设计尚未实施，现有测试通过不表示这些风险已验证。
