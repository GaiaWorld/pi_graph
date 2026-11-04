use graph_algo::topo::TopoWorkspace;

// 公共 seam：工作区 new / Default，允许不具备图身份约束的类型。
#[test]
fn creates_empty_workspace_without_id_bounds() {
    struct NotAnId;
    let _: TopoWorkspace<NotAnId> = TopoWorkspace::new();
    let _: TopoWorkspace<NotAnId> = TopoWorkspace::default();
}
