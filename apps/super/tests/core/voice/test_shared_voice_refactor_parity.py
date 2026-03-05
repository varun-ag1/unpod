import ast
from pathlib import Path


SHARED_MIXIN_PATH = Path("super/core/voice/shared_mixin.py")
LIVEKIT_HANDLER_PATH = Path("super/core/voice/livekit/lite_handler.py")
PIPECAT_HANDLER_PATH = Path("super/core/voice/pipecat/lite_handler.py")


def _get_class_methods(path: Path, class_name: str) -> dict[str, ast.AST]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods: dict[str, ast.AST] = {}
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[node.name] = node
    return methods


def _nested_async_fn(parent: ast.AST, fn_name: str) -> ast.AsyncFunctionDef:
    assert isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
    for node in parent.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == fn_name:
            return node
    raise AssertionError(f"Nested async function not found: {fn_name}")


def _await_call_name(stmt: ast.stmt) -> str | None:
    if not isinstance(stmt, ast.Expr):
        return None
    if not isinstance(stmt.value, ast.Await):
        return None
    if not isinstance(stmt.value.value, ast.Call):
        return None

    func = stmt.value.value.func
    if not isinstance(func, ast.Attribute):
        return None

    if isinstance(func.value, ast.Name) and func.value.id == "self":
        return func.attr

    if (
        isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "self"
    ):
        return f"{func.value.attr}.{func.attr}"

    return None


def _collect_ordered_await_calls(stmts: list[ast.stmt], out: list[str]) -> None:
    for stmt in stmts:
        call_name = _await_call_name(stmt)
        if call_name:
            out.append(call_name)
            continue

        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            _collect_ordered_await_calls(stmt.body, out)
            _collect_ordered_await_calls(stmt.orelse, out)
            continue

        if isinstance(stmt, ast.Try):
            _collect_ordered_await_calls(stmt.body, out)
            for handler in stmt.handlers:
                _collect_ordered_await_calls(handler.body, out)
            _collect_ordered_await_calls(stmt.orelse, out)
            _collect_ordered_await_calls(stmt.finalbody, out)


def _ordered_await_calls(fn_node: ast.AST) -> list[str]:
    assert isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef))
    calls: list[str] = []
    _collect_ordered_await_calls(fn_node.body, calls)
    return calls


def test_shared_idle_monitor_uses_agent_speaking_hook() -> None:
    methods = _get_class_methods(SHARED_MIXIN_PATH, "SharedVoiceMixin")
    idle_loop = methods["_idle_monitor_loop"]
    assert isinstance(idle_loop, ast.AsyncFunctionDef)

    has_hook_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_is_agent_speaking"
        for node in ast.walk(idle_loop)
    )
    assert has_hook_call


def test_pipecat_set_agent_state_updates_local_state() -> None:
    methods = _get_class_methods(PIPECAT_HANDLER_PATH, "LiteVoiceHandler")
    set_state = methods["set_agent_state"]
    assert isinstance(set_state, ast.AsyncFunctionDef)

    updates_local_state = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == "_agent_state"
            for target in node.targets
        )
        for node in ast.walk(set_state)
    )
    assert updates_local_state


def test_pipecat_disconnect_broadcast_happens_before_cleanup() -> None:
    methods = _get_class_methods(PIPECAT_HANDLER_PATH, "LiteVoiceHandler")
    setup_events = methods["_setup_transport_events"]
    disconnect_fn = _nested_async_fn(setup_events, "on_participant_disconnected")
    awaited = _ordered_await_calls(disconnect_fn)

    assert "plugins.broadcast_event" in awaited
    assert "_cleanup_runtime_resources" in awaited
    assert awaited.index("plugins.broadcast_event") < awaited.index(
        "_cleanup_runtime_resources"
    )


def test_pipecat_end_call_broadcast_happens_before_cleanup() -> None:
    methods = _get_class_methods(PIPECAT_HANDLER_PATH, "LiteVoiceHandler")
    end_call = methods["end_call"]
    awaited = _ordered_await_calls(end_call)

    assert "plugins.broadcast_event" in awaited
    assert "_cleanup_runtime_resources" in awaited
    assert awaited.index("plugins.broadcast_event") < awaited.index(
        "_cleanup_runtime_resources"
    )


def test_livekit_has_active_kb_auto_inject_trigger() -> None:
    methods = _get_class_methods(LIVEKIT_HANDLER_PATH, "LiveKitLiteHandler")

    trigger_call_found = False
    for name, method in methods.items():
        if name == "_auto_inject_kb_context":
            continue
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "_auto_inject_kb_context"
            for node in ast.walk(method)
        ):
            trigger_call_found = True
            break

    assert trigger_call_found


def test_pipecat_data_response_uses_unified_publish_helper() -> None:
    methods = _get_class_methods(PIPECAT_HANDLER_PATH, "LiteVoiceHandler")
    assert "_publish_data_to_client" in methods

    send_data = methods["_send_data_response"]
    uses_helper = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_publish_data_to_client"
        for node in ast.walk(send_data)
    )
    assert uses_helper
