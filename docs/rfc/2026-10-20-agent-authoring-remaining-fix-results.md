# Agent 用户编写接口剩余问题修复与验证记录

日期：2026-10-20\
状态：阶段 A B C D 已实施并完成验证\
关联规划：[剩余问题修复规划](2026-10-20-agent-authoring-remaining-fix-plan.md)\
关联审查：[Agent 用户编写接口审查](2026-10-19-agent-authoring-api-review.md)

本轮处理 DSL 能力映射、通用 agent 装配，以及 Prompt 与历史写入三组问题。实现复用现有 World、AgentHandle、工具注册和 session service，保留前两轮已经交付的权限、审批、轮次与恢复契约。

## 已交付的公共接口

```python
from ecs_agent import AgentDefinition, build_agent, inspect_runtime, tool


@tool()
def total(values: list[int]) -> str:
    """计算整数之和。"""
    return str(sum(values))


agent = build_agent(AgentDefinition(model=model, prompt="准确回答。", tools=[total]))
result = await agent.run_turn("计算 2 加 3", max_steps=10)
if result.success:
    agent.set_prompt("简短回答。")
    snapshot = agent.history_snapshot()
    agent.replace_history(
        [], reason="开始新的上下文",
        expected_history_id=snapshot.history_id,
        expected_revision=snapshot.revision,
    )
print(inspect_runtime(agent).concurrency_groups)
```

新增 `AgentDefinition`、`ModelRef`、`SkillRef`、`RuntimeConfig`、`SystemStep`、`AssemblyReport`、`build_agent()`、`build_agent_from_specs()`、`inspect_runtime()` 和 `restore_runtime()`。构建结果仍为 AgentHandle；没有新增另一套运行手柄或会话存储。

AgentHandle 增加 `set_prompt()`、`inspect_prompt()`、`adopt_prompt()`、`history_snapshot()`、`replace_history()` 和 `reset_session()`。完整用法、可修改时机与恢复步骤见 [agent 编写指南](../features/agent-authoring.md)、[生命周期说明](../features/agent-lifecycle.md) 和 [API 参考](../api-reference.md)。

## 工作包交付

| 工作包 | 交付与实现位置 |
| --- | --- |
| W1 DSL 来源与注册 | [loader 与 compiler](../../src/ecs_agent/dsl/compiler.py) 保留逐 spec 的来源文件；[定义归一化](../../src/ecs_agent/authoring/normalize.py) 检查资源、路径越界、占位符和冲突；skills、普通工具与框架工具统一合并注册 |
| W2 Prompt 缓存 | [渲染系统](../../src/ecs_agent/systems/system_prompt_render_system.py) 将模板、版本、用户值、来源和内置 provider 输出纳入缓存键；用户 callable 每次渲染只求值一次；文件模板采用显式内容快照 |
| W3 共同装配 | [authoring API](../../src/ecs_agent/authoring/api.py) 实现 Python / DSL 共用流程；[系统安装](../../src/ecs_agent/authoring/scheduling.py) 验证具名依赖、优先级和重复安装；报告包含组件及工具来源、执行组、依赖和部分运行时诊断 |
| W4 子 agent 对齐 | [child World](../../src/ecs_agent/systems/subagent/child_world.py) 使用共同安装逻辑；本地 skills、占位符、triggers、权限和嵌套委派落地；自定义 profile 的必要能力和顺序在创建模型前检查 |
| W5 Prompt 编辑 | [编辑与诊断](../../src/ecs_agent/prompts/editing.py) 提供配置提交、缓存及响应链失效；[绑定引用](../../src/ecs_agent/prompts/bindings.py) 支持可恢复的 callable；旧字段迁移对冲突要求明确选择 |
| W6 历史编辑 | [history](../../src/ecs_agent/history.py) 与 [session service](../../src/ecs_agent/sessions/service.py) 保存完整有序快照；内存与 SQLite store 支持事务；替换、重置、writer 失败和提交确认丢失均有恢复验证 |
| W7 checkpoint | [序列化](../../src/ecs_agent/serialization.py) 升级至 schema 4；保存配置、子 agent 定义及输入归一化状态；恢复时校验系统清单、重建绑定和缓存，不重装已保存的历史及轮次 |
| W8 示例与验收 | 迁移 JSON / Markdown DSL、基础委派、Prompt 和会话示例；补充功能文档、API 参考、回归测试及本验证记录 |

基础完整运行时的顺序是归一化、Prompt 渲染、推理、工具执行、错误处理。启用委派后增加委派和等待系统及控制工具；启用压缩时先运行压缩。其他系统通过具名依赖扩展，不依赖同优先级中的注册先后。

`compile_agent_specs()` 继续返回 `(EntityId, World)` 并保留部分装配契约。完整 DSL 示例使用 `build_agent_from_specs()`。工具注册不授予权限，DSL tools 仍表示权限映射，省略、空映射和显式拒绝的语义保持不变。

## 逐项行为验收证据

下表中的修复回归均位于 [test_agent_authoring_remaining.py](../../tests/test_agent_authoring_remaining.py)，其他测试文件另列链接。测试断言覆盖实际 provider 请求、handler 调用、持久化记录或恢复结果。

| 编号 | 可观察结果及验证依据 |
| --- | --- |
| D1 | `test_loaded_skill_keeps_origin_and_tools_merge` 在 JSON / Markdown 加载后注册普通工具，实际调用 skill details 和普通 handler；框架工具重复安装及外部重名由 `test_framework_collision_and_repeat_binding_are_explicit` 验证 |
| D2 | `test_multiple_dsl_origins_resolve_same_relative_skill_independently` 从两个目录加载相同相对路径，分别得到 alpha / beta skill 并实际委派调用；缺失基准目录、资源不存在和 symlink 越界均有失败测试 |
| D3 | `test_dsl_child_fields_reach_actual_delegated_request` 捕获子模型真实请求，验证占位符、归一化输入及 skill 工具结果；父实体没有被安装子 agent 的本地 skill |
| D4 | [工具授权回归](../../tests/test_tool_authorization.py) 验证 DSL 权限、动态工具、父子权限合并与实际拒绝次数；共同安装器复用相同权限编译和执行入口 |
| A1 | `test_python_and_dsl_build_equivalent_skill_requests` 比较实际消息、工具 schema 和系统顺序；工具执行、skills 和委派分别通过完整入口回归 |
| A2 | 原有 [DSL 编译测试](../../tests/test_agent_dsl_compiler.py) 与集成测试保持部分装配用法；`test_complete_builder_executes_tools_without_manual_systems` 不手动补系统即完成工具循环 |
| A3 | 缺少角色、依赖环、递归定义、重名和重复安装均有回归；静态失败测试断言 model_factory 未被调用；安装后 session 失败测试断言 skill 资源卸载、session catalog 回滚 |
| A4 | `test_script_trigger_updates_phase_before_render` 验证当轮请求使用新 phase；`test_named_runtime_dependencies_and_repeat_installation` 验证先写后读且只执行一次；[plan-and-task 集成](../../tests/integration/test_plan_and_task_flow.py) 和 [replanning 测试](../../tests/test_replanning.py) 保持结果依赖语义 |
| A5 | 实际同步委派、`test_complete_builder_background_wait_reaches_child_result` 的后台启动 → 等待 → 读取结果，以及嵌套自定义 profile 委派全部覆盖 |
| P1 | 模板和变量从 Hello Ada 改为 Goodbye Bob 后，实际渲染与下一次请求使用新内容；配置持久化失败保留旧配置和请求 |
| P2 | 动态回调次数、宿主绑定方法所有权、文件修改后未 reload 保持旧快照、显式 reload 更新及 callable 持久化绑定均有测试 |
| P3 | [渲染系统测试](../../tests/test_system_prompt_render_system.py) 覆盖 phase、摘要及 scratchbook；[缓存前缀审计](../../tests/test_prompt_cache_prefix_audit.py) 检查稳定前缀、动态尾部与连续请求的字节行为 |
| P4 | `test_legacy_prompt_adoption_requires_explicit_conflict_resolution` 验证旧 template / LLM 字段冲突、显式迁移及实际请求；文件 Prompt 测试验证诊断不执行回调；诊断实现只读取组件快照，不消费队列或运行 trigger |
| H1 | 修改快照内消息和嵌套图片 URL 不影响运行状态，替换后修改调用者的原列表也不影响历史 |
| H2 | 历史 ID / revision 乐观检查拒绝过期编辑；完整替换记录可由 `replay_history()` 重建；成功后清除旧用户渲染和 Responses 链 |
| H3 | 内存与 SQLite 均验证历史记录提交失败、提交确认丢失、新 writer 失败、reset 最终记录失败和事务提交确认丢失；失败保持旧可见状态，已确认提交恢复到新状态 |
| H4 | paused 轮次与 blocked 的未知工具结果拒绝 Prompt、历史及 reset 操作；未知调用 ID 与历史保留；[AgentHandle 测试](../../tests/test_agent_handle.py) 继续覆盖活动驱动、取消与工具协议 |
| H5 | 内部追加上下文后恢复不再次执行 script；后台通知不会被旧用户渲染覆盖；[session 运行](../../tests/test_session_runtime.py)、[压缩](../../tests/test_session_compaction.py) 和 AgentHandle 回归验证记录顺序、结果归属及工具不重放 |
| R1 | 实际 JSON 文件保存加载后，新的 Prompt、root / child skills、子占位符和 triggers 继续生效；自定义系统缺失时拒绝恢复；审批重审继续由 [checkpoint 授权测试](../../tests/test_tool_authorization.py) 验证 |
| R2 | AgentHandle 的多实体范围、单一驱动、取消、流式提前退出及 child World 测试继续通过；新装配不维护另一份轮次状态 |
| R3 | 内存与 SQLite 的旧快照分支测试确认 script 只执行一次；schema 3 缓存匹配可恢复，缺失或属于另一轮的证据则阻止恢复；未绑定 session writer 时也拒绝驱动和编辑 |

## 验证中补充修复的组合问题

后台等待会追加一条 user 类型的完成通知。旧消息组装直接把归一化结果替换到最后一条 user 消息，导致通知被原始输入覆盖。现在仅替换索引及指纹匹配的来源消息，保留其他通知与消息元数据。

新创建的后台任务原先没有标记为当前 runtime 已接管，下一 tick 会被恢复协调逻辑误判为丢失 task handle。入队后现在登记该所有权；完整后台等待测试验证最终结果能够被父 agent 读取。

全量验证还发现 TUI 卸载期间可能只剩状态栏容器、标签已移除。状态更新现在先检查全部所需节点；新增部分卸载测试验证定时器不会因标签缺失而报错。

## 最终检查与真实接口

| 检查 | 最终结果 |
| --- | --- |
| 本地全量测试（Python 3.12，已安装 MCP 的开发环境） | **3407 passed，132 skipped，4 xfailed，6 warnings**；54.47 秒 |
| 严格类型检查 | **183 个源码文件全部通过** |
| DeepSeek Responses API | **19 passed**；44.74 秒 |
| 迁移示例 | 5 个离线示例通过 subprocess 实际运行，纳入全量测试；多轮与磁盘 checkpoint 另有集成测试 |
| 文档 | 规划、审查、结果记录和编写指南的本地链接、代码块检查通过；21 项验收编号均对应证据 |
| `git diff --check` | 通过 |

```bash
uv run pytest -q --tb=short --maxfail=3
uv run mypy src/ecs_agent/
git diff --check
```

132 个跳过项来自环境条件或显式 opt-in；上述 19 项真实接口用例另行启用执行。4 个既有 native conversation tree 预期失败保持原断言，未计为通过。6 条 warning 是既有 structlog 配置提示。验证期间发现的后台通知、任务归属和 TUI 卸载问题均已修复后重新运行全量检查。

真实接口使用 `https://api.deepseek.com`、`deepseek-flash`、`ApiFormat.OPENAI_RESPONSES`，覆盖以下 19 项：

- [本轮真实接口测试](../../tests/live/test_agent_authoring_remaining_live.py)：3 项，验证流式 / 非流式完整工具循环、Prompt 与历史更新后的新请求、session reset，以及 DSL 子 agent 的实际 skill 调用、占位符和 trigger。
- [首轮 P2 回归](../../tests/live/test_agent_authoring_p2_live.py)：6 项，验证嵌套参数类型、默认值、预算恢复、多轮归属及权限 / 审批拒绝。
- [P1 授权回归](../../tests/live/test_tool_authorization_live.py)：10 项，验证允许、权限拒绝、审批拒绝、审批入口和 DSL 禁用，均包含流式与非流式。

可复现的 opt-in 命令如下，`LLM_API_KEY` 由宿主进程环境提供：

```bash
ECS_AUTHORING_LIVE=1 ECS_P2_LIVE=1 ECS_P1_LIVE=1 \
LLM_BASE_URL=https://api.deepseek.com LLM_MODEL=deepseek-flash \
uv run pytest -q --tb=short \
  tests/live/test_tool_authorization_live.py \
  tests/live/test_agent_authoring_p2_live.py \
  tests/live/test_agent_authoring_remaining_live.py
```

凭据仅经不回显输入传给测试进程，测试输出在展示和临时保存前脱敏；未写入代码或文档。工具效果限于内存函数与临时 skill 文本读取。

## 迁移与保留边界

- 基础用法采用完整构建入口；手动 World 和部分 DSL 编译仍可使用，但需要调用方完成系统组合。相同 priority 表示并发，扩展系统必须显式声明依赖。
- Prompt 文件不自动监视。需要更新时重新提交配置；动态 callable 通过稳定 PromptBinding 恢复，不能依赖序列化函数对象。旧 SystemPromptComponent 的 template 优先作为迁移输入，content 作为兼容回退。
- 公共编辑只在已结束或已安全放弃的轮次间进行。未知工具或 script 副作用不因编辑、重置、诊断或 checkpoint 恢复而消失。
- provider、session service、script handler 和 phase graph 由宿主提供并保持原有所有权。`restore_runtime()` 只重新绑定运行能力，不覆盖已恢复组件；自定义系统需重新提供原计划，自定义 skill 需提供匹配的绑定适配器。
- schema 4 保存归一化状态；旧格式证据不足时明确阻止 script 重放。历史上仅保存差量、无法重建完整顺序的替换记录仍需配合其 checkpoint，`replay_history()` 不猜测丢失内容。
- ConversationComponent 的低层可变接口保留。内置运行路径使用受控写入；兼容的无 World 上下文 helper、独立 message bus 历史和 parent child stub 的只读投影不冒充新的持久化写入入口。
- 原生 conversation tree 写入缺陷、同 World 多请求调度器、工具热替换和任意既有实体通用补装仍不在本轮范围。新入口继续拒绝未支持的原生树配置，相关预期失败不计为通过。

此前 [P1](2026-10-19-agent-authoring-p1-fix-results.md) 与 [首轮 P2](2026-10-20-agent-authoring-p2-fix-results.md) 的测试数量保留为各轮历史记录。本文件记录本轮工作区的最终验证，不覆盖其历史数据。

## 2026-10-21 CI 环境复验

提交 `ef6b318` 的 [CI 运行](https://github.com/MoveCloudROY/ecs-agent/actions/runs/39463706804) 实际结果为 **2 failed，3403 passed，134 skipped，4 xfailed**。此前本地通过结果未覆盖 CI 的依赖环境：CI 使用 Python 3.11 和基础 dev 依赖，不安装可选 MCP SDK。

两项失败分别是缺少 MCP 时仍断言旧版导入错误文案，以及真实 MCP 服务测试未检查 SDK 是否存在。修复保留惰性导入的安装指引；缺失依赖测试改在独立子进程中模拟 SDK 不可用，验证基础导入、全部四个 MCP 公共符号及明确的安装错误，避免本机安装 MCP 后跳过此项。

真实 SDK 测试仅在 MCP 未安装时跳过，已安装但导入损坏时仍会失败。CI 增加基础依赖和 MCP 两组任务，均执行完整 pytest；MCP 任务实际执行连接、工具发现、调用及断开测试。两组都使用锁定依赖，pytest 通过 `uv run --no-sync` 复用各自已同步的环境。

修复后本地全量复验：

| 环境 | 结果 |
| --- | --- |
| Python 3.11.16，独立基础 dev 环境 | 3405 passed，134 skipped，4 xfailed；60.91 秒 |
| Python 3.11.16，独立 dev + MCP 环境 | 3406 passed，133 skipped，4 xfailed；56.93 秒 |
| Python 3.12.12，原开发环境 | 3408 passed，131 skipped，4 xfailed；54.74 秒 |

三组均没有失败项，均有 6 条既有 structlog warning。MCP 组比基础组多通过的用例是实际 SDK 的内存服务往返测试；各环境的跳过数量取决于可选依赖，4 个既有 native conversation tree 预期失败保持不变。此次只调整测试隔离、可选依赖测试边界及 CI 覆盖，没有将 MCP 变成基础运行依赖。

复验期间还发现 `uv.lock` 原先被 `.gitignore` 排除，导致启用 `--locked` 后的首个远端运行在安装阶段停止。现将上述测试实际使用的锁文件纳入版本控制，让干净 checkout 和本地复验使用相同依赖版本。

使用 `git checkout-index` 导出仅含待提交文件的干净目录后，两组 `uv sync --locked` 均成功；对应 MCP 定向测试分别为基础组 42 passed、1 skipped，MCP 组 43 passed。
