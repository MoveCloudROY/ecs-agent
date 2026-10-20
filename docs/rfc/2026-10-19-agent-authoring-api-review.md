# Agent 用户编写接口审查

日期：2026-10-19\
状态：P1 与两轮 P2 已完成修复和验证\
范围：当前工作区的公共 API、`examples` 和相关运行时实现，包含审查时尚未提交的修改。

P1 修复设计见 [Agent 用户编写接口 P1 问题解决方案](2026-10-19-agent-authoring-p1-fix-plan.md)，实现与测试结果见 [P1 修复与验证记录](2026-10-19-agent-authoring-p1-fix-results.md)。以下问题描述保留审查时的行为。

P2 中运行与会话生命周期、工具类型映射和注册接口两项已完成首轮实施。设计见 [P2 运行生命周期与工具接口改进规划](2026-10-20-agent-authoring-p2-fix-plan.md)，实现和验收证据见 [首轮 P2 修复与验证记录](2026-10-20-agent-authoring-p2-fix-results.md)。以下问题描述继续保留审查时的行为。

其余 DSL 能力映射、通用 agent 装配及 Prompt 与历史写入问题已完成实施。设计见 [剩余问题修复规划](2026-10-20-agent-authoring-remaining-fix-plan.md)，逐项证据、兼容边界及 DeepSeek Responses API 结果见 [剩余问题修复与验证记录](2026-10-20-agent-authoring-remaining-fix-results.md)。下文继续保留原始审查的问题描述。

当前接口把较多运行时装配责任交给了 agent 作者，部分配置错误会静默变成错误行为。建议保留 ECS 的组合能力，优先修复权限、DSL 和运行生命周期问题，再补充统一装配、配置校验与面向 agent 的运行接口。

本次覆盖基础聊天、工具调用、权限与审批、DSL、子 agent、流式和 `plan_and_task` 示例。行为缺陷使用 FakeModel 或无外部依赖的最小 World 复现；设计问题基于源码和示例审查。已有 `AgentSpec`、`@tool`、`RunResult` 和 child runtime profile 可作为改进基础。

## 问题优先级

| 问题 | 优先级 | 主要影响 | 依据 |
| --- | --- | --- | --- |
| 权限与审批依赖手写系统顺序 | P1 | 被禁止或未获批准的工具仍可能执行 | 行为复现 |
| DSL 声明与实际能力不一致 | P1 / P2 | 禁用配置失效、skill 工具丢失、子 agent 配置未落地 | 行为复现与源码审查 |
| 运行与会话生命周期边界不清晰 | P2 | 上次运行的终止状态影响下一次运行 | 行为复现与 API 审查 |
| 工具类型映射和注册接口不完整 | P2 | schema 与 Python 签名不一致，用户依赖私有属性 | 行为复现与示例审查 |
| 常用能力需要过多手动装配 | P2 | 缺失组件、系统或错误顺序造成能力失效 | API 与示例审查 |
| Prompt 与会话存在多个写入入口 | P2 | 修改不生效，直接改历史绕过持久化 | 源码与 API 审查 |

P1 表示应优先修复的行为正确性问题；P2 表示随后处理的运行契约、类型支持和作者体验问题。

## 权限和审批的正确性依赖用户手写系统顺序

[权限示例](../../examples/permission_agent.py#L76) 使用 `Permission(-10) → ToolExecution(5)`，并提前创建 `PendingToolCallsComponent`。将这个配置与常见的 `Reasoning(0)` 组合后，顺序变为：

```text
Permission(-10) → Reasoning(0) → ToolExecution(5)
```

权限检查时还没有待执行调用；Reasoning 在同一 tick 中产生调用，随后工具直接执行。[审批示例](../../examples/tool_approval_agent.py#L144) 也把 Approval 放在 Reasoning 前面，存在相同问题。

实际复现结果：

- 设置 `PermissionComponent(denied_tools=["blocked"])`，采用上述顺序，`blocked` handler 仍被调用。
- 将 Permission 调整到 priority 1，保持 Reasoning 为 0、Execution 为 5，同一调用被拦截。
- 设置 `ApprovalPolicy.ALWAYS_DENY`，采用 `Approval(-5) → Reasoning(0) → Execution(5)`，handler 仍被调用。

权限示例预先写入 pending calls，无法暴露真实 reasoning 循环里的这一缺陷。执行入口目前也没有对权限与审批做最终核验。

建议默认流水线保证 `Reasoning → Permission → Approval → Execution`，并将权限与审批纳入工具执行契约。执行入口应核验当前调用的策略与审批状态，避免漏注册系统或错误排序绕过限制。配置检查应在运行前报告缺失的检查阶段。

验收应覆盖模型当轮生成调用、拒绝、允许、审批缺失，以及系统排序错误等场景，并断言 handler 是否实际执行。

## DSL 声明与实际生效的能力不一致

### 禁用工具的信息在编译时丢失

[权限编译逻辑](../../src/ecs_agent/dsl/compiler.py#L108) 仅收集值为 `True` 的工具，始终设置 `denied_tools=[]`。[PermissionSystem](../../src/ecs_agent/systems/permission.py#L28) 又把空的允许名单解释为“不限制”。

因此，`tools={"blocked": False}` 编译后得到两个空名单。即使把 Permission 正确安排在 Reasoning 之后，被禁用的工具仍会执行；该行为已复现。

编译器还没有注册 PermissionSystem，DSL 示例补齐的运行系统也不包含它。因此，仅生成 PermissionComponent 不能保证策略生效。

建议明确区分“未设置限制”和“允许集合为空”，保留显式拒绝项，并由统一装配逻辑安装或强制执行相应策略。应定义工具映射的默认行为，避免用户猜测遗漏条目的含义。

### 安装 skill 后覆盖工具注册表

[DSL 编译器](../../src/ecs_agent/dsl/compiler.py#L123) 先调用 `SkillManager.install()`，随后无条件添加新的空 `ToolRegistryComponent`，覆盖安装期间创建的注册表。

复现中，`SkillComponent` 仍包含 `demo` skill，最终工具注册表却为空，`load_skill_details` 已丢失。仅检查 skill 元数据会误判安装成功。

建议在安装 skills 前初始化注册表，后续只复用或合并，明确禁止装配过程中的意外覆盖。声明了 skills 但缺少 `source_dir` 时，目前只记录警告并跳过；应保留来源目录或明确报错。

### 子 agent 的部分声明没有落地

[subagent 编译](../../src/ecs_agent/dsl/compiler.py#L145) 只映射 `name`、`model`、`prompt`，没有映射 spec 接受的 `tools`、`skills`、`placeholders`、`triggers`。

建议为每个 DSL 字段定义明确的运行时映射。不支持的字段组合应在校验阶段报错。Python 配置与 DSL 应共用装配逻辑，并验证主 agent 和子 agent 的能力一致性。

## 一次运行和长期会话的生命周期边界不清晰

[Runner.run](../../src/ecs_agent/core/runner.py#L44) 在没有指定 `target_entity` 时，会被任意非 child stub 实体的 TerminalComponent 结束。达到 tick 上限时，[Runner](../../src/ecs_agent/core/runner.py#L123) 还会新建一个带 `reason="max_ticks"` 的终止实体。

在同一个 World 上，使用每轮只递增计数、不会自行终止的 system，复现结果如下：

| 调用 | 设定预算 | 实际执行 tick 数 | 结果 |
| --- | ---: | ---: | --- |
| 第一次 `run` | 1 | 1 | `max_ticks` |
| 第二次 `run` | 5 | 1 | 被第一次遗留的终止标记结束 |

这使“继续运行”依赖调用者清理内部状态。多实体场景还要求作者理解全局终止和目标实体终止的区别。

交互式示例需要自行订阅完成与错误事件、重新挂载 UserInputComponent，并注册 TerminalCleanupSystem，才能持续接收用户输入。参见 [交互运行装配](../../examples/e2e/plan_and_task/runtime.py#L255)。

框架已经有 [RunResult](../../src/ecs_agent/core/run_result.py#L39)，包含 `status`、`final_message`、`partial_message` 和 `error`，但基础示例普遍忽略返回值，再遍历 conversation 判断结果。参见 [工具示例](../../examples/tool_agent.py#L162)。

建议：

- 明确单轮执行、恢复执行和持续会话的契约，并由 agent 级入口自动绑定目标实体。
- 将运行预算耗尽保留在本次 RunResult 中，避免遗留全局终止实体影响下一次调用。
- 将多轮输入、取消、恢复和流式事件适配纳入公共运行接口。
- 示例统一处理 RunResult，明确区分完成、部分输出、错误和预算耗尽。

验收应覆盖预算耗尽后继续运行、连续两轮输入、多实体独立完成，以及取消后恢复。

## 工具类型映射和注册接口不完整

`@tool` 已提供自动生成 schema 的基础，但[类型映射](../../src/ecs_agent/tools/discovery.py#L14) 对不支持的注解统一返回 `string`。实际生成结果包括：

| Python 参数注解 | 当前 schema | 缺失语义 |
| --- | --- | --- |
| `list[int]` | `{"type": "string"}` | array 及 integer items |
| `dict[str, int]` | `{"type": "string"}` | object 及属性值类型 |
| `Literal["fast", "slow"]` | `{"type": "string"}` | enum 取值约束 |

这会导致模型收到的工具描述与 Python 函数要求不一致。

工具注册还要求用户维护 schema 和 handler 两张表。[workflow 示例](../../examples/workflow_agent.py#L96) 直接访问 `_tool_schema`、`_tool_handler`，并使用 `type: ignore`；[审批示例](../../examples/tool_approval_agent.py#L122) 则要手动拆分 `scan_module()` 的返回值。

建议提供公开的函数列表注册入口，例如 `register_tools([fn1, fn2])` 或 `ToolRegistry.from_functions(...)`，统一维护 schema 与 handler。支持容器和枚举类型，对不支持的注解明确报错，并在注册阶段检查名称、签名与 schema 的一致性。

## 常用能力需要过多手动装配

[基础委派示例](../../examples/subagent_delegation_basic.py#L209) 为添加一个 subagent，需要配置注册表、会话表、工具表、多个系统，再分别安装委派工具和控制工具。遗漏某一步可能表现为能力缺失或空转。

`priority` 同时出现在系统构造参数与 `world.register_system()` 中，实际调度采用注册值；[SystemExecutor](../../src/ecs_agent/core/system.py#L115) 还将同优先级系统并发执行。作者必须理解数据依赖、优先级和并发语义，才能组合基础能力。

建议先提供薄的统一装配入口，例如 `build_agent` 或 `install_agent_runtime`，负责常用组件和系统的安装、依赖检查、重复注册检查，并展示最终执行顺序。已有 [child runtime profile](../../src/ecs_agent/systems/subagent/runtime_profiles.py#L77) 可以作为主 agent 装配的基础。

优先级应只有一个权威配置来源。常用流程可提供具名阶段或顺序约束，底层数值优先级保留给需要精细调度的用户。统一装配继续使用现有 World、EntityId、component、system 和 Runner，不复制一套运行状态。

## Prompt 和会话状态存在多个写入入口

Prompt 同时存在于 `LLMComponent.system_prompt`、`SystemPromptComponent`、模板配置和渲染结果中。[解析逻辑](../../src/ecs_agent/prompts/message_assembly.py#L369) 优先使用 RenderedSystemPromptComponent，再回退到 SystemPromptComponent 和 LLMComponent。

当更高优先级的配置或派生结果存在时，修改较低优先级字段可能完全不影响最终请求。用户很难从写入位置判断修改是否生效。

ConversationComponent 也直接暴露可变的 `messages`，但启用持久化后，写入需要经过 [append_conversation_message](../../src/ecs_agent/history.py#L55) 才会同步记录。直接追加消息会绕过该写入路径。

建议统一用户配置入口，将渲染结果明确标为派生状态；提供消息提交、历史替换等操作，协调持久化和响应链失效。调试接口应能展示最终 prompt、各部分来源和覆盖关系。

## 改进顺序和接口职责

1. 先修复权限与审批顺序、DSL 权限映射、skill 注册表覆盖和重复运行问题，并补充行为回归测试。
2. 抽取 Python 与 DSL 共用的装配逻辑，增加依赖与配置校验，复用现有运行时能力。
3. 完善工具注册、类型支持、运行结果与会话操作接口，再迁移基础示例。

面向用户的接口可收敛为三个职责：

| 职责 | 用户主要操作 |
| --- | --- |
| 定义 agent | model、prompt、tools、skills、subagents |
| 选择运行配置 | 系统组合、权限、审批、上下文管理 |
| 使用 agent | run、stream、提交消息、取消、恢复、读取结果 |

入门示例围绕这些职责编写，手动组件与系统装配保留在进阶示例。基础示例应成为公共 API 的可用性验收场景，避免它们长期展示已经不推荐的接入方式。

审查时上述入口名称属于尚未实现的设计建议；最终接口以本文开头关联的实施记录为准。设计依据还包括 [Python First Agent Authoring API RFC](2026-03-29-python-first-agent-authoring-api.md) 和 [Composable Agent Interface RFC](2026-07-18-composable-agent-interface.md)。

## 验证记录

本次审查执行了以下既有测试：

```bash
uv run --no-sync pytest -q \
  tests/test_agent_dsl_compiler.py \
  tests/test_permission_system.py \
  tests/test_tool_discovery.py \
  tests/test_run_outcomes.py
```

结果：**67 passed**。这是相关测试子集的结果。

另外通过临时 Python 脚本复现了权限绕过、审批绕过、DSL 禁用信息丢失、skill 工具注册表被覆盖、工具注解映射不完整和重复运行提前结束。脚本使用 FakeModel、临时 skill 文件或简单计数 system，没有请求真实模型，也没有调用外部业务工具。

现有测试通过不能排除这些组合缺陷。例如 DSL 测试断言所有工具为 False 时生成两个空名单，权限测试又断言空允许名单表示不限制；两项局部断言均通过，但组合后的行为违背禁用工具的意图。后续应增加“用户配置 → 编译装配 → reasoning 产生调用 → 实际执行”的集成测试，直接验证最终可观察行为。
