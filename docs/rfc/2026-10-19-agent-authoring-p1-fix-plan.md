# Agent 用户编写接口 P1 问题解决方案

日期：2026-10-19\
状态：已实现并完成验证\
关联审查：[Agent 用户编写接口审查](2026-10-19-agent-authoring-api-review.md)\
实施结果：[P1 修复与验证记录](2026-10-19-agent-authoring-p1-fix-results.md)

本方案修复两类 P1 问题：权限与审批依赖系统顺序而被绕过，以及 DSL 禁用配置未正确生效。核心改动是由工具执行入口强制完成授权检查，并统一 Python、DSL 和子 agent 的权限语义。

拒绝路径的消息契约与这两项修复一并处理：审批拒绝必须回答原始 tool call，混合允许与拒绝时必须保持结果顺序。实现继续使用现有 ECS 组件、系统、事件和 session 写入机制。

## 已确认的行为问题

| 场景 | 当前行为 | 目标行为 |
| --- | --- | --- |
| Permission 在 Reasoning 前执行 | 同一 tick 新产生的工具调用可绕过权限 | 执行入口必须检查权限 |
| Approval 在 Reasoning 前执行 | `ALWAYS_DENY` 仍可能执行 handler | 执行入口必须取得有效审批决定 |
| DSL 工具值全部为 false | 编译成空允许名单和空拒绝名单，实际不限制 | 明确禁止所有工具 |
| 审批拒绝工具调用 | 写入 system 消息，缺少对应 tool result | 写入关联原始 call ID 的 tool result |
| 批次中先允许后拒绝 | Permission 提前写入后一个调用的拒绝结果 | 按原始调用顺序写入全部结果 |

前两类绕过和 DSL 问题的复现见关联审查。进一步检查拒绝路径时，使用已有 `tool_span_error()` 得到：

```text
审批拒绝：Unanswered tool calls at end of history: ['first']
混合权限：Out-of-order or unknown tool result second (messages[2])
```

混合权限场景的原始调用顺序为 `[read, write]`，其中 `write` 被拒绝；最终结果顺序却是 `[write, read]`。这些历史行为先通过临时 Python 脚本验证，现已添加对应回归测试并修复。

## 工具执行入口强制授权

在 [ToolExecutionSystem](../../src/ecs_agent/systems/tool_execution.py#L109) 调用 handler 前，强制经过共享授权逻辑：

```text
校验调用 → 检查权限 → 获取审批决定 → 执行前复核 → 调用 handler
```

授权规则如下：

| 条件 | 执行行为 |
| --- | --- |
| 没有权限或审批配置 | 保持当前默认行为，仍进行工具与参数校验 |
| 权限禁止 | 返回拒绝结果，不调用 handler |
| `ALWAYS_DENY` | 返回拒绝结果 |
| `ALWAYS_APPROVE` | 权限允许后执行 |
| `REQUIRE_APPROVAL` | 获得明确批准后执行 |
| 审批超时、失败或没有审批处理器 | 拒绝执行 |
| 等待期间被取消 | 传播取消，不调用 handler |

权限检查先于审批，避免为已经禁止的调用发起审批。审批等待期间可能发生参数或策略变化，因此在调用 handler 前应再次核验当前调用与授权决定是否一致。

`PermissionSystem` 和 `ToolApprovalSystem` 保留为提前处理入口，与执行器共用决策逻辑和状态。即使用户漏注册它们，执行器也必须完成检查；已有配置不能因为对应 system 缺失而失效。

示例与默认装配顺序同步调整为：

```text
Reasoning(0) → Permission(1) → Approval(2) → Execution(5)
```

顺序影响处理时机，授权约束由执行入口保证。错误排序、同优先级并发和漏注册均应纳入行为测试。

## 统一决策状态和结果写入

### 执行器统一消费调用队列

两个检查系统只记录决策，由执行器统一消费原始调用队列。允许的调用执行 handler，拒绝的调用生成拒绝结果，所有结果按原始顺序写入 conversation。

例如模型生成 `[read, write]`，其中 `write` 被拒绝，历史应为：

```text
assistant: calls=[read, write]
tool: read 的执行结果
tool: write 的拒绝结果
```

拒绝消息必须使用 `role="tool"` 和原始 `tool_call_id`。[当前审批拒绝实现](../../src/ecs_agent/systems/tool_approval.py#L150) 中的 system 消息不能替代 tool result。

写入和事件契约：

- 每个调用只写入一次结果、只发布一次拒绝事件。
- 被拒绝的调用不发布执行开始事件，也不持久化为已经执行。
- 对话结果与 session 持久化使用统一写入路径。
- 多个系统并发检查同一个调用时，共享一次审批请求与决定。
- 检查系统不再提前删除待处理调用或直接追加拒绝结果；相关单元测试需随职责调整更新。

### 审批决定绑定具体调用

审批状态绑定实体、调用批次、call ID、工具名和参数摘要，避免同名工具或重复 call ID 复用旧决定。执行前同时复核当前策略；参数或策略变化后重新检查。

现有 `approved_calls: list[str]` 可保留为历史记录，但不能单凭 call ID 放行。可复用的授权状态应由运行时管理，执行时消费，checkpoint 恢复后未执行调用重新授权。

对于已经开始执行但结果未确认的调用，继续遵守既有 session 执行记录约束，不能因重新授权而自动重跑可能具有副作用的 handler。

## 统一权限数据语义

PermissionComponent 已采用以下接口：

```python
from dataclasses import dataclass, field


@dataclass(slots=True)
class PermissionComponent:
    """配置工具允许名单与拒绝名单。"""

    allowed_tools: list[str] | None = None
    denied_tools: list[str] = field(default_factory=list)
```

语义固定为：

| 配置 | 含义 |
| --- | --- |
| `allowed_tools=None` | 不设置允许名单，仍检查拒绝名单 |
| `allowed_tools=[]` | 禁止全部工具 |
| `allowed_tools=["read_file"]` | 只允许列出的工具 |
| 工具在 `denied_tools` 中 | 始终拒绝 |

所有调用者共用一个权限判断函数，顺序为先检查显式拒绝，再判断允许名单是否为 None，最后检查集合成员关系。避免 PermissionSystem、执行器、session 安装检查和子 agent 各自解释名单。

## DSL 权限编译与子 agent

将 `AgentSpec.tools` 调整为 `dict[str, bool] | None = None`，保留字段缺失与空字典的区别。JSON 与 Markdown 加载、校验和解析链路都必须保留这一区别。

| DSL 输入 | 编译结果 |
| --- | --- |
| 未声明 `tools` | 不附加权限限制 |
| `tools: {}` | 显式禁止全部工具 |
| `{"read": true, "write": false}` | 只允许 read，显式拒绝 write |
| 所有值都是 false | 禁止全部工具 |

[编译器](../../src/ecs_agent/dsl/compiler.py#L108) 同时保留 true 与 false 信息：true 进入允许名单，false 进入拒绝名单；显式声明的空允许名单保持为空。不能仅将 false 加入拒绝名单后继续把空允许名单解释为不限制，否则仍会放行未声明的工具。

运行时动态安装的工具受同一策略约束。权限检查不能只发生在编译或安装阶段。

子 agent 的 `tools` 声明使用相同编译逻辑，并应用到 child World。权限继承保留 None 与空集合的区别。若某种子 agent 权限配置暂时无法落实，应在编译时明确报错，不能接受配置后静默忽略。

## 兼容性迁移

空允许名单的含义发生变化，迁移作为本次修复的一部分交付：

| 位置 | 迁移要求 |
| --- | --- |
| Python 调用 | 原来用 `allowed_tools=[]` 表示无限制的调用改为 None |
| phase 效果 | 未声明 tools 时写入 None，显式空集合时写入 `[]` |
| 子 agent 继承 | 保留 None 与空集合的区别，完整复制权限配置 |
| session 安装检查 | 使用共享权限判断函数 |
| checkpoint | 提升 schema 版本，将旧版本空允许名单迁移为 None |
| DSL 配置 | 旧配置若以 `tools: {}` 表示无限制，应删除该字段 |
| 文档和示例 | 同步更新权限含义、系统顺序和拒绝结果契约 |

审查时 [checkpoint schema 版本](../../src/ecs_agent/serialization.py#L106) 为 1。新版本写出新的权限语义；读取旧版本时按旧运行语义迁移，缺失版本仍按既有旧版本规则处理。

已经丢失在旧 checkpoint 中的 DSL 禁用信息无法从两个空名单恢复。这类配置需要从原 DSL 重新编译，迁移器不能猜测旧名单原本代表的意图。

## 回归测试与验收

测试从 FakeModel 当轮生成调用开始，最终断言 handler 调用次数、结果消息和事件。不能仅检查组件字段或提前塞入 pending calls 后验证单个 system。

| 场景 | 必须满足 |
| --- | --- |
| 权限 system 缺失、排在 Reasoning 前或与 Execution 同优先级 | 禁止的 handler 调用次数为 0 |
| 审批 system 缺失或排序错误 | 执行器仍完整执行配置的审批策略 |
| 审批拒绝、超时、失败或处理器缺失 | handler 调用次数为 0 |
| 审批等待期间取消 | 取消传播，handler 未执行 |
| 正常批准 | handler 调用一次，审批请求一次 |
| 多个 system 并发检查同一调用 | 无重复审批、重复执行或重复结果 |
| 混合允许和拒绝 | 每个调用都有结果，顺序与原调用一致 |
| 批准后参数或策略改变 | 原决定不能直接放行变更后的调用 |
| DSL 全 false、空字典及动态安装新工具 | 保持禁止 |
| DSL 工具字段缺失 | 保持不附加限制的默认行为 |
| 子 agent 继承和声明自身权限 | 权限在 child World 实际生效 |
| phase 切换 | None 与空集合按新契约生效 |
| checkpoint 恢复 | 旧名单正确迁移，未执行调用重新授权 |
| 拒绝事件与 session 记录 | 无执行开始记录，拒绝结果只写入一次 |

对存在模型工具调用的测试，使用已有 [tool_span_error](../../src/ecs_agent/providers/_message_contract.py#L77) 校验完整消息历史，检查未回答调用、重复结果和结果乱序。

相关回归范围包括权限、审批、DSL 编译与集成、工具执行、phase、子 agent、session 和序列化测试。涉及公开类型与组件字段变化时运行严格类型检查。

## 实施顺序

1. 完成权限数据语义、共享判断函数及旧 checkpoint 迁移，同步所有权限读写位置。
2. 完成执行入口强制授权、审批状态共享和结果统一写入，补充完整执行链路的行为测试。
3. 更新 DSL 与子 agent 权限映射、示例和文档，完成跨能力回归。

实施步骤可分提交，但修复验收需要同时覆盖权限语义、执行检查和结果契约。接口调整、迁移规则与测试矩阵的实现及验证证据见关联的修复记录；下文测试结果以修复记录为准。
