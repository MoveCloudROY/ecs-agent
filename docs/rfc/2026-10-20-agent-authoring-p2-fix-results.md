# Agent 用户编写接口首轮 P2 修复与验证记录

日期：2026-10-20\
状态：阶段 A B C 已实施并完成验证\
关联规划：[P2 运行生命周期与工具接口改进规划](2026-10-20-agent-authoring-p2-fix-plan.md)\
关联审查：[Agent 用户编写接口审查](2026-10-19-agent-authoring-api-review.md)

本轮完成两项 P2 改进：运行与会话生命周期，以及工具类型映射和注册接口。实现建立在工作区已有 P1 修复上，继续保持执行入口授权、审批重新核验、工具结果顺序及禁止自动重跑未知副作用的约束。

## 已交付的公共接口

```python
from ecs_agent.core import AgentHandle, RunStatus
from ecs_agent.tools import register_tools, tool

@tool()
def total(values: list[int]) -> str:
    return str(sum(values))

register_tools(world, entity, [total])
agent = AgentHandle(world, entity)
result = await agent.run_turn("计算这些数的总和", max_steps=20)
if result.status is RunStatus.MAX_TICKS:
    result = await agent.resume(max_steps=20)
```

AgentHandle 还提供 `cancel()`、`stream_turn()`、`stream_resume()`、`adopt_turn()` 和 `abandon_turn(reason=...)`。流式入口使用 async context manager，提前退出会停止尚未完成的运行并清理订阅。`get_tool_definition()` 和 `ToolDefinition` 提供装饰器元数据读取与手写工具适配，不再要求作者访问私有属性。

完整契约见 [运行生命周期](../features/agent-lifecycle.md)、[工具编写与注册](../features/tool-discovery.md) 和 [API 参考](../api-reference.md)。

## 规划验收对应关系

| 规划要求 | 实现与可观察行为 | 验证依据 |
| --- | --- | --- |
| 预算耗尽后可以继续运行 | 不创建额外终止实体；单次预算与累计 tick 分开 | [Runner 生命周期测试](../../tests/test_runner_lifecycle.py) 验证先运行 1 tick 再运行 5 tick，累计执行 6 次 |
| 保持旧预算调用语义 | `start_tick=5, max_ticks=8` 仍最多执行 3 次；新入口使用 `max_steps` | Runner 生命周期和原有 resume 测试 |
| 已完成目标不额外执行 | Runner 在处理系统前检查停止状态，Reasoning 不再特殊放行已完成标记 | Runner 生命周期及 [AgentHandle 测试](../../tests/test_agent_handle.py) |
| 新输入与恢复分开 | 新输入创建 turn_id；恢复保留 turn_id、更新 run_id，不重复提交消息 | 连续两轮、零预算后恢复、输入事件只发一次测试 |
| 结果归属正确 | RunResult 只携带当前轮输出；历史压缩后保留输出归属；复用 Message 对象不改写旧轮 | 当前轮无输出、历史替换、复用 provider Message 测试 |
| 多实体独立处理 | 内置 agent 系统使用 `query_active()`；其他实体的模型、待执行工具和审批保持不动 | 空闲或终止实体隔离、无关 pending calls、实际 child World 委派测试 |
| 同 World 单一驱动 | 并发 Handle、Runner 和手动 process 被拒绝；取消释放驱动状态 | 并发驱动、首 tick 前取消测试 |
| 取消和恢复不重复副作用 | 审批等待可安全恢复；已开始但结果未确认的工具阻止恢复与新输入 | 有无 session recording 两种取消场景、运行中与取消后 checkpoint 测试 |
| 完成的工具不重跑 | 结果提交后消费对应的原始调用对象；部分批次 checkpoint 不重复已完成前缀 | 两个工具之间保存恢复、执行开始后 call ID 被修改的测试 |
| checkpoint 保存生命周期 | schema 版本 3 保存轮次与终止状态；旧格式要求显式接续；审批缓存不恢复 | 实际 JSON 文件保存加载、完成轮次、多媒体结果、未执行调用重新审批测试 |
| 流式与非流式共享状态 | 流式适配共用运行入口，事件携带实体、轮次和 run 标识，最终结果排在全部 delta 之后 | 提前退出、旧 run 延迟事件、旧非阻塞配置下的事件顺序测试 |
| 输入持久化失败不留下假状态 | 提交失败保留上一轮、历史与终止状态，不启动执行 | 输入记录失败回滚测试 |
| 常见类型生成完整 schema | 支持基础类型、嵌套 list/dict、Literal、联合、nullable 和 Annotated；不支持类型明确报错 | [工具注册测试](../../tests/test_tool_registration.py) 使用 JSON Schema 验证合法和非法值，并执行实际 handler |
| 必填与默认值独立 | 无默认值的 nullable 参数仍必填；省略可选参数使用 Python 默认值 | nullable 参数、默认值和同步 handler 调用测试 |
| 注册接口完整且批次原子 | 校验后合并已有注册表，拒绝冲突，不覆盖 skill 工具；保留原函数签名 | 重名整批失败、schema 与签名不匹配、module discovery 与 skill 合并测试 |
| P1 约束保持 | 注册不授予权限，执行器继续执行权限与审批检查 | 原有授权测试、新注册工具的拒绝测试及真实 Responses API 回归 |
| 示例使用公共接口 | 聊天、工具、审批、workflow、流式示例迁移；增加多轮和 checkpoint 示例；plan-and-task CLI 使用轮次入口 | 示例双模式测试、[交互与 checkpoint 示例测试](../../tests/integration/test_agent_session_example.py) 及实际脚本运行 |

## 实施细节与迁移

Runner 的 `target_entity` 仍表示停止目标和结果选择；执行范围由 AgentHandle 与 `world.query_active()` 协作管理。`world.query()` 保留完整查询能力，自定义 agent 系统需要遵循新的执行资格契约。子 agent 在各自的 child World 执行。

`AgentTurnComponent` 是轮次状态的权威来源，运行 task 和锁由 World 持有。历史写入沿用现有持久化路径。每次新输入发布一个带 turn_id 和 run_id 的 UserInputReceivedEvent，恢复不重复发布；观测插件不会在每次 managed run 开始时重新发布全部旧用户消息。

`abandon_turn()` 是显式结束已停止轮次的操作，不将失败改判为成功。它拒绝未解决的工具历史和结果未知的调用。旧 checkpoint 的 `adopt_turn()` 同样不猜测丢失的完成状态。

checkpoint 现在保留 TerminalComponent，不再为了恢复而统一删除终止状态。验证过程中还修复了审批策略枚举的 JSON 编码：旧的纯字典往返未暴露这一问题，新增文件级恢复测试覆盖了策略持久化和重新审批。

工具注册第一阶段保留 ToolRegistryComponent 的双表存储与旧构造方式。公开注册入口负责校验和一致写入，SkillManager 的工具批次接入同一路径。原有直接修改字典的低层用法不具备注册入口的校验保证。自动类型推导不再把未知类型降级成 string；复杂对象通过显式 schema 和 handler 适配器接入。

plan-and-task CLI 使用 `setup_turn_input()` 和 AgentHandle，不再重新挂载 UserInputComponent。旧 `setup_interactive_input()`、TUI 和 debug 适配器保留为兼容路径。命令或可恢复失败通过明确的结束轮次操作返回输入提示，不会清除未知工具执行证据。

## 最终验证结果

| 检查 | 结果 |
| --- | --- |
| `uv run pytest -q` | 3342 passed，129 skipped，4 xfailed，6 warnings；48.43 秒 |
| `uv run mypy src/ecs_agent/` | 175 个源码文件全部通过 |
| DeepSeek Responses API 回归 | 16 passed；42.71 秒 |
| 相关文档测试 | 12 passed |
| 修改示例的语法编译 | 通过 |
| `git diff --check` | 通过 |

真实接口配置：`https://api.deepseek.com`，模型 `deepseek-flash`，`ApiFormat.OPENAI_RESPONSES`。凭据仅通过测试进程环境传入。工具效果限于内存计数和固定结果。

真实接口测试包括：

- [首轮 P2 测试](../../tests/live/test_agent_authoring_p2_live.py)：6 项，覆盖流式和非流式、嵌套字典列表、Literal、必填 nullable 参数、Python 默认值、预算耗尽后恢复、下一轮输入、session 记录归属，以及权限与审批拒绝。
- [P1 授权回归](../../tests/live/test_tool_authorization_live.py)：10 项，覆盖允许、权限拒绝、审批拒绝、人工审批入口和 DSL 禁用配置，分别验证流式与非流式。

默认全量测试中的环境依赖或显式 opt-in 用例仍会跳过；上述 16 项真实接口用例另行开启执行。4 项 xfail 是既有 native conversation tree 的写入、协议信息和恢复限制，未计为通过。6 条 warning 来自既有 structlog 配置提示。

另外实际运行了不带 API key 的聊天、流式和 workflow 示例：聊天返回正常回答，流式完整输出 141 个字符并返回 completed，workflow 完成 `DRAFT → REVIEW → DONE` 并返回 completed。多轮交互和磁盘 checkpoint 示例通过集成测试执行。

## 保留的边界

本轮不处理审查表中 DSL 剩余能力映射、通用 agent 装配和 prompt 配置入口统一问题。原生 conversation tree 的写入缺陷也仍然存在；AgentHandle 对该配置在接受输入前明确报错，避免输入被只读树上下文遮蔽，已有低层树接口继续保留。

同一 World 的多请求调度器、同名工具热替换，以及复杂 Python 对象的自动转换仍按规划留待后续扩展。它们不影响本轮已经交付的串行驱动、显式恢复和 JSON 原生类型工具接口。
