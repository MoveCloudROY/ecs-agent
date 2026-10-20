# Agent 用户编写接口 P2 运行生命周期与工具接口改进规划

日期：2026-10-20\
状态：阶段 A B C 已实施并完成首轮验证\
关联审查：[Agent 用户编写接口审查](2026-10-19-agent-authoring-api-review.md)\
前置修复：[P1 修复与验证记录](2026-10-19-agent-authoring-p1-fix-results.md)\
实施结果：[首轮 P2 修复与验证记录](2026-10-20-agent-authoring-p2-fix-results.md)

本规划覆盖审查表中单独标为 P2 的前两项：运行与会话生命周期边界不清晰，以及工具类型映射和注册接口不完整。目标是让 agent 作者能够明确地区分新输入与恢复执行，并通过公开接口注册与 Python 签名一致的工具。

实现复用现有 World、组件、系统、Runner、RunResult 和工具参数校验。新增入口管理这些对象的协作，不另建一套会话历史或执行状态。下文保留设计与验收要求；最终公共 API、兼容边界和验证证据见实施结果。

## 范围与完成标准

本次交付包括运行预算修复、轮次状态、绑定实体的运行入口、执行范围约束、checkpoint 迁移，以及工具 schema 推导与统一注册。基础聊天、工具调用和多轮交互示例同步迁移，并以用户可观察行为验收。

审查表中标为 P1 / P2 的 DSL 剩余问题、通用 agent 装配接口，以及 prompt 配置入口统一不纳入本次规划。工具注册与这些能力的接入点需要保持兼容，但不据此认定相关问题已全部解决。

完成标准：

- 预算耗尽不会遗留影响下一次运行的全局终止标记。
- 新轮、恢复、取消和完成具有明确状态转换；历史消息与工具结果不会因恢复重复提交。
- agent 级入口同时限定执行范围和结果归属，多实体状态不会互相污染。
- 常见 Python 参数注解能够生成完整 schema，不支持的注解在定义或注册阶段明确报错。
- 工具作者通过公开入口完成注册，无需维护两张表或访问装饰器私有属性。
- P1 的权限、审批、结果顺序和未知工具结果恢复约束继续成立。

## 已确认的实现问题

| 位置 | 当前行为 | 本次目标 |
| --- | --- | --- |
| [Runner](../../src/ecs_agent/core/runner.py) | 耗尽 tick 预算时创建额外终止实体 | 预算结果只属于本次 Run |
| Runner | `target_entity` 限定停止条件，`world.process()` 仍处理全部实体 | 区分停止目标与执行范围 |
| Runner | `max_ticks` 与 `start_tick` 组合形成累计上限 | 新入口使用单次预算，旧调用保持兼容 |
| [ReasoningSystem](../../src/ecs_agent/systems/reasoning.py) | 对 `reasoning_complete` 终止状态特殊放行 | 由轮次状态控制是否允许继续执行 |
| Runner checkpoint | 保存时统一排除 TerminalComponent | 显式保存并恢复轮次生命周期 |
| [RunResult](../../src/ecs_agent/core/run_result.py) | 从整个 conversation 查找最近 assistant 消息 | 输出归属限定到当前轮 |
| [工具发现](../../src/ecs_agent/tools/discovery.py) | 容器和 Literal 等注解降级为 string，部分参数被跳过 | 递归生成 schema，不支持的签名明确报错 |
| [workflow 示例](../../examples/workflow_agent.py) | 读取 `_tool_schema`、`_tool_handler` 并手写双表 | 使用公开的函数列表注册接口 |

已有 [工具参数校验](../../src/ecs_agent/tools/validation.py) 使用 JSON Schema 校验模型调用，本次复用这一能力，并补充注册阶段校验。

## 运行与会话生命周期方案

### 区分会话轮次与执行尝试

| 概念 | 职责 | 标识与持续时间 |
| --- | --- | --- |
| Session | 保存会话历史、配置和持久化记录 | 复用现有 session 标识，跨多轮输入 |
| Turn | 处理一次用户输入，直至完成或明确终止 | 新增 `turn_id`，允许跨多次 Run |
| Run | 在给定预算内执行一次调度循环 | 复用 `run_id`，对应一次 Runner 调用 |

预算耗尽后恢复同一任务，保留 `turn_id`，生成新的 `run_id`。用户提交新的输入时才创建新轮。未启用持久化 session 的实体也应支持轮次管理。

轮次状态保存在 ECS component 中，至少记录轮次标识、状态和必要的恢复原因。AgentHandle 仅持有 World 与 EntityId 的绑定；共享锁、活动执行和轮次状态由运行时统一维护。

### 修正 Runner 的预算和停止契约

1. 删除预算耗尽时创建 `TerminalComponent(reason="max_ticks")` 实体的行为，返回 `RunStatus.MAX_TICKS`，保留尚未完成的工作。
2. 每次处理系统前检查目标是否已经完成、失败或被中断，避免已经终止的目标再执行一个 tick。系统处理后仍检查新产生的停止状态。
3. 将本次执行步数与累计 tick 分开计数。新增入口的 `max_steps=N` 表示本次最多执行 N 个完整 tick；`RunResult.ticks` 表示本次已完成的 tick 数。tick 上限不等同于时间限制，tick 内部仍可能等待模型、工具或输入。
4. 旧 `Runner.run(max_ticks=..., start_tick=...)` 保持累计上限语义。例如 `start_tick=5, max_ticks=8` 仍最多执行 3 次。兼容层将其转换为单次预算，避免改变现有恢复调用的含义。
5. 新入口允许 `max_steps=0` 返回未执行的预算耗尽结果，`None` 表示不设置 tick 上限，负值明确报错。累计计数继续用于观测和 checkpoint。

停止状态在下一轮启动时显式转换。新入口不依赖 TerminalCleanupSystem 自动删除标记，也不允许通过无条件删除 ErrorComponent、InterruptionComponent 或待执行工具来实现恢复。

### 增加绑定实体的运行入口

拟议用法：

```python
agent = AgentHandle(world, entity)

result = await agent.run_turn("分析这些数据", max_steps=20)

if result.status is RunStatus.MAX_TICKS:
    result = await agent.resume(max_steps=20)

if result.success:
    assert result.final_message is not None
    print(result.final_message.content)
```

| 操作 | 契约 |
| --- | --- |
| `run_turn(message, max_steps=...)` | 校验当前状态，创建新轮，经现有 history API 提交一次用户消息，启动执行 |
| `resume(max_steps=...)` | 继续可恢复的未完成轮次，不追加用户输入，不重置工具状态 |
| `cancel()` | 请求当前执行停止，等待停止过程完成，保留结果快照及恢复信息 |
| 连续调用 `run_turn()` | 上一轮结束且没有未解决工具调用后，允许启动下一轮 |

`run_turn()` 使用 [append_conversation_message](../../src/ecs_agent/history.py) 等现有写入路径，保证持久化与工作历史一致。提交失败不能留下一个看似已接受输入、实际没有对应消息的活动轮次。

轮次转换规则如下，具体枚举名称可在实现时确定：

| 当前状态 | 操作或事件 | 结果 |
| --- | --- | --- |
| 空闲或已完成 | 提交新输入 | 创建新轮，清理目标上一轮可清理的结束标记，开始执行 |
| 正在执行 | 再次提交或恢复 | 明确报错，不启动第二个驱动循环 |
| 正在执行 | 预算耗尽 | 当前 Run 结束，轮次暂停 |
| 可安全恢复的暂停 | `resume()` | 同一轮进入新的 Run |
| 正在执行 | 产生有效最终回答 | 轮次完成，保留最终结果 |
| 正在执行 | 错误或取消 | 保留原因，按实际工具执行状态分类为可恢复、失败或恢复受阻 |
| 存在结果未知的工具 | 自动恢复或提交新输入 | 拒绝绕过，返回明确的恢复阻塞原因 |

失败轮次不能通过通用清理函数自动变成可恢复轮次。实现需根据错误类别、消息完整性和工具记录判断；不能恢复时，要求显式处理错误或结束该轮。

ReasoningSystem 使用轮次状态判断执行资格，取消对 `reasoning_complete` 的特殊放行。没有新输入的已完成实体不再调用模型。旧的 UserInputSystem 与 TerminalCleanupSystem 用法通过兼容路径保留；示例迁移到新入口后，移除与新生命周期重复的重新挂载和清理逻辑。

### 分开执行范围和停止目标

AgentHandle 需要建立显式执行范围，包含目标实体及按委派规则显式调度的子任务。仅传入 `target_entity` 不构成执行隔离。

reasoning、planning、工具审批、工具执行、输入处理等会改变 agent 状态的系统，应共用公开的执行资格检查接口。World 级基础设施可以继续运行；不通过全局改写 `World.query()` 隐藏实体。

自定义系统若会处理 agent 实体，需要遵守相同的范围契约。迁移文档应给出接入方式；对于未接入的自定义系统，不能宣称已经提供执行隔离。底层显式处理整个 World 的能力继续保留。

同一个 World 只允许一个活动执行驱动者。第一版通过 World 级运行时锁或活动标记拒绝并发驱动，多个 AgentHandle 不能绕过该限制；异常和取消路径均需释放活动标记。统一协调同 World 多个并发请求的调度器留待后续设计。

### 结果与事件按轮次归属

在 RunResult 中增加兼容性的 `turn_id` 字段，保留现有 `status`、`reason`、`ticks`、`final_message`、`partial_message` 和 `error` 语义。旧底层调用没有轮次标识时允许为空，新入口必须填充。

最终输出和部分输出只来自当前轮。消息归属使用稳定标识或记录元数据，并贯穿 session 写入、流式事件、历史压缩和替换；不能仅使用会变化的 conversation 列表下标。新轮没有产生输出时，相应字段为空，不回退到上一轮 assistant 消息。

保留先创建稳定结果快照、再发布完成事件的行为，防止事件订阅者的后续修改改变已经交付的结果。基础示例直接使用 RunResult，不再通过遍历整个历史推测运行是否成功。

流式适配复用同一轮次启动和执行路径，对事件附带实体、轮次和运行归属，并交付最终 RunResult。流式与非流式路径不能分别维护生命周期；消费者提前结束流式消费时，应进入取消和清理流程。

### 取消恢复与 checkpoint

外部任务的 `CancelledError` 继续传播，Runner 保留 `last_result` 快照，保持现有异步取消契约。恢复必须区分三个情况：

- 未开始的工具调用可以保留，继续执行前重新满足所需授权条件。
- 已有确认结果的调用不能重复执行或重复写入结果。
- 已开始但结果未知的调用不能自动重跑；恢复入口给出阻塞原因，要求先明确处理执行结果。

这延续 P1 的 session 执行记录与审批约束。checkpoint 不持久化可复用审批决定，加载后的未执行调用重新授权。

checkpoint 在当前 schema 版本基础上升级，保存轮次标识、状态及必要的恢复信息。停止统一排除全部 TerminalComponent；完成和失败状态应随快照保留，保证加载本身不会让已完成轮次重新运行。

保存时不修改活跃 World。Future、锁、活动 task、瞬时执行范围及审批缓存不序列化。加载时，原先正在运行的轮次需根据持久化的工具记录归类为可恢复暂停或恢复受阻，不能直接视为仍有活跃任务。

旧 checkpoint 缺少轮次信息且可能已丢弃终止标记，不能仅凭最后一条 assistant 消息猜测状态。兼容加载需明确报告无法推断的部分，由显式恢复路径建立轮次；缺少必要证据时阻止自动重放副作用。

## 工具类型与注册接口方案

### 递归生成完整 schema

将当前只返回类型名称的映射改为返回完整 JSON Schema 的递归转换函数。自动推导首先支持以下范围：

| Python 注解 | schema 规则 |
| --- | --- |
| `str`、`int`、`float`、`bool`、`None` | 对应基础 JSON 类型 |
| `list[T]` | `type: array`，递归生成 `items` |
| `dict[str, T]` | `type: object`，递归生成 `additionalProperties`；其他键类型明确报错 |
| `Literal["fast", "slow"]` | 保留 `enum` 及相应类型；值必须能按 JSON 表达 |
| `T \| None`、`T \| U` | 保留每个分支的完整约束，必要时使用 `anyOf` |
| `Annotated[T, "说明"]` | 在 T 的完整 schema 上追加描述 |

容器和联合类型允许嵌套，例如 `list[dict[str, int | None]]`，不能在组合过程中丢失 items、enum 或对象值类型约束。

必填与可空分别定义：

```python
value: str | None          # 必填，允许 null
value: str | None = None   # 可以省略，允许 null
```

是否必填由参数是否有默认值决定，是否可空由注解决定。调用参数省略时交由 Python 函数使用默认值，不要求模型填充默认值。第一版不承诺将任意 Python 默认对象写入 JSON Schema。

### 不支持的定义尽早报错

不支持的类型、缺失注解以及未能解析的前向引用不再隐式降级成 string。自动推导无法确定参数契约时，要求使用明确的注解或显式 ToolDefinition。

自动推导模式下，位置专用参数、`*args`、`**kwargs` 明确报错，不静默跳过。受支持的签名按参数名构建 schema，并在调用绑定阶段拒绝不符合签名的参数。错误信息包含工具名、参数路径、原始注解和失败原因。

Python Enum、自定义类、TypedDict、dataclass 等复杂类型不列入首批自动推导范围。使用者可通过显式 schema 与 handler 适配器接入；需要对象转换时，由适配器把 JSON 值转为函数所需对象，不能仅生成 schema 就声称支持该类型。库不增加 Pydantic 的导入时依赖。

显式 schema 继续使用现有校验能力。注册器校验可验证的结构和参数绑定关系，不承诺自动证明任意 schema 与函数实现的语义等价。

### 提供公开的统一注册入口

拟议用法：

```python
from typing import Literal

from ecs_agent.tools import register_tools, tool


@tool()
def summarize(
    values: list[int],
    mode: Literal["fast", "slow"] = "fast",
) -> str:
    """汇总整数列表。"""
    return f"mode={mode}, total={sum(values)}"


register_tools(world, entity, [summarize])
```

新增 `ToolDefinition(schema, handler)` 作为一项工具注册的完整单元，提供 `get_tool_definition(fn)` 读取装饰器生成的定义。`@tool()` 继续返回原函数，并通过类型参数保留原始调用签名，使工具函数仍能被普通 Python 代码直接调用和测试。

`register_tools(world, entity, tools)` 接受装饰后的函数或显式 ToolDefinition，并按以下顺序工作：

1. 解析整批工具定义，检查名称、签名、schema 合法性及重复项。
2. 同时检查批次内冲突和已有注册表冲突。
3. 全部校验通过后合并到已有注册表，成对写入 schema 与 handler；不存在注册表时再创建。
4. 任一步校验失败时不写入部分工具，也不覆盖原有工具。

第一阶段保留 ToolRegistryComponent 的 `tools`、`handlers` 存储和旧构造方式，统一入口负责一致性，避免一次性改变所有组件消费者。ToolDefinition 是注册单元，不再新增另一套独立可变的权威注册表。

`scan_module()` 的现有返回形式保留兼容适配，SkillManager 和手写工具注册逐步接入同一写入逻辑。注册始终合并已有内容，不能覆盖 skill 安装的工具。全局 `_TOOL_REGISTRY` 不作为实体工具可用性或名称唯一性的权威来源。

### 注册与执行职责

注册阶段检查工具定义，执行阶段检查具体参数。复用 [tool_call_error](../../src/ecs_agent/tools/validation.py) 的 JSON Schema 校验，并保证非法参数不会进入 handler。

默认拒绝同名注册。第一版不开放存在 pending 或正在执行调用时的同名替换；若后续增加替换或注销 API，必须定义安全时机和审批失效规则，防止已批准的调用实际执行另一个 handler。

动态增加工具不授予调用权限。安装的新工具继续经过 P1 的权限、审批和执行前复核，注册入口不能改变允许名单语义。已有工具返回值处理保持原有契约，通用返回值序列化不纳入本次类型支持范围。

## 兼容策略与迁移位置

| 位置 | 迁移要求 |
| --- | --- |
| 旧 Runner 调用 | 保留 `max_ticks + start_tick` 累计上限；去掉预算终止实体属于行为修复 |
| AgentHandle | 使用单次 `max_steps`；默认绑定实体、轮次和执行范围 |
| 旧手动输入循环 | 保留兼容路径；迁移后移除与新入口重复的输入重新挂载和终止清理 |
| RunResult 与事件 | 新增字段保持旧构造调用兼容，明确旧调用的空轮次标识 |
| checkpoint | 升级 schema，兼容加载旧格式；无法恢复的信息不猜测 |
| `@tool()` | 保留原函数及调用签名，停止不支持类型的静默降级 |
| ToolRegistryComponent | 第一阶段保留双表构造；示例与内部注册调用逐步使用公开入口 |
| `scan_module()` 与 skills | 保留既有发现方式，接入共同校验与合并路径 |
| 手写 schema 和 handler | 支持 ToolDefinition，不强制迁移到装饰器 |

建议修改位置：

- 运行核心：`core/runner.py`、`core/run_result.py`、`components/definitions.py`，以及新增的 agent 级入口和轮次辅助模块。
- 系统接入：reasoning、输入、终止清理、planning、权限审批、工具执行和子任务调度中的实体处理入口。
- 持久化与事件：`history.py`、`sessions/`、`serialization.py`、事件类型和流式适配，保持归属与恢复一致。
- 工具接口：`tools/discovery.py`、`tools/validation.py`、工具公共导出、SkillManager 及其他注册调用点。
- 示例与文档：基础聊天、工具、审批、workflow、多轮交互和 checkpoint 示例，以及相应 API 参考和功能说明。

## 行为验收与测试计划

测试以 FakeModel、计数 system 和可记录调用次数的 handler 为主。关键断言包括实际执行次数、完整消息历史、持久化记录、事件归属和最终 RunResult，不能只验证组件字段。

### 生命周期验收

| 场景 | 必须满足 |
| --- | --- |
| 同一 World 先用 1 tick，再用 5 tick | 第二次得到完整预算，没有遗留全局终止标记 |
| `max_steps=0`、负值、无上限 | 分别不执行、报参数错误、按正常停止条件结束 |
| 旧 `start_tick=5, max_ticks=8` | 最多执行 3 tick，保持兼容 |
| 目标在调用前已经完成 | 不调用模型或工具，不额外执行 tick |
| 连续两轮输入 | 各提交一次，输出分别归属各自轮次 |
| 新轮尚无输出 | 不返回上一轮的最终或部分回答 |
| 预算耗尽后恢复 | 保留 turn_id，更新 run_id，不重复提交输入 |
| 多实体 | A 的运行不触发空闲 B，不被 B 的终止状态结束 |
| 同 World 并发驱动 | 第二个请求明确失败，不发生重复系统执行 |
| 安全位置取消并恢复 | 保留快照，取消契约正确，恢复后没有重复消息 |
| 工具已完成或结果未知 | 已完成调用不重跑，结果未知调用阻止自动恢复 |
| checkpoint 加载 | 完成状态保留，活动状态经过恢复分类，未执行调用重新授权 |
| 历史压缩或替换 | 当前轮输出归属仍正确 |
| 流式消费提前结束 | 执行按取消契约收尾，活动标记释放 |

### 工具接口验收

| 场景 | 必须满足 |
| --- | --- |
| 基础类型、嵌套容器、Literal 和联合 | schema 保留完整约束，合法参数能正确进入 handler |
| nullable 与默认值 | 必填、允许 null 和允许省略三个条件独立正确 |
| 不支持注解或签名 | 定义或注册阶段报错，指出工具与参数路径 |
| 非法模型参数 | handler 调用次数为 0，错误路径可定位 |
| 同批次或与已有工具重名 | 整批失败，原注册表保持完整 |
| skill 工具已安装后注册函数 | 合并成功，原工具和 handler 均保留 |
| 同步与异步函数 | 公开入口均可注册，原函数仍可直接调用 |
| 显式 schema 与适配器 | 校验与参数绑定符合声明，不冒充自动对象转换 |
| 权限禁止或审批拒绝 | 新注册工具仍不能执行 |
| 示例迁移 | 不再访问装饰器私有属性，不再使用相关 `type: ignore` |

### 验证范围

实现过程中按阶段扩展现有测试，重点包括 `test_runner.py`、`test_runner_infinite.py`、`test_resume.py`、`test_run_outcomes.py`、`test_checkpoint.py`、`test_user_input.py`、`test_tool_discovery.py`、`test_tool_authorization.py`，以及 session、子 agent、流式和多轮交互集成测试。

改变旧行为的测试应改为断言新契约。例如删除“预算耗尽生成终止实体”的断言，改为验证重复运行的实际 tick 数。新增测试同时覆盖旧调用的兼容行为。

交付前运行完整测试和严格类型检查：

```bash
uv run pytest
uv run mypy src/ecs_agent/
```

实际执行的测试命令、结果和逐项验收证据见关联的实施结果记录。

## 分阶段实施

| 阶段 | 交付内容 | 阶段验收 |
| --- | --- | --- |
| A 修复 Runner 基础契约 | 删除预算终止实体，增加执行前停止检查，分离预算与累计计数 | 重复运行、预先终止、旧恢复参数测试通过 |
| B 完善工具接口 | 递归 schema、明确错误、ToolDefinition、公开注册入口和示例迁移 | 类型矩阵、注册原子性和 P1 授权回归通过 |
| C 引入轮次与 agent 入口 | 轮次状态、AgentHandle、执行范围、取消恢复、结果归属、checkpoint 和交互示例迁移 | 生命周期矩阵、跨能力集成测试与完整检查通过 |

阶段 A 只修复底层行为，不宣称已经完成多轮会话和多实体隔离。阶段 C 将系统执行资格、持久化迁移与新入口一并交付，避免提供只有包装方法、没有生命周期保证的 API。

实施结果应回写关联审查的状态，并新增验证记录，列出实际公共 API、兼容迁移方式和测试证据。
