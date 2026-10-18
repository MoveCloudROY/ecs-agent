# RFC: Composable Agent Interface — 一套符合 ECS 架构的优雅用户接口

Status: Draft
Date: 2026-07-18
Target Area: 公共 API、作者体验（authoring）、能力组合、DSL 互操作、运行时手柄
Supersedes / Refines: [`docs/rfc/2026-03-29-python-first-agent-authoring-api.md`](./2026-03-29-python-first-agent-authoring-api.md)

## 摘要

本 RFC 提出一套面向用户的高层接口，让「组装一个智能体」从**手工拼装 ECS 零件**变成**声明式地组合能力（capabilities）**，同时不隐藏、不替换底层 ECS 运行时（`World` / `EntityId` / component / system / `Runner`）。

一句话概括设计：

> **能力即（组件 + 系统）的自装配包；用户只声明「要什么能力」，接口负责把它们编译成正确、有序、无死锁的 ECS 世界。**

三条主线：

1. **`Agent`** —— 唯一显式入口。90% 场景一行构造，进阶场景用 `.use(...)` 叠加能力。
2. **`Capability` + `Band`** —— 每个特性（工具、委派、记忆、审批、阶段机……）是一个自描述的能力对象，声明自己需要的组件、系统、依赖，并注册进**具名执行档位（Band）**而非魔法优先级数字。
3. **运行时手柄** —— `await agent.send("...")` 返回助手文本、`async for tok in agent.stream("...")` 流式、`agent.history`、`agent.world`/`agent.entity` 逃生舱。

本设计相对 2026-03 RFC 的关键改进是：用**正交可组合的能力**取代**会组合爆炸的 preset + 巨型 builder**，从而真正满足「使用现有全部功能 + 简练 + 可扩展」。

---

## 1. 背景与问题

### 1.1 现状：运行时强大，作者体验繁琐

框架的 ECS 运行时非常可组合，但**公共作者接口把太多层过早地暴露给用户**。以仓库现有最小示例 `examples/chat_agent.py` 为例，一个「打招呼」的 agent 需要：

```python
from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import Message

world = World()
agent_id = world.create_entity()
model = Model(name, base_url=..., api_key=..., api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS)
world.add_component(agent_id, LLMComponent(model=model, system_prompt="You are helpful."))
world.add_component(agent_id, ConversationComponent(messages=[Message(role="user", content="Hi")]))
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(ErrorHandlingSystem(priority=99), priority=99)
runner = Runner()
await runner.run(world, max_ticks=3)
conv = world.get_component(agent_id, ConversationComponent)
print(conv.messages[-1].content)
```

- 7 个模块的 import 才够「hello」。
- **魔法优先级**：`ReasoningSystem(0)`、`ErrorHandlingSystem(99)`。用户凭什么知道是 0 和 99？
- 读结果要手工挖 `ConversationComponent.messages[-1]`。

### 1.2 三个「隐性专家知识」的税

从对全部 19 个特性的接线审计（见 §9）可以归纳出用户反复踩坑的三类税：

| 税 | 例子 | 后果 |
|---|---|---|
| **优先级魔法数字** | render=-20、normalize=-10、approval/wait=-5、subagent=-1、reason=0、tool=5、compaction=-30 或 20、error=99 | 顺序错→静默行为异常 |
| **组件↔系统 配对与注册** | 有 `SystemPromptConfigSpec` 却忘了 `SystemPromptRenderSystem`；产生 `PendingToolCallsComponent` 却没有 `ToolExecutionSystem`→**永久空转死锁**；子代 profile 缺 `ToolExecutionSystem` 致死锁；新组件要进 `COMPONENT_REGISTRY` 才能序列化 | 死锁 / 特性静默失效 / 崩溃 |
| **运行时循环手工编排** | 多轮对话要自行拼 `UserInputSystem` + `TerminalCleanupSystem(clear_reasons=("reasoning_complete",))` + resolve `future`；流式要订阅 6 种 `Stream*Event` | 交互 agent 门槛陡峭 |

### 1.3 现有 DSL 只解决了一半

`compile_agent_specs()` 是最接近门面的抽象，但编译完成后**仍然**要：手工加 `ConversationComponent`、注册 `ReasoningSystem`、建 `Runner`、`run()`、再从 `ConversationComponent` 挖结果；而且要求用户写一个 `model_factory(model, prompt)` 闭包。DSL 是「声明式的一半」。

### 1.4 已有 RFC（2026-03）的取向与其局限

2026-03 的 RFC 已正确地判断：应加一层**薄门面**、**ECS 仍是运行时真相**、**保留逃生舱**、**presets 而非启发式猜测**。本 RFC 完全继承这些原则，但指出其核心方案（**preset + 巨型 `AgentBuilder`**）存在两个结构性问题：

1. **Preset 会组合爆炸。** 现实 agent 往往同时是「工具型 + 可委派 + 流式 + 记忆 + 审批」。preset 是互斥命名栈，叠加语义未定义（该 RFC 自己把「preset 组合必须确定」列为**未解的 guardrail**）。
2. **`AgentBuilder` 是 God object，不可扩展。** 该 RFC 的 builder 有 `.rag()`、`.message_bus()`、`.task_runtime()`、`.scratchbook()`、`.compaction()`…… 约 30 个专有方法。每加一个特性 = 加一个 builder 方法；**第三方特性无法接入**。这与框架「组合优于继承」的精神相悖。

本 RFC 用 **Capability + Band** 修正这两点（详见 §3、§7）。

---

## 2. 设计目标与原则

对齐用户诉求：

1. **能用现有全部功能** —— §9 给出 19 个特性到能力对象的**满覆盖映射**；未包装的一律可经 `.use(RawSystem(...))` / `agent.world` 直达。
2. **用户友好、易读** —— 单一入口 `Agent`、意图化命名、结果即返回值。
3. **用法简练** —— chat 三行、tool 五行；魔法数字与死锁陷阱由能力自动规避。
4. **架构清晰、可扩展** —— 三层分明；`Capability` 是唯一扩展缝；`Band` 取代魔法数字。

继承自 2026-03 RFC 的原则（不变）：

- **P1 ECS 是运行时真相**：门面只编译到既有 `World`/component/system/`Runner`，绝不引入第二套隐藏运行时。
- **P2 新用户有默认路径**：文档首例从 `Agent(...)` 起，而非 `World()`。
- **P3 进阶用户永远有逃生舱**：任何时刻可下沉到裸 ECS。
- **P4 显式优于启发式**：不猜测该装哪些系统；但把「显式」从 *preset 名* 升级为 *能力对象 + 具名档位*。
- **P5 运行时托管组件保持内部**：`PendingToolCallsComponent`、`RenderedSystemPromptComponent`、`ChildStubComponent` 等不进入作者视野。

新增原则：

- **P6 特性是正交能力**：能力之间可自由叠加、自动解析依赖、注册进具名档位，不产生组合爆炸。
- **P7 一个扩展缝**：第一方与第三方特性都通过实现 `Capability` + `.use()` 接入，无特权路径。

---

## 3. 总体架构

三层，严格**增量叠加**（不改动 Layer 1 任何现有代码）：

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3  语法糖 (Sugar)                                        │
│   Agent("gpt-4o", instructions=..., tools=[...])              │
│   Agent.from_markdown(...) / from_env()                       │
│   await agent.send(...) / async for agent.stream(...)         │
├─────────────────────────────────────────────────────────────┤
│ Layer 2  能力编排 (Capabilities)                               │
│   Capability 协议 · Band 具名档位 · WiringContext             │
│   依赖解析 · 冲突策略 · 自动注册（registry / profile / tool）  │
│   Tools() Delegation() Memory() Approval() Phases() ...       │
├─────────────────────────────────────────────────────────────┤
│ Layer 1  ECS 内核 (未改动，完全暴露)                           │
│   World · EntityId · Component · System · Runner · EventBus   │
└─────────────────────────────────────────────────────────────┘
```

- 用户默认待在 Layer 3；进阶时用 Layer 2 的能力对象；专家随时 `agent.world` 落到 Layer 1。
- **能力对象是 Layer 3 与 Layer 1 之间唯一的编译单元。** `Agent` 构造器的便捷 kwargs（`tools=`、`instructions=`）只是「隐式 `.use(...)` 常用能力」的糖。

---

## 4. 核心概念一览（先看效果）

```python
from ecs_agent import Agent
from ecs_agent.compose import Tools, Delegation, Memory, Approval

# (a) 最简 chat —— 一行
agent = Agent("gpt-4o", instructions="You are helpful.")
print(await agent.send("你好"))

# (b) 工具型 —— 传函数即可，@tool 自动推断 schema
@tool
async def add(a: int, b: int) -> int: ...
agent = Agent("gpt-4o", tools=[add])                 # tools= 是 .use(Tools(add)) 的糖

# (c) 正交叠加 —— 同一个 agent 同时是 工具型 + 可委派 + 记忆 + 审批
agent = (
    Agent("anthropic/claude-opus-4")
    .instructions("You are a research manager.")
    .use(
        Tools(add, read_file),
        Delegation(researcher, writer),             # 子代自动装 default runtime profile
        Memory(compact_at=120_000),                  # 自动 archive + trim + compaction
        Approval(policy="require", on_request=my_ui), # 自动 ToolApprovalSystem + 事件回调
    )
)

# (d) 流式 / 多轮 / 逃生舱
async for tok in agent.stream("讲个故事"):
    print(tok, end="")
async with agent.session() as chat:                  # 自动装 UserInput + TerminalCleanup
    await chat.send("第一轮"); await chat.send("第二轮")
world = agent.world                                   # 落到裸 ECS，一切照旧
```

对照 §1.1 的 13 行样板 + 7 个 import：能力叠加是**加法**且**语义正交**，不存在「calculator 型该选哪个 preset」的问题。

---

## 5. `Agent`：入口与运行时手柄

### 5.1 构造

```python
class Agent:
    def __init__(
        self,
        model: str | LLMModel,                 # "gpt-4o" | "anthropic/claude-..." | Model(...) 实例
        *,
        instructions: str | None = None,       # → SystemPromptConfigSpec（默认走可缓存的新路径）
        tools: Sequence[ToolLike] | None = None,   # 糖 → .use(Tools(*tools))
        capabilities: Sequence[Capability] = (),   # 显式能力（等价于 .use）
        name: str | None = None,                # 传给 World(name=...)，贯穿日志
        # —— 高级：直接给一个已构造模型 / 注册表 ——
        registry: ProviderRegistry | None = None,
    ) -> None: ...

    # 便捷构造
    @classmethod
    def from_env(cls, **overrides) -> "Agent": ...          # 读 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL
    @classmethod
    def from_markdown(cls, path: str | Path) -> "Agent": ...  # 复用 DSL loader
    @classmethod
    def from_json(cls, path: str | Path) -> "Agent": ...
    @classmethod
    def from_dir(cls, path: str | Path) -> "Agent": ...       # 多 agent 目录 → 主 agent + 子代注册
    @classmethod
    def from_spec(cls, spec: AgentSpec) -> "Agent": ...       # 桥接既有 AgentSpec

    # 能力叠加（唯一扩展缝）
    def use(self, *capabilities: Capability) -> "Agent": ...   # 返回 self，可链式
    def instructions(self, text: str) -> "Agent": ...
```

**模型糖**（消灭 `model_factory` 闭包）：
- `"gpt-4o"` → 用 `from_env` 的 base_url/key + 推断 `ApiFormat`。
- `"anthropic/claude-opus-4"` → 经 `parse_model_id` + `ProviderRegistry` 解析。
- `Model(...)` 实例 → 直接用。
- 子代（`Delegation`）模型未显式给出时，默认继承父模型 —— 复用 free-form 子代已有的「从父 `LLMComponent.model` 派生」逻辑。

### 5.2 生命周期：声明 → 物化（materialize）→ 冻结

`Agent` 是**声明式规格**；首次访问 `.world` / `.send` / `.stream` 时**惰性物化**成 ECS 世界并缓存。物化后再调 `.use()` 抛错（清晰可教的规则，避免「运行后再改配置」的歧义）。

```python
class Agent:
    @property
    def world(self) -> World: ...      # 触发物化；逃生舱
    @property
    def entity(self) -> EntityId: ...  # 主实体
    @property
    def history(self) -> list[Message]: ...   # = ConversationComponent.messages

    async def send(self, text: str, *, max_ticks: int | None = None) -> str: ...
    def stream(self, text: str) -> AsyncIterator[str]: ...   # 逐 token
    def session(self) -> "AgentSession": ...                 # 交互多轮上下文管理器
    def on(self, event_type: type[E], handler: Callable[[E], Awaitable[None]]) -> None: ...  # event_bus 订阅糖
    async def run(self, *, max_ticks: int | None = 100) -> None: ...  # 低层：直接跑 Runner
```

`send()` 封装了最大的人体工学收益：追加 user `Message` → 自动确保 `TerminalCleanupSystem(clear_reasons=("reasoning_complete",))` 已装 → `Runner().run(world, max_ticks=None)` 跑到本轮 `reasoning_complete` → 返回 `history[-1].content`。用户不接触 `Runner`、`ConversationComponent`、`TerminalComponent`。

`stream()` 内部装 `StreamingComponent(enabled=True)`，订阅 `StreamContentDeltaEvent` 转成异步生成器，`StreamEndEvent` 收尾——把「订阅 6 种事件」压成一个 `async for`。

`session()` 自动装 `UserInputSystem` + `UserInputComponent` + `TerminalCleanupSystem`，并把 `UserInputRequestedEvent.input_future` 的 resolve 收拢进 `AgentSession.send()`，让多轮交互无需手写 future 编排。

### 5.3 前后对照

| 场景 | 现状（真实行数） | 新接口 |
|---|---|---|
| chat | 13 行 + 7 imports | `Agent("gpt-4o", instructions=...)`；`await agent.send(...)` |
| tool | 手搓 `ToolSchema`/`handlers`/系统顺序 | `Agent("gpt-4o", tools=[add, mul])` |
| 多轮交互 | 手拼 UserInput+TerminalCleanup+future | `async with agent.session() as s: await s.send(...)` |
| 委派编排 | ~30 行装 Subagent* 组件/系统/工具安装 | `.use(Delegation(researcher, writer))` |
| DSL | 4 阶段 + `model_factory` + 补 conv/系统/Runner | `Agent.from_markdown("agent.md")` |

---

## 6. `Capability`：能力协议（架构核心）

一个能力就是「知道如何把自己接进世界」的自描述对象。它把 §1.2 的三类税**内聚**到能力内部。

```python
@runtime_checkable
class Capability(Protocol):
    name: str                                   # 语义槽位标识，用于冲突检测
    def requires(self) -> tuple[str, ...]: ...  # 依赖的其他能力 name（自动补齐）
    def conflicts(self) -> tuple[str, ...]: ...  # 互斥能力（如 Planning vs TreeSearch）
    def install(self, ctx: "WiringContext") -> None: ...   # 唯一装配入口
```

`WiringContext` 是能力与 `World` 之间的受控接口——能力**不直接**碰 `register_system(priority=...)`，而是声明**档位**：

```python
class WiringContext:
    world: World
    entity: EntityId
    def add_component(self, component: object, *, replace: bool = False) -> None: ...
    def register_system(self, system: System, *, band: Band, order: int = 0) -> None: ...
    def register_component_type(self, cls: type) -> None: ...  # 自动登记 COMPONENT_REGISTRY
    def install_tool(self, schema: ToolSchema, handler: ToolHandler) -> None: ...
    def install_skill(self, skill: Skill) -> None: ...          # 复用 SkillManager
    def on_event(self, event_type, handler) -> None: ...        # 复用 event_bus
    def register_child_profile(self, name: str, builder) -> None: ...  # 子代 runtime profile
```

### 6.1 一个能力的样子（`Tools`）

```python
@dataclass(slots=True)
class Tools:
    fns: tuple[ToolLike, ...]
    max_concurrency: int = 8
    name = "tools"
    def requires(self) -> tuple[str, ...]: return ()
    def conflicts(self) -> tuple[str, ...]: return ()
    def install(self, ctx: WiringContext) -> None:
        ctx.add_component(ToolRegistryComponent(tools={}, handlers={}))
        ctx.add_component(ToolExecutionConfigComponent(max_concurrency=self.max_concurrency))
        for fn in self.fns:
            schema, handler = _adapt_tool(fn)         # 复用 @tool / scan_module 元数据
            ctx.install_tool(schema, handler)
        ctx.register_system(ToolExecutionSystem(), band=Band.ACT)   # 关键：不再手填 priority=5
```

注意：`Tools` **不需要**用户记得「否则 `PendingToolCallsComponent` 会永久空转」——这个死锁陷阱被能力吃掉了。

### 6.2 编译流程（惰性物化时执行）

```
1. 收集所有 .use() 的能力 + 构造糖隐式能力（含始终存在的 Core=Reasoning+ErrorHandling+PromptRender）
2. requires() 传递闭包 → 自动补齐缺失依赖（如 Delegation 需要 Tools）
3. conflicts() 校验 → 冲突则 fail-fast，报清晰错误
4. 按 name 去重 / 合并同槽位（默认 fail，除非 replace=True）
5. 拓扑 + Band 排序，逐个 install(ctx)
6. Band → 真实 priority 映射后 register_system；返回冻结的 World
```

**Core 能力**（隐式、始终装）保证任何 agent 都至少有 `SystemPromptRenderSystem(PRE_RENDER)` + `ReasoningSystem(REASON)` + `ErrorHandlingSystem(RECOVER)`，并按 caching 需要默认用 `SystemPromptConfigSpec`（而非 legacy `SystemPromptComponent`），让 prompt 缓存前缀/易变尾拆分、`${...}` 占位符、阶段 prompt 都开箱可用。

---

## 7. `Band`：具名执行档位（消灭魔法优先级）

把散落的魔法数字收敛为一个**小而有序**的枚举。能力声明档位，档位再映射到既有真实 priority，**行为完全等价**，但语义可读、可扩展、防错序。

```python
class Band(IntEnum):
    PRE_TURN   = -30   # CompactionSystem(when="before_turn")（当轮即反映压缩）
    PRE_RENDER = -20   # SystemPromptRenderSystem（默认；见下方例外）
    INGEST     = -15   # UserInputSystem（须严格早于 normalize，把输入写入 conversation）
    NORMALIZE  = -10   # UserPromptNormalizationSystem / PermissionSystem / RAGSystem
    GATE       = -5    # ToolApprovalSystem / SubagentWaitSystem（审批、背景会话回收——须早于 reason）
    DELEGATE   = -1    # SubagentSystem（把 subagent 工具装进注册表——须早于 reason）
    REASON     =  0    # ReasoningSystem / PlanningSystem / TreeSearchSystem（互斥，见 conflicts）
    ACT        =  5    # ToolExecutionSystem / MessageBusSystem
    POST_ACT   =  7    # ReplanningSystem（docstring 明确要求晚于 ToolExecution）
    MAINTAIN   =  20   # CompactionSystem / CheckpointSystem（默认：回合后维护）
    RECOVER    =  99   # ErrorHandlingSystem
```

> **⚠️ 校正（源于 §12A 对抗性审计）**：初版把档位定为 8 段，且断言真实排序权威是各系统的默认 priority。审计证伪了这两点——排序权威实际是 `register_system(system, priority)` 传入的 int（`core/system.py` 只按注册 priority 分 TaskGroup 组，从不读 `system.priority`），而真实系统里存在 `ReplanningSystem@7`（须跨过 tools 的 `5`）与 `UserInputSystem@-15`（须早于 normalize 的 `-10`）等跨档依赖，8 段无法忠实复现。故补入 `INGEST=-15` 与 `POST_ACT=7`，并确立下列铁律。

设计要点（含审计后的硬约束）：

- **Band 派生自系统默认，绝不复制成第三份真值源。** 每个 System 类声明唯一 `default_band`；`WiringContext.register_system` 在能力未覆盖时读取它。编译器加 **fail-fast 断言** `band_to_priority(declared_band) == class_default_priority`，并对全系统 catalog 参数化测试。否则「改系统 priority 忘更 Band → 静默错序」这一税只是从示例搬进了 band 层。
- **档位映射对任何 write→read 依赖对必须单射。** 两个系统若一写一读同一组件（如 `UserInputSystem` 写 conversation、`UserPromptNormalizationSystem` 读它），绝不能映射到同一档位（同 priority = 并发 = 漏序）。`INGEST`/`POST_ACT` 即为此而设。
- **档位可 per-instance 覆盖。** `ctx.register_system(sys, band=..., order=...)` 支持覆盖：例如 phase 场景下 `SystemPromptRenderSystem` 须晚于 NORMALIZE 内同步 `advance()` 改 phase 的 script 触发器，因此 `Phases` 能力把 render 覆盖到 `GATE`（`examples/e2e/plan_and_task` 真实用 `-5` 而非 `-20`）。**初版「PRE_RENDER 与 NORMALIZE 次序不承重」的说法是错的**——当 script 触发器在 NORMALIZE 内改 `PhaseComponent` 时，render 必须晚于它，否则渲染上一 tick 的过期 phase prompt。
- **Compaction 的双重时机**：通用推荐 `MAINTAIN(20)`（回合后压缩），`plan_task` 用 `PRE_TURN(-30)`（当轮即反映）。`Memory(when="after_turn"|"before_turn")` 暴露为语义选项而非裸数字。

Band 直接回答 P4：显式，但显式的是**意图档位**而非 0/5/99；且档位是**从系统默认派生的可校验别名**，不是并列的第三份配置。

---

## 8. 能力目录（意图化命名）

一等能力对象，均实现 `Capability`。构造器 kwargs（`tools=`、`instructions=`）是最常用几个的糖。

| 能力 | 意图 | 关键参数 |
|---|---|---|
| `Tools(*fns_or_skills)` | 工具调用 | `max_concurrency` |
| `Delegation(*subagents)` / `Subagents(...)` | 子代委派/编排 | `background`, `allow_freeform`, `profile` |
| `Memory(...)` | 压缩+归档+上下文预算 | `compact_at`, `when`, `trim_tool_results` |
| `Streaming()` | 流式 | `non_blocking` |
| `Approval(policy=...)` | 工具人审 | `policy`, `timeout`, `on_request` |
| `Permissions(allow=, deny=)` | 工具白/黑名单 | — |
| `Planning()` / `ReAct()` | 计划+重规划 | — |
| `TreeSearch(...)` | MCTS 计划搜索 | `max_depth`, `max_branching` |
| `RAG(embedder, store)` | 检索增强 | `top_k` |
| `Checkpointing()` | 快照/撤销 | `max_snapshots` |
| `Phases(graph)` / `Workflow(graph)` | 阶段机/审批门 | `graph`, `agent_key` |
| `Todos()` | 会话清单 | — |
| `Prompts(placeholders=, triggers=)` | 占位符/触发器/上下文池 | `enable_context_pool` |
| `MCP(*servers)` | 外部 MCP 工具服务器 | `transport` |
| `MessageBus(...)` | 多 agent pub/sub·请求响应 | `subscriptions` |
| `HumanInput()` | 异步人工输入 | `prompt`, `timeout` |
| `Observability(*plugins)` | Langfuse/Prometheus/自定义 | 复用 `install_plugins` |
| `Workspace(root)` | 文件系统绑定 | `root` |
| `RawComponent(c)` / `RawSystem(s, band=)` | 逃生舱能力 | —— 用于任何尚未包装的特性 |

`RawComponent` / `RawSystem` 是**满覆盖保证**：任何未来或未包装的系统都能以能力形态接入同一编排管线（含 Band 排序与依赖），无需回退到裸 `world.register_system`。

---

## 9. 全景映射表：19 个特性 → 能力 → ECS 接线

证明「能用现有全部功能」。列基于对 `src/ecs_agent/{components,systems,...}` 的完整审计，class 名与真实 priority 一致。

| # | 特性 | 能力 | 组件（主要） | 系统（档位） | 自动登记/陷阱规避 |
|---|---|---|---|---|---|
| 1 | Reasoning/Chat | *Core（隐式）* | `LLMComponent`,`ConversationComponent`,`SystemPromptConfigSpec` | `SystemPromptRenderSystem(PRE_RENDER)`,`ReasoningSystem(REASON)`,`ErrorHandlingSystem(RECOVER)` | 默认 ConfigSpec 路径→缓存/占位符可用 |
| 2 | 工具调用 | `Tools` | `ToolRegistryComponent`,`ToolExecutionConfigComponent` | `ToolExecutionSystem(ACT)` | 自动装执行系统，防「pending 空转」死锁 |
| 3 | 技能 | `Tools`(skill)/`install_skill` | `SkillComponent`,`WorkspaceBindingComponent` | 复用 `SkillManager.install` | 文件技能自动先绑 workspace |
| 4 | 子代委派 | `Delegation` | `SubagentRegistryComponent`,`SubagentSessionTableComponent` | `SubagentSystem(DELEGATE)`,`SubagentWaitSystem(GATE)` | 自动 `install_subagent_tool`；子代 default profile 含 ToolExecution（防子代死锁）；单例并发 cap |
| 5 | 计划/重规划 | `Planning` | `PlanComponent` | `PlanningSystem(REASON)`,`ReplanningSystem(REASON+order)` | 与 `Reasoning`/`TreeSearch` 互斥（conflicts） |
| 6 | 树搜索 MCTS | `TreeSearch` | `PlanSearchComponent` | `TreeSearchSystem(REASON)` | 与 `PlanComponent` 互斥（conflicts） |
| 7 | RAG | `RAG` | `RAGTriggerComponent`,`EmbeddingComponent`,`VectorStoreComponent` | `RAGSystem(NORMALIZE)` | 缺 embedder/store fail-fast 而非静默 |
| 8 | 检查点 | `Checkpointing` | `CheckpointComponent` | `CheckpointSystem(MAINTAIN)` | undo 静态方法照旧可用 |
| 9 | 压缩 | `Memory` | `CompactionConfigComponent`,`ConversationArchiveComponent` | `CompactionSystem(MAINTAIN 或 PRE_TURN)` | 自动补 archive（否则压缩静默失效）；`when=` 暴露时机 |
| 10 | 上下文预算/池 | `Memory`/`Prompts` | `ContextTrimConfig`,`UserPromptConfigComponent(enable_context_pool)`,`PromptContextQueueComponent` | `PromptContextCollectorSystem(REASON)` | 池 opt-in |
| 11 | 流式 | `Streaming` | `StreamingComponent(enabled=True)` | —（Reasoning 内） | `stream()` 把 6 事件收成 `async for` |
| 12 | 权限/审批 | `Permissions`/`Approval` | `PermissionComponent`,`ToolApprovalComponent` | `PermissionSystem(NORMALIZE)`,`ToolApprovalSystem(GATE)` | 均早于 `ToolExecutionSystem(ACT)` |
| 13 | 消息总线 | `MessageBus` | `MessageBusConfigComponent`,`MessageBusSubscriptionComponent` | `MessageBusSystem(ACT)` | — |
| 14 | 用户输入 | `HumanInput`/`session()` | `UserInputComponent` | `UserInputSystem(NORMALIZE)` | 配 `TerminalCleanup` 支持多轮 |
| 15 | 阶段/工作流 | `Phases` | `PhaseComponent`,`PhaseDefinitionComponent`,`PhaseApprovalsComponent` | 复用 `bind_phase_graph`（无专用系统） | 提供 `agent.advance()/force()/record_approval()` 糖；`${_phase_prompt}` 自动 |
| 16 | Prompt 归一化 | `Prompts` | `SystemPromptConfigSpec`,`UserPromptConfigComponent`,`RenderedUserPromptComponent` | `UserPromptNormalizationSystem(NORMALIZE)` | 内置占位符自动展开 |
| 17 | Todo | `Todos` | `TodoListComponent` | 经 `todo_write` 工具（`TodoSkill`） | 已在 `COMPONENT_REGISTRY`；穿越压缩 |
| 18 | MCP | `MCP` | `MCPConfigComponent` | 经 `MCPSkillAdapter`+`SkillManager` | 命名空间 `server/tool` |
| 19 | 可观测性 | `Observability` | —（插件自管状态） | 复用 `install_plugins` | 子代世界自动 `propagate_plugins` |

> 任何上表未列尽的系统（如未来新增），均可 `.use(RawSystem(sys, band=Band.ACT))` 一等地接入。

---

## 10. DSL 统一

把「声明式的一半」补全：DSL loader 的产物直接就是 `Agent`，不再需要 `model_factory` 与善后。

```python
agent = Agent.from_markdown("agents/assistant.md")   # 复用 discover/load/resolve/compile
await agent.send("Hi")                                # 立即可跑，无需补 conv/系统/Runner
```

- `compile_agent_specs` 内部化：门面提供 `model_factory`（由 `Agent` 的模型糖统一构造），用户不再手写闭包。
- `AgentSpec` 保持**严格、可序列化、文件友好**的边界（不接受 live callable/provider）；live 行为归 Python 层。`Agent.from_spec(spec)` 桥接二者。
- 目录多 agent：`Agent.from_dir(...)` 返回主 agent，子代 spec 自动落入 `Delegation` 能力。

---

## 11. 扩展性

### 11.1 自定义能力（第三方零特权接入）

```python
@dataclass(slots=True)
class RateLimit:
    rpm: int
    name = "rate_limit"
    def requires(self): return ("tools",)      # 声明依赖，编译期自动校验/补齐
    def conflicts(self): return ()
    def install(self, ctx: WiringContext) -> None:
        ctx.register_component_type(RateLimitComponent)   # 自动进 COMPONENT_REGISTRY
        ctx.add_component(RateLimitComponent(rpm=self.rpm))
        ctx.register_system(RateLimitSystem(), band=Band.GATE)

agent = Agent("gpt-4o", tools=[...]).use(RateLimit(rpm=60))
```

第一方能力（`Tools`、`Delegation`……）与它走**完全相同**的路径——没有 God-builder 的专有方法特权。这是相对 2026-03 RFC 的关键可扩展性提升。

### 11.2 逃生舱分级

```python
agent.use(RawSystem(MySystem(), band=Band.REASON))  # 能力级：仍享 Band 排序 + 依赖
w = agent.world; e = agent.entity                     # 裸 ECS：与今日 100% 一致
w.register_system(MySystem(), priority=3)             # 想手填 priority 也行
```

### 11.3 子代 profile / 插件

`Delegation` 可传 `profile=` 复用 `register_child_runtime_profile`；`Observability(LangfusePlugin(), PrometheusPlugin())` 复用 `install_plugins`，子代世界自动传播——既有机制原样复用，不重造。

---

## 12. 兼容性与迁移

- **严格增量**：Layer 1 一行不改，现有 300+ 测试与所有 `examples/*.py` 继续通过。新代码位于新子包 `ecs_agent.compose`（`Agent` 从顶层 `ecs_agent` 再导出）。
- **配对示例**：每个 example 增加「糖路径 / 裸 ECS 路径」对照，帮助读者建立心智映射（satisfies 文档 P2）。
- **无破坏性弃用**：`SystemPromptComponent` 等 legacy 路径仍支持；门面默认走 `SystemPromptConfigSpec` 只影响新建 agent。
- **plan_and_task 作为验收样例**：现存最完整的手工装配（~10 组件 / 7 系统 / phase 绑定 / 4 子代）应能用能力表达而**行为不变**——这是本设计的「能表达全部功能」硬验收。

---

## 12A. 兼容性与可维护性审计结论（对抗性核验，2026-07-18）

对本方案跑了一轮多智能体审计：8 个接缝并行探查 → 28 个高/中风险逐一对抗性核验（尝试用真实代码推翻）→ 综合。结论如下。

### 评级

| 维度 | 评级 | 首要原因 |
|---|---|---|
| **兼容性** | **中** | 序列化 shape 不变、内置组件已全在 `COMPONENT_REGISTRY`、命名零冲突、子世界委托稳定；但「Band→priority 行为一致」这一核心承诺在 ≥3 处被证伪，且是**静默错序**。 |
| **可维护性** | **中** | Band 若实现为独立配置即是第三份真值源；facade 编译器与 `compile_agent_specs` 完整度契约不同，构成长期漂移面。均可经下方铁律消除。 |

被推翻/降级的担忧（good news）：序列化 shape 不变、`_materialize` 单次幂等不会覆盖 restore、`Agent`/`Capability`/`Band` 无命名冲突、`test_types.py` 不锁 `__all__`、子世界 `default_child_system_specs` 的裸 int 恰与 Band 相等（可一次收敛）。

### 确认的高危风险（must-fix，已在 §7/§15 校正部分吸收）

| 风险 | 为何真实 | 缓解 |
|---|---|---|
| `replanning-band-inversion` | `ReplanningSystem@7` 须晚于 tools(5)，`REASON(0)+order` 跨不了 priority 组边界 | 已加 `POST_ACT=7` band |
| `prompt-render-two-bands` | render 在 DSL=-20、plan_task=-5（须晚于 NORMALIZE 内改 phase 的 script 触发器） | Band per-instance 覆盖；`Phases` 把 render 置于 `GATE`；已订正 §7 错误论断 |
| `core-skeleton-omits-render-system` | §15 骨架未装 render 且 `LLMComponent` 无 system_prompt → stable prefix 为空 = **缓存全崩** + 丢占位/phase 渲染 | Core.install 无条件建 `SystemPromptConfigSpec` + 注册 render@PRE_RENDER；Phase 0 合并门加**离线 prefix-audit 字节等价断言**（复用现有 cache-audit harness），而非只断言 priority 相等 |
| `checkpoint-restore-loses-phase-graph` | `PhaseDefinitionComponent` 不序列化、`undo()` 不 rebind → `.use(Checkpointing(), Phases())` restore 后 render 抛 `PhaseIntegrityError` | `Phases` 能力保留 graph 并 `on_event(CheckpointRestoredEvent, rebind)`（过滤 entity_id、决定是否 on_resume） |
| `compose-init-eager-optional-dep-import` | `compose/__init__` 在 `import ecs_agent` 关键路径，任何模块级 `from ecs_agent.mcp import` 会让裸装环境 `import ecs_agent` 崩（`mcp/__init__.py` eager `import mcp`） | `compose/__init__` 只 re-export 纯 Layer-1；`MCP` 能力惰性导入；加子进程回归测试断言 `mcp/langfuse/prometheus` 不在 `sys.modules` |
| `handle-shutdown-leak` / `async-install-in-sync-capability` | `install_plugins` 是 **async 且做真实 I/O**，塞不进 sync `install(ctx)`；`PluginsHandle` 持 socket/client 而 `Agent` 无 teardown → 泄漏 + 丢 telemetry | 见铁律 3（两相激活 + `aclose()`/`__aexit__`） |

### 确认的中危（维护面）

- **两编译器漂移**：`compile_agent_specs` 是**部分世界**（只装 render/-20、normalize/-10、条件 subagent/-1；**不装** Reasoning/ToolExec/Error/Conversation，也**从不装** `SubagentWaitSystem`）。若 `from_markdown` 直接返回其 world，`send()` 会空转到 max_ticks，且 background+wait 腿失效。
- **Band 第三真值源**：今天 `ToolExecutionSystem` 默认 0 vs 处处注册 5 已漂移；Band 若不派生自系统默认会再加一份。
- **自定义组件 checkpoint 静默丢失**：`from_dict` 对未注册组件名 `.get()->None: continue` 静默跳过，而 `COMPONENT_REGISTRY` 是只读字面量。`register_component_type` 须带可选 `to_dict/from_dict` hook（否则 set/嵌套 dataclass/live handler 走默认 `asdict` 会抛 `TypeError` 或静默错序）。

### 使其安全的 5 条铁律（并入实施计划）

1. **Band 派生自系统默认、编译期校验**：系统类声明唯一 `default_band`；编译器断言 `band_to_priority(band)==class_default`；映射对 write→read 依赖对保持单射。
2. **单一编译器**：`from_markdown/json/dir/spec` 经 `spec_to_capabilities(AgentSpec, model_factory)` 翻译成能力后走**同一** `_materialize`，**不**直接返回 `compile_agent_specs` 的半成品世界；加断言「materialized world 必含 REASON-band 系统，否则 fail-fast 而非空转」。
3. **Observability 走异步两相 + 显式 teardown**：`install()` 纯声明；`Agent._activate()` 在首个 `await send/stream/session` 幂等触发 `install_plugins`；`Agent` 加 `async def aclose()`（先 flush 后 `uninstall_plugins`）与 `__aexit__`，observable agent 用 `async with`；`.world` 若有待激活 async 插件必须报错。
4. **restore 显式重建接线**：`Phases` 注册 `CheckpointRestoredEvent` rebind + 提供 `agent.resume_phase()` 包 `resume_phase_graph`（保留 `approval_replay:` 不追加 ledger 的语义）；`register_component_type` 带 serialize hook，且 re-install 早于 `from_dict`，未知组件名 strict 报错而非静默丢。
5. **Core 是缓存不变量的唯一 choke point**：无条件建 `SystemPromptConfigSpec` + 注册 render；Phases 存在时确保模板含 `${_phase_prompt}`；合并门含字节级 prefix-audit。

---

## 13. 取舍与开放问题

1. **惰性物化 vs 显式 `.build()`**：本 RFC 取「首次使用即物化 + 物化后冻结」，比 2026-03 的显式 `.build()→BuiltAgent` 少一步仪式，但需清晰的「冻结后 `.use` 抛错」规则。是否仍保留显式 `.build()` 供想先检视 world 的用户？倾向：保留 `agent.world` 即等价检视点，不单列 `BuiltAgent`。
2. **同步包装**：是否提供 `send_sync()`（内部 `asyncio.run`）？倾向：提供，便于脚本/notebook，但文档以 async 为主。
3. **冲突/去重策略**：同槽位能力重复默认 fail-fast，`replace=True` 覆盖；`Planning`↔`TreeSearch`、`Planning`↔`Reasoning` 经 `conflicts()` 静态互斥。边界用例（两个都想改 system prompt 的能力）需策略文档化。
4. **Band 是否够用**：8 档能否覆盖所有现实排序？`order` 微调是否足以避免再引入数字？倾向：先 8 档 + `order`，据实践再加档。
5. **命名**：能力子包名 `ecs_agent.compose`（呼应「组合优于继承」）vs `ecs_agent.agent`（2026-03 提法）。`Agent` 与 `AgentSpec`/subagent 概念是否会混淆，需在文档明确「`Agent`=作者手柄，`AgentSpec`=可序列化 DSL 记录」。

---

## 14. 分阶段实施计划

遵循项目 TDD + env-gated real-LLM 测试规范；每阶段附文档与 README 更新。

**Phase 0 — 骨架（Capability/Band/WiringContext + Core + Agent.send）**
- `Capability` 协议、`Band` 枚举、`WiringContext`、依赖/冲突解析器、编译器。
- 隐式 `Core` 能力；`Agent(model, instructions=)`、`send()`、`history`、`world`/`entity`。
- 模型糖（str/registry/实例，`from_env`）。
- 验收：`chat_agent` 等价于三行；单测覆盖编译顺序 = 既有真实 priority。

**Phase 1 — 常用能力**
- `Tools`（`@tool`/`scan_module` 适配）、`Streaming`（`stream()`）、`Approval`、`Permissions`、`Memory`。
- `session()` 多轮 + `HumanInput`。

**Phase 2 — 编排与声明式**
- `Delegation`/`Subagents`、`Phases`/`Workflow`、`Prompts`、`Todos`、`Planning`/`ReAct`。
- DSL 统一：`from_markdown/json/dir/spec`；内部化 `model_factory`。

**Phase 3 — 进阶特性与逃生舱**
- `RAG`、`TreeSearch`、`Checkpointing`、`MessageBus`、`MCP`、`Observability`、`Workspace`。
- `RawComponent`/`RawSystem`；自定义能力指南。
- 用 plan_and_task 做「全功能可表达」验收，并给出配对示例。

---

## 15. 附：最小实现骨架（说明性）

```python
# ecs_agent/compose/bands.py
class Band(IntEnum):
    PRE_TURN=-30; PRE_RENDER=-20; INGEST=-15; NORMALIZE=-10; GATE=-5
    DELEGATE=-1; REASON=0; ACT=5; POST_ACT=7; MAINTAIN=20; RECOVER=99

# Core 无条件保证缓存不变量：建 SystemPromptConfigSpec + 注册 render@PRE_RENDER
class Core:
    def __init__(self, instructions=None): self.instructions = instructions or ""
    name = "core"
    def install(self, ctx):
        ctx.add_component(SystemPromptConfigSpec(
            template_source=PromptTemplateSource(inline=self.instructions)))
        ctx.register_system(SystemPromptRenderSystem(), band=Band.PRE_RENDER)
        ctx.register_system(ReasoningSystem(),          band=Band.REASON)
        ctx.register_system(ErrorHandlingSystem(),      band=Band.RECOVER)

# ecs_agent/compose/agent.py
class Agent:
    def __init__(self, model, *, instructions=None, tools=None, capabilities=(), name=None):
        self._model_ref = model
        self._caps: list[Capability] = [Core(instructions=instructions)]
        if tools: self._caps.append(Tools(*tools))
        self._caps += list(capabilities)
        self._name = name
        self._world: World | None = None          # 惰性

    def use(self, *caps: Capability) -> "Agent":
        if self._world is not None:
            raise RuntimeError("agent already materialized; configure before first send()")
        self._caps.extend(caps); return self

    def _materialize(self) -> None:
        model = _resolve_model(self._model_ref)
        world = World(name=self._name); entity = world.create_entity()
        world.add_component(entity, LLMComponent(model=model))
        world.add_component(entity, ConversationComponent(messages=[]))
        plan = _resolve_capabilities(self._caps)   # requires/conflicts/dedup → 有序列表
        ctx = WiringContext(world, entity)
        for cap in plan:                           # 已按 Band 排序
            cap.install(ctx)
        ctx.flush_systems()                        # Band → priority → register_system
        self._world, self._entity = world, entity

    @property
    def world(self) -> World:
        if self._world is None: self._materialize()
        return self._world

    async def send(self, text: str, *, max_ticks=None) -> str:
        w = self.world
        conv = w.get_component(self._entity, ConversationComponent)
        conv.messages.append(Message(role="user", content=text))
        _ensure_terminal_cleanup(w)                # 自动装，若未装
        await Runner().run(w, max_ticks=max_ticks)
        return w.get_component(self._entity, ConversationComponent).messages[-1].content
```

---

## 16. 结论

用**能力（自装配的组件+系统包）+ 具名档位**取代**魔法优先级 + preset/God-builder**，得到一套：

- **能用全部功能**（19 特性满覆盖 + `RawSystem` 兜底），
- **友好易读**（单入口、意图命名、结果即返回），
- **简练**（chat 三行、能力叠加是加法），
- **架构清晰、可扩展**（三层分明、一个扩展缝、ECS 仍是真相）

的用户接口。它继承 2026-03 RFC 的正确取向，并修正其组合爆炸与不可扩展的结构缺陷。

一句话：

> **让作者只声明「要什么能力」，让 ECS 诚实地做运行时。**
