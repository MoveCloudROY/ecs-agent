# Agent 用户编写接口 P1 修复与验证记录

日期：2026-10-19\
状态：已完成\
方案：[P1 问题解决方案](2026-10-19-agent-authoring-p1-fix-plan.md)

权限与审批检查已移入工具执行的必经路径。DSL 中字段缺失、空映射和显式禁用具有不同且一致的运行语义。检查系统缺失、排序错误或并发执行时，被禁止或未获批准的 handler 均不会执行。

## 实现结果

| 方案要求 | 实现与证据 |
| --- | --- |
| 执行入口强制权限与审批 | [ToolExecutionSystem](../../src/ecs_agent/systems/tool_execution.py) 在执行前调用 [共享授权逻辑](../../src/ecs_agent/tools/authorization.py)，不依赖预检查 system 是否注册 |
| 并发共享审批及单次消费 | 授权状态存放在实体的临时 ToolRuntimeStateComponent 中，以 pending batch 区分批次；每个调用共享锁和决策，执行器共享批次消费锁 |
| 参数与策略变化后复核 | 授权指纹包含 call ID、工具名、参数与当前策略；变化后重新授权，历史 approved_calls 不作为放行依据 |
| 有序回填拒绝结果 | 执行器统一按调用顺序写入 tool 消息与 session 记录；拒绝只发布一次 ToolDeniedEvent，不发布执行开始事件 |
| 审批异常与取消 | 超时覆盖事件订阅者执行和等待答案；没有处理器、订阅者报错或明确拒绝均禁止执行；外部取消传播并清理等待 Future |
| None 与空允许名单 | [共享权限判断](../../src/ecs_agent/tools/permissions.py) 规定 None 不限制、空列表全部禁止，显式拒绝优先 |
| DSL 主 agent 与子 agent | [编译器](../../src/ecs_agent/dsl/compiler.py) 保留 true/false 和缺失/空映射差别；[child world 装配](../../src/ecs_agent/systems/subagent/child_world.py) 应用本地策略并与继承策略取交集 |
| phase 与 session | phase 未声明工具设置 None，显式空集合设置空名单；session 工具安装复用同一权限判断 |
| checkpoint 迁移 | [WorldSerializer](../../src/ecs_agent/serialization.py) 写入 schema version 2，读取 v1 或未标版本时将旧空允许名单迁移为 None；授权运行状态不序列化，恢复后重新审批 |
| 可观测性 | 拒绝事件在结果持久化后发布；参数校验失败仍产生包含调用参数的失败观察记录，无须伪造执行开始事件 |

授权在执行开始时提交并冻结参数和所选 handler。进入 handler 前不向异步观察订阅者让出控制权；后续策略变化影响尚未开始的调用。已经开始但结果未确认的持久化调用仍禁止自动重跑。

## 自动化测试

执行命令：

```bash
uv run pytest -q
uv run mypy src/ecs_agent/
```

结果：

- 全量 pytest：**3280 passed，123 skipped，4 xfailed，0 failed**，用时 47.99 秒。
- 严格类型检查：**170 个源文件通过**。
- 4 个预期失败来自已有的 [原生对话树缓存契约测试](../../tests/test_context_cache_native_tree.py)，本次未新增或放宽 xfail。
- 默认测试运行中的跳过项保留环境或可选依赖门控；本次新增的 10 个实网用例已另行启用并全部通过。

主要行为测试位于 [test_tool_authorization.py](../../tests/test_tool_authorization.py)，覆盖：

- 模型当轮生成调用，权限和审批 system 缺失、早于 reasoning、同优先级并发及晚于 execution。
- 明确允许、拒绝、审批超时、处理器缺失、处理器失败和外部取消。
- 多个预检查与执行器并发时，审批请求、handler 调用及结果均只发生一次。
- 批准后更改参数、权限或审批配置，审批等待期间修改参数，以及新批次复用 call ID。
- 混合允许与拒绝的消息闭合、顺序及 session 持久化，拒绝时无执行开始记录。
- JSON 与 Markdown DSL 的缺失字段、空映射、全部 false、混合映射及编译后安装工具。
- 子 agent 的 DSL 权限、本地与继承权限交集、配置序列化和实际 child World 执行。
- phase 切换、session 工具安装限制、旧 checkpoint 迁移和恢复后重新审批。

既有权限、审批、DSL、phase、序列化与可观测性测试已按新的职责和语义更新。

## DeepSeek Responses API 实网测试

测试文件：[test_tool_authorization_live.py](../../tests/live/test_tool_authorization_live.py)。

| 配置 | 值 |
| --- | --- |
| Base URL | `https://api.deepseek.com` |
| 模型 | `deepseek-flash` |
| 协议 | Responses API |
| 模式 | 非流式、流式 |
| 工具 | 只使用返回内存值的 probe 工具 |
| 调用边界 | 每个用例最多 4 ticks、180 秒；单次请求和流读取超时 60 秒 |

共测试 5 类策略，每类分别执行流式和非流式版本：正常允许、权限拒绝、审批拒绝、要求审批并批准、DSL 显式禁用。

结果：**10 passed，0 skipped，0 failed**，用时 36.48 秒。

所有用例均由真实模型生成一次 probe 调用，再读取实际工具结果或拒绝结果并生成最终回答。测试断言最终运行成功、调用参数正确、消息协议闭合、handler 调用次数、审批次数及执行与拒绝事件数量。每条助手消息都验证 provider output 的协议为 openai_responses；测试使用显式 store=false 禁止静默退回 Chat Completions。

复现命令使用环境变量提供凭据：

```bash
ECS_P1_LIVE=1 \
LLM_BASE_URL=https://api.deepseek.com \
LLM_MODEL=deepseek-flash \
uv run pytest -q tests/live/test_tool_authorization_live.py --tb=short
```

运行前在环境中设置 LLM_API_KEY。测试文件与本文均不保存凭据。

## 迁移注意事项

- Python 调用原来以 `allowed_tools=[]` 表示不限制时，应改为 None。
- 旧 DSL 原来以 `tools: {}` 表示不限制时，应删除该字段；现在空映射明确禁止所有工具。
- 旧 checkpoint 已丢失的 DSL 禁用信息无法从空名单恢复，需要从原 DSL 重新编译。
- 预检查 system 不再删除 pending calls 或直接追加拒绝消息；执行器负责消费队列与结果写入。

本次完成方案中的 P1 范围。审查记录中的 skill 注册表覆盖、Runner 多次运行边界、工具类型推导和高层装配接口等 P2 项仍保留为后续工作。
