# Agent 消息流程修复执行记录

日期：2026-10-18。状态：本轮 RFC 修复和验证已完成；真实远端续接成功按用户确认保留为受服务端限制的未验证项。

完整范围以[修复 RFC](2026-10-17-agent-message-flow-repair.md)为准；本文记录当前工作树的实际进展，不缩减其验收要求。

## 已落实的行为

### A：Responses 原始输出保留与回放主路径

- 新增 `ProviderOutput`，以可序列化 JSON 保存有序 output items、协议与来源、完整性、response ID 和存储信息；核心消息不持有 SDK 对象。
- 非流式和流式均保留 reasoning opaque 字段、原始工具 arguments、item ID 与 call ID。流式按 output index 汇聚，以最终完整输出校正结果并避免重复工具分派。
- 同来源且展示内容未被修改时，下一轮直接回放原始输出项；来源不匹配或消息被编辑时使用可移植内容投影，不向其他 provider 发送旧 opaque 状态。
- 序列化保留协议状态与历史身份；修复序列化已初始化 SDK 和绑定工具 handler 时深复制运行时对象的问题。
- 压缩跳过待执行工具，删除旧工具段时要求所有调用均已获得回执；完整工具身份与出站历史边界校验已补齐，见后续边界修复。

主要新增文件：`src/ecs_agent/providers/responses_state.py`、`tests/test_responses_replay.py`。

### B：存储语义与续接验证主路径

- `enable_store=True/False/None` 分别发送 true、false、省略 `store`；`None` 不发送 JSON null。独立的 `enable_response_chaining` 默认关闭。
- 显式 previous response 引用不依赖本次 `store`；本次 `store=False` 的结果不成为新的自动续接锚点。未知存储状态不能仅凭 response ID 判定为可续接。
- 自动续接校验 provider / endpoint / model / 凭据作用域、history ID / revision 和实际协议输入前缀；覆盖边界按协议项计算，不能仅凭历史消息条数跳过输入。
- 新增统一历史替换与失效入口，工作流重置、压缩、旧内容编辑和来源变化不会继续使用不匹配的旧链；快照恢复后重新验证来源与前缀。
- 明确归因于 previous response 的 400 / 404，仅在尚未交付输出、调用方提供完整历史且可无损回放时进行一次无链重试。此类 404 不触发 Responses → Chat 回退；显式存储要求也不会在兼容回退中被删除。
- SDK 实际请求测试覆盖三态存储、独立续接开关、正常增量续接、RetryModel 包装、快照恢复、消息重新分段、失效重试和无关 HTTP 错误。

主要新增文件：`src/ecs_agent/history.py`、`tests/test_responses_chaining.py`。配置、registry、example 环境变量及 Responses / models 文档已同步主路径语义。

### C：生成终态主路径

- 新增 `GenerationStatus`，区分 completed、tool_calls、incomplete、filtered、cancelled、unknown、error。消息、LLM 调用 / 观测事件和 StreamEnd 保留终态与 finish reason。
- Reasoning 不再把截断、过滤、取消、未知 EOF 当作 reasoning_complete，也不执行这些结果中的工具调用。非法 JSON 参数不会被静默转换成可执行的空字典。
- Responses 的 failed 状态作为带 error 的生成结果传递，保留服务端 usage；HTTP / 网络异常仍按异常处理。
- 用量完整性与回答完整性分开。中断前已经收到的 usage 保留；没有 usage 的 EOF 不再标为 COMPLETE。辅助流式调用在消费结束后才发布用量终态。
- 部分消息带状态写入历史并支持序列化。写入部分输出前提交本次已发送的上下文，避免重试重复注入。
- 交互前端报告可恢复的失败后重新等待用户输入；子 Agent 保留终态供结果判断。
- 保留非流式旧 provider 的完整 CompletionResult 兼容契约；流式 EOF 不使用该兼容推断。

主要新增文件：`src/ecs_agent/outcomes.py`、`tests/test_generation_outcomes.py`。

### D：结构化运行与委派结果主路径

- `Runner.run` 返回 `RunResult`，同时保存在 `last_result` 和 `RunCompletedEvent.result`；保留实际终止原因、目标实体、最终 / 部分消息和错误。
- 委派明确指定 `target_entity`；其他实体或 child stub 的终止不会被当作目标子任务完成。
- 成功要求存在完成的最终消息，且没有待执行工具。max_ticks、失败、取消和不完整结果不能由最后一条 assistant 文本冒充成功。
- `DelegationResult` 从运行快照派生，保留兼容的文本 / tuple 适配；失败中的中间文本标记为 unfinished。
- 同步 / 后台、完成事件、session/result 与通知消费同一终态。后台明确保留 cancelled 与 failed 的区别，部分输出不会写成成功产物。
- 直接取消后台任务时补充 runtime terminal signal，唤醒 result waiter，并发出非成功的完成事件。同步 timeout / cancel 也补充完成事件。
- 观测根记录使用真实任务结果，而非仅凭 runner 正常退出推断成功。

主要新增文件：`src/ecs_agent/core/run_result.py`、`src/ecs_agent/systems/subagent/results.py`、`tests/test_run_outcomes.py`。

### 后续边界修复

- 工具调用在 Reasoning / Planning 入队前校验已提供的工具名称、调用身份、JSON 参数及 JSON Schema Draft 2020-12。执行层再次校验手工加入的队列；身份缺失或重复的批次不执行任何工具。Schema 支持本地引用，禁止隐式网络获取远端 schema。新增显式依赖 `jsonschema` / `referencing`。
- 出站历史校验覆盖孤立回执、重复调用 / 回执、乱序和未闭合工具段。显式 Responses 增量输入允许开头的远端工具回执，声明完整本地历史时仍使用完整校验。损坏历史在 HTTP 请求前失败。
- 辅助模型流支持显式 `aclose()`，包括尚未开始消费的流；关闭和取消会释放底层连接、只发布一次终态，不再将提前关闭记为成功。工具分片与 Reasoning 共用聚合逻辑，缺失初始 call ID 不再自动伪造身份。
- Planning 的不完整结果不推进计划；Replanning 不用可解析但截断的 JSON 替换计划；Compaction 不用截断或空摘要替换历史及既有摘要。用量事件仍保留该次调用的实际状态和费用信息。
- 同步与后台委派在外部取消 / 超时后保留 child RunResult 的部分输出；完成事件、session/result 和组件快照一致。取消后台任务先等待子任务清理，再释放槽位并唤醒等待者。
- 使用明确的 deadline scope 区分超时与外部取消，child RunCompletedEvent.result 与委派结果分别保持 `timeout` / `cancelled` 的一致语义；父 World 的 child stub 历史在取消路径也同步。
- 非文本最终结果在旧文本委派接口中以带类型的文件 / 图片引用 JSON 表达，不返回空成功文本。旧 assistant 后出现新用户输入，或历史存在未闭合工具段时，不得作为本次有效最终结果。
- 修正原有两处类型标注：同步工具 wrapper 的字符串返回契约，以及 Claude system 内容块的返回类型；严格 mypy 已通过。

新增边界回归：`tests/test_auxiliary_outcomes.py`、`tests/test_tool_contracts.py`，以及 `tests/test_run_outcomes.py` 的取消 / 超时 / 非文本结果矩阵。

### 真实 example 运行发现的附加问题与修复

1. **辅助 workflow ID 调用漏计费、接受截断结果**：生成 slug 的调用接入统一 LLMInvocationEvent，使用 `workflow_id` operation；不完整、错误或工具调用结果不作为合法 slug，回落到确定性标识。
2. **任务边界绕开历史失效入口**：任务开始 / 恢复的会话重置也调用统一历史替换函数，更新 history identity 并使旧续接锚点失效。
3. **tool sink 隐藏模型需要的内容**：`read_file` 和子 Agent 的控制 / 结果工具使用 `inline_result=True`，避免读取归档又得到归档路径的递归。失败工具结果仍归档，但错误文本直接进入模型历史；普通成功工具仍按既定归档策略处理。
4. **评审反馈恢复时被截断**：持久化完整非批准评审意见，不再只保留前 500 字符；批准状态依旧按原契约保存 `notes=None`。
5. **编辑行号易错**：`read_file(line_numbers=True)` 返回含空行的绝对行号，默认纯文本与快照行为保持兼容；规划提示词要求先读取行号再编辑。
6. **受保护上下文超限缺少运行终态**：组装器以 `ContextBudgetExceededError` 表达预算契约，Reasoning / Planning / Replanning 生成 `budget_exhausted`，不发送模型请求；错误清理后 RunResult 仍保留该状态。
7. **评审回调打断工具调用 / 回执顺序**：真实 `replay_d` 在评审轮数上限路径复现重复工具回执。回调发生在工具返回前，直接追加用户提示会导致历史修复器先补占位回执，再出现真实回执。QA 通过后自动触发 writer 也有同一问题。现改为上下文队列，在工具回执全部写入后的下一次模型调用注入；example 启用 context pool。切换工作流 / 任务清理旧 workflow context，离开 WRITE_PLAN 清理未消费的 writer 提示，保留调用者上下文。
8. **模型误读成功工具的归档引用**：`replay_h` 中 bash 的实际通过输出已归档，但模型没有读取路径，声称结果无法验证并仍调用 `complete_task`。新增明确的归档读取提示：路径不是空 stdout，必须读取 YAML content，引用实际记录作为证据，未验证时不得调用 complete_task。另以 `verification_i` 验证模型能读取真实归档。独立 unittest 和五组输入输出验证已通过。该提示改进不构成业务正确性的机器证明；`complete_task` 仍是调用者完成声明的记录器，不是任意业务验收的独立裁判。原先提前声明完成的记录保留，不以最终测试通过掩盖当时的行为。

`tests/integration/test_plan_task_sdk_cache.py` 使用真实 SDK + MockTransport 覆盖上述 example 请求顺序、评审成功 / revise / incomplete / cancelled / failed / 无终止事件、sink 可读性，以及先成功续接再执行重置 / 压缩 / 模型切换的失效矩阵。它验证实际 HTTP JSON，但不作为真实网关命中率或远端存储成功的证据。

## 审计反例状态

原八个反例已迁入正式测试文件 `tests/test_message_flow_contracts.py`，未改低其断言要求。

| 反例 | 当前状态 |
|---|---|
| incomplete 流不应正常完成 | 通过 |
| cancelled 流不应正常完成 | 通过 |
| 无终止事件 EOF 不应正常完成 | 通过 |
| 子任务后续失败不能返回中间文本 + success | 通过 |
| 原始 reasoning item 工具续轮回放 | 通过 |
| 重置工作流后失效旧链 | 通过 |
| 压缩历史后失效旧链 | 通过 |
| 跨 provider 切换失效旧链 | 通过 |

三个旧链反例显式开启自动续接，并由正常追加可以续接的正向用例配套验证，避免仅靠默认关闭续接使测试通过。

## 验证证据

日志位于本地目录 `records/review/message_flow_repair_2026-10-18/`，该目录可能被 Git 忽略。

- `contracts-before.log`：修复前八个契约反例全部失败。
- `full-stage-c-after.log`：C 主路径完成后，排除整份审计反例文件的回归，2998 passed、112 skipped、6 warnings。这不是最终全量验收。
- `stage-d-followup.log`：D 主路径及相关回归，292 passed、1 skipped。
- `full-through-d.log`：当时完整测试结果为 3010 passed、112 skipped、7 failed；其中四项是未实施的 A / B，另外三项是事件位置参数兼容与观测断言。位置参数已恢复兼容，观测断言已按任务语义更新。
- `stage-cd-final.log`：C / D、事件类型、观测和八个反例的后续组合验证，391 passed、1 skipped、4 failed；四项失败均为尚未实施的 A / B。
- `mypy-stage-d.log`：检查 155 个源码文件，仅剩基线中的两处错误：`tools/discovery.py:113` 与 `providers/claude_model.py:98`。
- `stage-ab-contracts.log`：八个正式审计反例与当时的原始输出回放用例全部通过，共 13 passed。
- `stage-ab-final.log`：A / B 及相关配置、序列化、事件、观测和示例集成验证，261 passed。
- `full-through-ab-final.log`：A / B 主路径完成后的完整回归，3046 passed、112 skipped、6 warnings。跳过项和未补齐的边界仍不属于已验证范围。
- `mypy-stage-ab.log`：检查 157 个源码文件，仍仅有上述两处基线错误；严格类型检查尚未全绿。
- `boundaries-targeted.log`：工具身份、参数、HTTP 请求边界及子任务结果的组合验证，121 passed。
- `delegation-abort.log`：取消 / 超时部分结果与相关消费端，62 passed、1 skipped。
- `boundaries-followup.log`：共享流式工具聚合、deadline 终态、子任务与 Runner 回归，332 passed。
- `auxiliary-final.log`：辅助流提前关闭、工具片段、生成终态和用量观测，85 passed。
- `full-boundaries.log`：边界修复阶段的完整回归，3096 passed、112 skipped、6 warnings；后续又补充了 deadline 终态和共享工具聚合的验证，以最终日志为准。
- `full-boundaries-final.log`：包含本轮全部代码修复的最终完整回归，3098 passed、112 skipped、6 warnings。跳过项不作为已验收证据。
- `mypy-boundaries-final.log`：159 个源码文件通过，已无类型错误。
- `provenance-boundaries.log`：91 passed，直接验证 previous response 错误发生在首个 delta 后不重试；流式 / 非流式请求进行中切换 endpoint、凭据与存储配置仍保留原请求来源；待执行原始工具输出不被压缩，checkpoint undo 后按原协议回放；未闭合多工具段不能被部分裁剪。
- `diff-check.log`：`git diff --check`，无输出表示通过。

关键命令：

```bash
uv run pytest
uv run mypy src/ecs_agent/
uv run pytest tests/test_message_flow_contracts.py -q --tb=short
uv run pytest tests/test_run_outcomes.py tests/test_subagent.py tests/test_subagent_cancellation_wake.py tests/test_subagent_result_future.py tests/test_runner.py tests/test_types.py tests/test_observability_trace_lifecycle.py tests/test_langfuse_event_mapping.py tests/test_generation_outcomes.py tests/test_message_flow_contracts.py -q --tb=short
```

### 新增验证记录

- `full-live-feedback-final.log`：3117 passed、112 skipped、6 warnings；包含真实运行反馈引出的 inline 工具结果、完整评审意见、行号读取、辅助计费和预算终态修复，早于回调顺序修复。
- `mypy-live-feedback-final.log`：159 个源码文件通过。
- `callback-before.log`：真实 SDK 回归在 QA 批准和评审轮数上限两条回调路径均失败，复现工具段被打断。
- `callback-final.log`：回调修复及 example 相关集成回归 323 passed、2 skipped。评审上限拒绝提交时保留原阶段；测试明确验证真实回执仅一次，控制提示位于回执之后。
- `callback-boundary-final.log`：14 passed，另验证重置时删除旧 workflow 提示和关联 reservation，但保留调用者上下文并实际发送。
- `full-callback-final.log`：3122 passed、112 skipped、6 warnings；当前代码的完整离线回归通过。警告来自现有 structlog 异常格式配置，跳过项不记为通过。
- `mypy-callback-final.log`：159 个源码文件通过严格类型检查。
- `full-archive-guidance-final.log`：包含最后的归档读取提示修订，3122 passed、112 skipped、6 warnings。
- `mypy-archive-guidance-final.log`：159 个源码文件通过。最终 `git diff --check` 通过。

### 真实接口边界

原始请求 / 响应、调用事件、各轮快照与计费记录保存在 `records/review/message_flow_live_2026-10-18/`。该目录被 Git 忽略，不含 API key 或认证头。

| 运行 | 实际范围与结论 |
|---|---|
| `replay_a` | 7 次 HTTP，复现 sink 读取递归后停止；不是完整流程通过 |
| `replay_b` | 44 次 HTTP，完成真实访谈，因反复错误行编辑停止；不是完整流程通过 |
| `replay_c` | 25 次 HTTP，恢复同一工作流、真实 advisor revise，暴露反馈截断与读取问题后停止 |
| `replay_d` | 35 次 HTTP，真实 advisor 多轮修改，在评审上限提示路径复现重复工具回执；回调缺陷已修复 |
| `chain_probe` | 5 次 HTTP / 4 轮，原网关对 3 个 `store:true` 请求均返回 `store:false`；未建立自动续接锚点 |
| `new_endpoint_chain_probe` | 用户提供的新端点完成 5 次 HTTP / 4 轮；3 个 `store:true` 均返回 `store:false`，2 个显式 `store:false` 保持 false；均采用完整回放 |
| `new_endpoint_explicit_probe` | 直接使用 SDK 共 2 次请求：首个 `store:true` 返回 false；显式 `previous_response_id` + `store:false` 返回 HTTP 400，服务端说明 HTTP 续接需要 OpenAI API-key 类型账户 |
| `replay_e` | 在新端点恢复旧工作流；评审上限错误提示已正确出现在工具回执之后，无重复回执。旧工作流已耗尽非批准评审轮次，模型要求重新开始；主动中断避免重复无进展调用，未修改批准记录 |
| `replay_f` | 新隔离工作区完成访谈并经历三轮真实 advisor revise；第 56 次请求发生网关 ReadError，正确记录 error + 未知 usage 并返回用户输入边界。55 次有 usage 的响应与框架逐项一致，billing 总计一致；未通过完整流程 |
| `replay_g` | 37 次请求：恢复 F 的工作流，真实 advisor / QA / plan writer / plan QA 全部完成，三个门禁批准、到达 PLAN_FINALIZED；重新加载后的第 37 次请求 ReadTimeout。36 次有 usage 的响应和 billing 一致。第 17 次主请求明确先收到 QA 工具回执，再收到 writer context，无重复回执 |
| `replay_h` | 11 次请求：通过 `/task:start <workflow_id>` 从磁盘自动加载同一批准计划，生成代码与五组 subtests，运行测试并调用 complete_task，达到 TASK_COMPLETED。11 次 usage 全部一致。模型未读取归档便声明完成的语义边界及后续核验见下文 |
| `verification_i` | 8 次请求：模型读取真实 bash 归档，确认默认 unittest 和显式 discovery 均退出 0，写入 post_completion_verification.md；原声明保留。8 次 usage 和 billing 全部一致 |

两个端点均不能用当前账户证明有效远端续接成功。用户已明确选择“记录服务端限制，保留此项未验证”；该项作为已接受的外部验证边界，不再阻止其余修复和验收收尾。不能把忽略返回的 `store:false`、强行登记锚点或离线模拟成功当作真实续接验收；正向自动续接已有 SDK MockTransport 证据，真实服务端成功路径仍受账户能力限制。缓存零值与正值均按原值保留；下述真实运行的原始 usage、LLMInvocationEvent 与 billing 已按响应身份完成最终核对。

已完成的缓存样本：新端点五次探测总输入 19,864 token，cached 11,136，3 次正值、2 次零值，加权缓存比例 56.06%；`replay_f` 的 55 个已报告 usage 的响应总输入 916,803，cached 485,972，42 次正值、13 次零值，加权比例 53.01%。其第 56 次网络失败没有 usage，单列为未知，未混入零缓存样本。这些是不同场景的实测统计，不能用于比较客户端优劣或承诺未来命中率。

F / G / H 是同一个持久化工作流 `normalize-label-whitespace-tests` 的访谈、评审、恢复与执行轨迹，不是一次无中断成功运行。共 104 次 HTTP 尝试，其中 102 次有 usage，2 次网络失败无 usage；三段的 billing 都与各自调用事件一致。F / G / H 中分别有 50 / 28 / 10 对可比较的同上下文相邻请求，全部保持输入历史前缀，并逐项回放前次原始输出（包括 reasoning / function_call）；没有工具段顺序错误。这些配对排除了恢复、辅助调用与阶段提示不同的不可比较边界。

执行产物在 `replay_f/workspace/`。独立运行 `python -m unittest -v` 通过（1 个 unittest 方法、5 个 subtests）；另独立调用函数验证五组要求的输入输出全部通过，记录为 `independent-unittest.log` 与 `independent-cases.json`。H 的真实命令归档也包含通过输出及 `EXIT:0`。工作流状态与产物已完成；模型 H 当时未检查归档的事实仍需与结果正确性区分。

### RFC 逐项验收对应

| 要求 | 直接证据 | 当前结论 |
|---|---|---|
| 八个原反例 | `test_message_flow_contracts.py`，原失败日志与当前完整回归 | 离线通过 |
| 三态 store、独立显式引用、自动策略 | `test_responses_chaining.py::test_store_maps_directly_to_sdk_without_controlling_explicit_reference` 等；新旧端点存储探测 | SDK HTTP 请求语义通过，真实存储成功受账户能力限制 |
| 普通 / 流式 / 工具续轮 / save-load | `test_responses_replay.py::test_raw_output_survives_tool_loop_and_serialization`；真实请求中原始输出项逐项回放核对 | 离线通过，真实已采样消息 / reasoning / function_call |
| reset / compact / trim / restore / provider 切换 | `test_responses_chaining.py`、`test_responses_replay.py`、example 的 `test_example_invalidates_verified_chain_after_context_boundary` | 离线通过，先验证正常续接再测试失效 |
| 首 delta 后禁止重试、来源切换、远端 opaque-only | `test_invalid_reference_error_after_stream_delta_never_retries`、`test_request_keeps_original_provenance_during_provider_rotation`、`test_expired_chain_with_remote_only_reasoning_does_not_claim_lossless_recovery` | 离线通过 |
| 生成终态与用量完整性独立 | `test_generation_outcomes.py`、`test_auxiliary_outcomes.py`；example 真实 SDK 子评审终态矩阵 | 离线通过 |
| child / background / waiter / event / 父回执一致 | `test_run_outcomes.py`、`test_subagent_cancellation_wake.py` 等；实际子评审和父工具回执 | 离线通过；真实子评审和主任务执行已完成 |
| 工具 schema、身份、闭合顺序及上下文预算 | `test_tool_contracts.py`、`test_auxiliary_outcomes.py`、`test_context_budget_exhaustion_survives_error_cleanup`、回调顺序回归 | 离线通过 |
| 完整测试与类型检查 | `full-callback-final.log`、`mypy-callback-final.log` | 通过，跳过项和 live 边界单列 |
| 完整 plan/task、恢复和缓存复查 | 同一工作流 F → G → H，后续核验 I；独立测试与原始协议 / usage / billing | 已完成有恢复的完整轨迹；两次网络失败和提前完成声明明确保留；真实远端续接成功仍未验证 |

以上为逐项证据表。完整工作流达到 TASK_COMPLETED，产物与测试已独立核验；真实远端续接成功按用户确认保留为未验证。

## 最终交付与保留边界

- 已完成：A / B / C / D 契约修复、原八个反例、相关边界与 example SDK 请求回归、完整 pytest / mypy、同一真实工作流的访谈至执行和恢复、产物独立验证，以及原始协议 / usage / billing 核对。
- 后续核验文档：`records/review/message_flow_live_2026-10-18/replay_f/workspace/scratchbook/normalize-label-whitespace-tests/evidence/post_completion_verification.md`。它引用真实命令归档并保留旧声明；没有回写批准、修改计数器或伪造评审。
- 真实远端续接成功：按用户选择保留未验证。当前账户返回 store:false，显式 previous_response_id 返回 HTTP 400；未来有支持该能力的账户时可单独补测。
- 业务验收：生成 / 运行成功表示协议与运行终态满足结构契约，不证明自由文本结论正确。complete_task 是完成声明记录器。此次实际代码和测试已独立验证，但不能据此宣称该工具能自动判断任意验收标准；需要这种保证的应用应提供领域验证器。
- 原 RFC 8.3 范围保持不变：未证明 ConversationTree 全部分支读写、自定义系统和全部 provider 多模态组合。112 个跳过项也不计为通过。
- 原始审计目录被 Git 忽略，本文保存结论和证据定位。用户提供的认证信息未写入仓库、请求记录或文档。

## 提交前隔离验证（2026-10-18）

按用户要求整理本轮 SDK 迁移、消息流程修复、example 修复和缓存专项测试后，从 Git 暂存区导出完整源码快照，在独立目录执行验证；通过 `PYTHONPATH` 确认测试加载的是该快照中的源码。

- 待提交快照全量 pytest：**3114 passed、112 skipped、4 xfailed、6 warnings**。
- 待提交快照严格 mypy：**158 个源码文件通过**；两个缓存审计工具文件也单独通过严格 mypy。
- 完整工作树作为对照：3126 passed、112 skipped、4 xfailed；159 个源码文件通过 mypy。数量差异来自没有纳入本次提交的其他工作。
- 四个 strict xfail 均为已复现的原生 ConversationTree 生命周期缺陷，不计为修复通过；具体范围见[缓存分叉审计](2026-10-18-context-cache-branching-audit.md)。112 个跳过项及真实远端续接限制继续保留。
- MCP、日志和其他无关改动保留在工作区；必要的流式事件顺序依赖纳入本次修复。暂存内容通过 `git diff --cached --check`，未发现 API key 或私钥格式的凭据。
- 本次为离线提交验证，未重新调用收费模型。此前 DeepSeek 对照结论见[专项报告](2026-10-18-deepseek-context-cache-comparison.md)。本次操作仅创建本地提交，不推送远端。

验证日志：`records/review/message_flow_commit_2026-10-18/`，该原始记录目录沿用项目的 Git 忽略规则。
