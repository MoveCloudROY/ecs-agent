# DeepSeek Responses 上下文缓存对照

日期：2026-10-18。接口：`https://api.deepseek.com/responses`。请求与响应模型均为 `deepseek-flash`。

## 结论

**DeepSeek 直连的实测缓存复用接近发送前登记的条件预期，显著优于上一轮网关样本。同一套 Agent 消息链路可以实现接近理想值的复用，但本次更换了服务端和模型，不能据此唯一锁定网关代码为根因。**

| 指标 | 原网关 `gpt-5.6-sol` | DeepSeek 直连 `deepseek-flash` |
|---|---:|---:|
| 用户轮次 / HTTP 请求 | 23 / 25 | 23 / 25 |
| 普通 / 流式请求 | 16 / 9 | 16 / 9 |
| 工具调用 | 2 | 2 |
| 单次输入范围 | 11,797—19,564 token | 11,814—19,375 token |
| 全部请求理想复用基线 | 88.69%（事后补算） | **88.72%（逐请求发送前登记）** |
| 服务端报告的观测比例 | 57.71%（网关扩展字段） | **88.43%**（标准字段） |
| 标准 cached_tokens 比例 | 54.40%（网关计费分配值） | **88.43%** |
| 观测比例低于理想基线 | 30.98 个百分点 | **0.29 个百分点** |
| 排除两个冷对照后的理想 / 实测 | 95.94% / 62.43% | **96.02% / 95.70%** |
| 23 次可复用历史请求正命中 / 零命中 | 16 / 7 | **23 / 0** |
| 三次完全相同请求重发的观测比例 | 68.70%、77.86%、99.45% | **99.12%、99.12%、99.12%** |
| 客户端历史前缀、回放与隔离检查 | 通过 | 通过 |

DeepSeek 总输入为 **410,513 token**，预测可复用 **364,215 token**，实际报告缓存 **363,008 token**，差 1,207 token。25 次请求全为 HTTP 200，23/23 轮标识回忆正确，原始 usage 与框架调用事件逐项一致，连接全部关闭。

原网关比例继续分别保留 observed 与 billed 口径，不能拿其 54.40% 计费比例直接当作物理缓存命中率。DeepSeek 本轮响应没有 `linapi_cache_billing` 扩展字段，采用官方定义的标准 `input_tokens_details.cached_tokens`；它仍是服务端上报值，不是独立读取服务端 KV cache。

## 执行前登记的预期

本次修正了上一轮“只报实测、事后补基线”的问题：

1. 首个请求发出前，`config.json` 写入预测公式、假设、请求上限与调度方式。
2. 每次请求到 HTTP transport 后、发送到服务器之前，计算并保存 `predicted_reuse`。
3. 预测只读取此前请求的输入 token 数和已发送 JSON，不读取 cached_tokens 来拟合结果。

公式：

- 首次请求、修改开头 instructions 的冷对照：预期缓存为 0。
- 连续对话、工具续轮、分叉、切回、恢复：验证与实际祖先的输入前缀和稳定参数一致后，以祖先的 input_tokens 作为保守近似；不额外计入父响应输出可复用部分。
- 与此前完整 JSON 相同的重复请求：以此前相同请求的 input_tokens 作为预期，接近全量复用。

假设是匹配缓存已写入、可到达、未被淘汰且服务端渲染稳定。这是可复用前缀的条件基线，不是对后端调度概率的保证。最终比例的分母使用实际输入 token 数，适配不同模型 tokenizer。

DeepSeek 官方说明缓存按已持久化的完整前缀单元匹配，缓存构建需要数秒，采用 best-effort 策略。本次没有额外预热、等待或命中失败重试。[Context Caching](https://api-docs.deepseek.com/guides/kv_cache/)

## 保持相同的场景

沿用[上一轮专项测试](2026-10-18-context-cache-branching-audit.md)：初始 220 条合成档案，每轮新增 18 条；8 轮主线、从第 4 轮分出 A/B、从 A1 再分出 C、切回 A、磁盘保存恢复、相同请求重发三次，最后修改开头前缀一次。

使用同一份 Agent、Runner、ReasoningSystem、ToolExecutionSystem、WorldSerializer 和 OpenAI SDK 实现。`store:false`、自动 response 续接关闭，完整历史通过 Responses 发送。DeepSeek 的无状态接口与这个场景一致；不支持的 `include` 等兼容参数按其文档忽略。[Responses API](https://api-docs.deepseek.com/api/create-response/)

各次运行使用新的随机 salt，防止此前实验的缓存影响初始冷请求。模板、资料条数、分支结构和执行顺序相同；salt、模型生成输出和 tokenizer 不同，所以这不是跨模型逐字相同的请求对照。总运行上限 30 分钟、HTTP 上限 32 次，实际用了 25 次，HTTP 轨迹约 24.72 秒。

## 关键分支与控制样本

| 请求 | 当前输入 | 发送前预期缓存 | 实测缓存 | 实测占输入比例 |
|---|---:|---:|---:|---:|
| A 首次分叉 | 15,235 | 14,400 | 14,336 | 94.10% |
| B 首次分叉 | 15,241 | 14,400 | 14,336 | 94.06% |
| 二级 C 首次分叉 | 16,068 | 15,235 | 15,232 | 94.80% |
| 切回 A4 | 17,719 | 16,889 | 16,896 | 95.36% |
| A 磁盘恢复 | 19,371 | 18,544 | 18,560 | 95.81% |
| 相同请求重发 1 | 19,371 | 19,371 | 19,200 | 99.12% |
| 相同请求重发 2 | 19,371 | 19,371 | 19,200 | 99.12% |
| 相同请求重发 3 | 19,371 | 19,371 | 19,200 | 99.12% |
| 修改开头前缀 | 19,375 | 0 | 0 | 0% |

小幅高于祖先输入的样本不违背基线，因为预测没有计入新增回放的父响应输出；对齐和服务端处理的具体贡献未独立测量。首次请求也为零。因此本轮所有零命中都发生在预先指定的两个冷对照中。

## 对“是否网关问题”的判断范围

- **支持的结论**：Agent 的这条 SDK、历史回放、分叉和恢复路径能在真实服务端实现接近理想的缓存复用。原网关样本的低复用不是这条客户端路径必然造成的。
- **已有直接证据**：原网关返回了 observed/billed 分配字段，标准 cached_tokens 是计费分配值；这部分指标差异确实发生在网关侧。
- **仍未锁定的部分**：原网关 observed 仍低于基线，原因可能涉及其路由、账户池、缓存持久化、上游模型处理或上报。本次不能在这些因素间做唯一归因。
- **严格定位所需对照**：同一模型分别通过直连和该网关执行同一场景，且保持账户缓存隔离和参数可比。本轮没有把 DeepSeek 的密钥发送给原网关，也没有声称完成该项实验。

本实验仍是平铺历史的快照分叉，未修复或重新验收此前失败的原生 ConversationTree 生命周期。

## 工具更新与证据

审计工具新增发送前预测，以及网关 observed/billed 的分开记录，生产代码未改动。

相关离线回归为 37 passed、4 xfailed；4 个 xfail 是此前已确认的原生树缺陷，不计为通过。两个审计工具文件通过严格 mypy，文档链接与 `git diff --check` 通过。

- [HTTP 捕获与预测实现](../../tests/cache_audit/branching.py)
- [运行 CLI 与预登记配置](../../tests/cache_audit/run_branching.py)
- [预测和分析器回归测试](../../tests/test_context_cache_branching.py)
- [DeepSeek 预登记配置](../../records/review/context_cache_branching_2026-10-18/deepseek_a/config.json)
- [带发送前预测的实际请求与响应](../../records/review/context_cache_branching_2026-10-18/deepseek_a/requests.json)
- [DeepSeek 逐请求报告](../../records/review/context_cache_branching_2026-10-18/deepseek_a/report.json)
- [两组对照汇总](../../records/review/context_cache_branching_2026-10-18/deepseek-comparison.json)
- [DeepSeek 运行日志](../../records/review/context_cache_branching_2026-10-18/deepseek_a.log)

原始证据目录被 Git 忽略，报告与测试代码可纳入版本管理。密钥仅用于临时认证，不写入仓库或证据；测试后清理临时认证文件。

在不调用 API 的情况下重算：

```bash
uv run python -m tests.cache_audit.run_branching --analyze-only \
  --output records/review/context_cache_branching_2026-10-18/deepseek_a
```
