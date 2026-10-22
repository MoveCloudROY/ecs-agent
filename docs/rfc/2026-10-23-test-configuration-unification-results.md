# 测试配置统一重构与验证记录

日期：2026-10-23\
状态：重构完成，离线回归与指定 DeepSeek Responses 场景验证通过\
关联规划：[测试配置统一与执行流程整理规划](2026-10-21-test-configuration-unification-plan.md)\
使用指南：[测试运行与配置](../testing.md)

测试现已通过统一 profile、pytest 执行策略和模型工厂管理 LLM 配置。普通 pytest 默认离线；显式 live 在请求前验证配置，实际请求错误作为失败报告。CI 与 Release 已切换到相同入口，Release 使用逐项必测清单。

本次没有修改生产库或 examples 的接口。替换失效凭据后，DeepSeek Responses 分批执行的 83 个不同用例均已通过，其中包括 Release 的全部 21 项核心必测。该结果只覆盖本次配置和选择范围，不代表其他供应商、协议或全部 live 能力均已验证。

## 实施范围

| 部分 | 实际变更 |
| --- | --- |
| 配置 | [config.py](../../tests/support/config.py) 集中解析 TOML、具名 profile、角色、超时、素材、服务与旧变量；禁止隐含供应商默认地址和冲突拼接 |
| pytest | [pytest_plugin.py](../../tests/support/pytest_plugin.py) 提供 `--live`、`--live-config`、`--llm-profile`、`--live-required`、`--live-report`；统一选择、预检、超时、脱敏与覆盖报告 |
| 生命周期 | [models.py](../../tests/support/models.py) 使用现有 Model/provider，按测试创建与关闭；fixture 先取消并等待当前测试新增的后台任务，再关闭客户端 |
| 子进程 | [environment.py](../../tests/support/environment.py) 为离线子测试移除服务凭据与 live 开关，为真实 example 显式转换一个 profile；不修改宿主全局环境 |
| 真实测试 | 全部 pytest 真实服务用例迁入 `tests/live/`；直接构造器和 registry 测试保留原被测入口，统一登记资源清理 |
| 独立审计脚本 | `tests/cache_audit/run_branching.py` 使用同一配置解析器，并要求显式 `--live`；离线分析仍可单独运行 |
| CI | 保留 base/MCP 矩阵与锁定依赖，保存 JUnit 报告 |
| Release | 使用受版本控制的完整 Responses profile 和 21 个必测 node ID；缺项、跳过、xfail、失败均阻止发布 |
| 说明 | 新增权威测试指南，同步 README、模型、metrics、Langfuse、prompt caching 和 logging 文档 |

旧变量仍由中央兼容层支持一个迁移阶段。新配置文件生效时，旧连接字段不覆盖它；旧功能开关不能替代 `--live`。兼容层没有恢复原有按文件选择供应商默认值的行为。

## 覆盖迁移核对

[逐用例清单](../../tests/live/inventory.json) 保存原 node ID、当前 node ID、live 分类和协议/能力/服务要求。

- 共核对 133 个原有相关参数化用例，均有迁移后的对应项，没有丢失。
- 124 个真实服务用例统一使用 `live` marker。
- 9 个原先被凭据门控、实际不联网的用例迁回离线测试，保留断言和参数组合。
- 在原规划列出的文件之外，还发现并迁移了 6 个分散在 hashline、集成功能、subagent result、plan/task CLI 和 UI design 混合文件中的真实服务用例。

迁移同时修复了被门控隐藏的问题：控制器流程测试需要先提供计划文件才能完成 finalize；出站请求捕获改用 HTTP hook，避免破坏 Responses SDK 的客户端接口和连接关闭。缓存指标缺失测试改为验证失败语义；后台 subagent 的真实服务用例要求成功，不能将服务失败或超时当作通过。

真实接口验证还发现并修复了旧测试驱动与断言的问题：

- 阶段切换和多轮对话通过 `AgentHandle.run_turn()` 开始新轮次，替代在已终止的 entity 上追加消息后重复运行；增加第二轮记住 Alice 的内容断言。
- 提示词和 slash skill 测试分别检查用户输入与其后的上下文池消息，验证模板、内容和顺序；slash skill 测试补上真实模型调用，验证实际出站消息被接口接受。
- 调试会话显式停在提问边界并回答一次，避免自动答题把草稿访谈扩展成整个审阅流程。
- CLI 输入按空行结束一个多行命令；工具问答与 REPL 命令分开驱动，并显式暂停访谈，再验证 `/plan:status` 确实执行、输出阶段与持久化状态一致。
- tmux 测试使用唯一会话名、检查实际创建成功，并等待清理子进程结束。

上述修复没有提高用例超时、添加失败重试或把错误改成 skip。真实服务断言失败仍然阻止通过。

## 离线验证结果

| 环境或检查 | 结果 |
| --- | --- |
| Python 3.11，base 依赖，全量 pytest | 3434 passed，4 skipped，124 deselected，4 xfailed |
| Python 3.11，MCP 依赖，全量 pytest | 3435 passed，3 skipped，124 deselected，4 xfailed |
| 本地 Python 3.12，设置 `LLM_API_KEY` 哨兵值，全量 pytest | 3437 passed，1 skipped，124 deselected，4 xfailed |
| 新配置与 pytest 基础设施 | 38 passed；覆盖预检、选择、清理、超时、脱敏和覆盖门槛 |
| 配置、pytest、缓存结果、计划场景与文档针对性回归 | 57 passed |
| 严格 mypy，生产源码与测试支持层 | 189 个文件通过 |
| Release 必测清单收集验证 | 21 个必测 node ID 均存在并被选择 |
| 工作流 YAML、profile TOML、文档链接及差异空白检查 | 通过 |

三个全量环境均无失败。4 个既有 conversation-tree 预期失败未改变；skip 数量差异来自可选依赖与运行环境。124 个 deselected 是默认离线规则产生的明确选择结果，不是服务失败后的跳过。

哨兵密钥验证用于确认真实凭据的存在不会自动启用任何 live 用例。基础设施回归还验证了任务取消发生在模型连接关闭之前，并在 Python 3.11 的两组依赖环境补跑。

## 真实 DeepSeek 验证

运行配置为 `https://api.deepseek.com`、`deepseek-flash`、`openai_responses`。凭据通过运行时注入，未写入配置或仓库；输出日志经过脱敏。

执行命令的非敏感部分为：

```bash
uv run pytest --live --live-config tests/live/profiles.example.toml \
  -m "live and (smoke or authoring) and not vision" \
  --live-required tests/live/required-responses-core.json \
  --live-report .artifacts/live-core.json -q

uv run pytest --live --live-config tests/live/profiles.example.toml \
  -m "live and not (smoke or authoring or vision or cache)" \
  --ignore tests/live/test_langfuse_observability_live.py \
  --live-report .artifacts/live-extended.json -v
```

最终结果按 node ID 去重，并采用修复后的定向复测结果：

| 范围 | 结果 |
| --- | --- |
| Responses 基础 smoke | 2 passed；后续也包含在核心必测中 |
| Release 核心必测 | 21 passed、8 not applicable；必测清单全部满足 |
| 扩展场景 | 62 passed、20 not applicable；包含工具、子 agent、上下文压缩、会话、提示词、metrics 和工作流 |
| 合计，不重复计算 smoke 与复测 | **83 passed、28 not applicable，无剩余失败** |

这是分批验证的汇总，不是一次完整 live 执行的 pytest 原始总计。扩展首轮为 55 passed、7 failed、20 skipped；修复后的定向复测覆盖全部 7 个失败用例，并验证了 tmux 清理与 CLI 驱动变更。最终配置基础设施回归仍为 38 passed。

初始密钥的请求曾返回 `401 Unauthorized / authentication_error`，测试如实失败。用户补充有效凭据后，请求在 `/responses` 成功执行。密钥仅在测试进程内存与子进程环境中使用，完成后已关闭持有凭据的运行进程。

原始 JSON 报告位于本地忽略目录 `.artifacts/`：`live-core-valid.json`、`live-extended-valid.json`、`live-repairs.json`、`live-cli-repair.json`。`live-validation-combined.json` 保存上述来源及按 node ID 合并的最终结果；原始失败记录未被覆盖。

## Release 范围与剩余验证

当前发布必测项为 2 个 Responses 基础构造/调用用例，以及 19 个 authoring 参数组合，配置与清单分别位于：

- [profiles.release.toml](../../tests/live/profiles.release.toml)
- [required-responses-core.json](../../tests/live/required-responses-core.json)

Chat、Anthropic、图像、缓存、并行工具、第二模型与外部 Langfuse 服务需要各自配置，目前未纳入发布必测范围，也未在本次真实接口验证中通过。所选集合内有 28 项因协议或能力不适用而跳过；另外 13 项图像、缓存及 Langfuse 场景未选择。响应链能力没有独立的 pytest live 用例，不能仅通过声明 capability 计入真实覆盖。

GitHub workflow 的变更已在本地进行语法与命令收集验证；上述结果来自本地运行，远端验证以推送后的 Actions 结果为准。Release 仍需仓库 Secret `LLM_API_KEY` 保存有效凭据，本地验证不会更新该 Secret。离线全量测试保留 4 个原有 conversation-tree 预期失败，本次不将它们计为已修复。
