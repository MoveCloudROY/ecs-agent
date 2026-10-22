# 测试运行与配置

普通 `uv run pytest` 只运行离线用例。宿主环境存在 LLM 或 Langfuse 密钥不会启用真实服务测试；真实用例必须显式传入 `--live`。所有模型配置在 `tests/support/config.py` 解析，模型创建与清理由 fixture 管理。

## 离线回归

```bash
uv sync --locked --group dev
uv run --no-sync pytest
uv run --no-sync mypy src/ecs_agent/ tests/support/

# 验证实际安装 MCP SDK 的依赖组合
uv sync --locked --group dev --extra mcp
uv run --no-sync pytest
```

普通 CI 分别在 base 和 MCP 环境运行离线全量测试。真实 LLM 用例显示为 deselected；可选依赖测试仍按自身依赖条件执行。

## 模型配置

复制 `tests/live/profiles.example.toml` 为已忽略的 `tests/live/profiles.local.toml`，或通过 `--live-config PATH` 指定文件。示例文件仅在显式选择时生效，不自动加载 `.env`。

```toml
[timeouts]
request_seconds = 60
stream_read_seconds = 60
case_seconds = 180

[profiles.deepseek-responses]
base_url = "https://api.deepseek.com"
model = "deepseek-flash"
api_format = "openai_responses"
api_key_env = "LLM_API_KEY"
capabilities = ["text", "tools", "streaming"]

[roles]
primary = "deepseek-responses"
```

将凭据注入 `api_key_env` 指向的环境变量。配置文件只保存变量名。每个 profile 必须完整声明端点、模型、协议和凭据引用；CLI 的 `--llm-profile NAME` 只切换整个主要 profile，不逐字段覆盖连接。

协议使用 `openai_chat_completions`、`openai_responses` 或 `anthropic_messages`。不同协议建立不同 profile，不能因测试需要另一协议而临时改写当前连接。`secondary`、`vision` 等角色可引用另外一个完整 profile；场景通过 `live(role="vision")` 指定主要角色，通过 `roles=["secondary"]` 声明额外角色。

能力词表为 `text`、`tools`、`streaming`、`vision`、`cache`、`response_chaining`、`parallel_tools`、`multi_model`。声明表示该配置应当支持相应能力，测试将验证行为。不要仅为消除“不适用”而添加未经支持的能力。模型切换测试需要 `multi_model` 与 `roles.secondary`；单一文本配置不需要额外凭据。

超时必须为有限正数。`request_seconds` 控制请求预算，`stream_read_seconds` 控制流式读取间隔，`case_seconds` 控制整个异步场景。profile 内的 `[profiles.NAME.timeouts]` 可覆盖全局值。直接 registry 构造测试使用其公开接口支持的请求超时；流式场景通过模型工厂传递独立读取超时。

## 选择真实测试

```bash
# 从本地配置中选择模型，执行基础协议检查
uv run pytest --live --llm-profile deepseek-responses -m "live and smoke and not vision"

# 执行全部 authoring 参数组合
uv run pytest --live --llm-profile deepseek-responses -m "live and authoring"

# 显式配置文件，限定具体场景文件
uv run pytest --live --live-config tests/live/profiles.example.toml \
  tests/live/test_compaction_live.py -v

# 查看选择结果而不请求服务；收集模式不要求密钥值
uv run pytest --live --live-config tests/live/profiles.example.toml \
  -m "live and authoring" --collect-only -q
```

`--live` 允许真实服务测试执行，`-m`、`-k` 和路径决定选择范围。未给 `-m` 时，离线和 live 用例可一起执行。选择完成后才校验实际适用场景需要的凭据、素材和依赖；不会要求未选择服务的凭据。

场景 marker 包括 `smoke`、`authoring`、`tools`、`subagent`、`compaction`、`session`、`prompts`、`workflow`、`vision`、`cache`、`observability`。每个测试通过 `live(...)` 声明协议、能力和依赖，`api_format` 参数化用例按各参数分别判定。一个 profile 不代表验证了所有协议。

| 结果 | 含义 |
| --- | --- |
| deselected | 未启用 live，或不在路径、marker、关键字选择范围内 |
| not applicable | 所选 profile 的协议或声明能力不适用于该用例；以带原因的 skip 单独汇总 |
| 预检失败 | 配置、凭据、角色、素材或必需依赖缺失；尚未发送请求 |
| failed | 请求认证、网络、限流、超时或服务错误，或行为断言未通过 |

显式 live 没有选中用例、没有适用用例或实际结果全部跳过时返回非零。请求错误不再转换为 skip，也不自动重跑整个 agent 场景。

## 扩展服务与素材

图像测试需要 `roles.vision` 引用声明 `vision` 能力的完整 profile，并配置素材：

```toml
[assets]
image_url_env = "IMAGE_URL"
# 也可以使用 image_url = "https://..."，两种方式不能同时设置。
```

Langfuse 与 LLM profile 分开配置：

```toml
[services.langfuse]
host = "https://your-langfuse-host.example"
public_key_env = "LANGFUSE_PUBLIC_KEY"
secret_key_env = "LANGFUSE_SECRET_KEY"
```

```bash
uv sync --locked --group dev --extra langfuse
uv run --no-sync pytest --live --live-config tests/live/profiles.local.toml \
  tests/live/test_langfuse_observability_live.py -v
```

协议不匹配的 Langfuse 用例不适用；协议匹配的选中用例必须具备 SDK 与服务凭据。Prometheus 不需要 Langfuse 凭据，交互式 shell 场景单独要求 tmux。

缓存实验使用声明 `cache` 的 profile，较长等待预算集中在 `[cache]`：

```toml
[cache]
write_lag_seconds = 150
poll_interval_seconds = 30
case_seconds = 1200
```

```bash
uv run pytest --live --live-config tests/live/profiles.local.toml -m "live and cache" -s
```

明确报告零命中仍作为测量结果；声明缓存能力却完全不返回缓存指标时失败。手动 branching 审计也使用相同配置：

```bash
uv run python -m tests.cache_audit.run_branching --live \
  --live-config tests/live/profiles.local.toml --output records/cache-audit/run-001
uv run python -m tests.cache_audit.run_branching --analyze-only --output records/cache-audit/run-001
```

## 编写与清理测试

普通场景使用 `live_model` 或 `live_model_factory`。工厂按测试函数创建客户端，使用 `create()`、`openai()`、`claude()`，通过 `create(role="secondary")` 使用另一角色。测试结束、断言失败或取消后，先取消并等待当前函数作用域新增的后台任务，再统一关闭客户端；已有 fixture 任务不受影响，不共享带状态的客户端。

构造器和 registry 契约测试继续直接调用生产接口：通过工厂的 `connection()` 或 `registry_entry()` 取得连接参数，并用 `track(model)` 登记清理。不要在场景中读取环境变量、设置供应商默认值，或用可复用模型替代要验证的构造分支。

子进程使用 `offline_environment()` 去除测试凭据与 live 开关；确需运行真实 example 时，使用 `example_environment()` 将一个 profile 转换为 example 现有环境变量契约。应用本身的配置接口不受此次测试重构影响。

## CI Release 与覆盖

Release 使用受版本控制的 `tests/live/profiles.release.toml`。当前必测范围是 DeepSeek Responses 的 2 个基础构造/调用用例与 19 个 authoring 参数组合，清单在 `tests/live/required-responses-core.json`。

```bash
uv run pytest --live --live-config tests/live/profiles.release.toml \
  -m "live and (smoke or authoring) and not vision" \
  --live-required tests/live/required-responses-core.json \
  --live-report .artifacts/live-core.json \
  --junitxml .artifacts/live-core.xml
```

清单按完整 node ID 校验，任何必测项未选中、不适用、跳过、xfail、失败或未执行都会阻止发布。新增必测范围时，需要同时修改矩阵、完整 profile 和清单，不能通过更换模型能力声明缩减必测项。

Chat、Anthropic、图像、缓存、响应链、第二模型和外部 Langfuse 服务不属于当前发布必测范围，分别使用匹配配置执行，结果单独报告。当前 pytest live 清单没有独立的真实响应链用例；该能力有离线协议覆盖，不能仅添加 capability 就声称已有真实覆盖。Release 不再将核心验证称为“全量 live”。

终端输出配置来源及 profile/场景维度的结果统计；JSON 报告记录每个选中用例的状态和总 deselected 数，JUnit 记录具体失败。报告不保存密钥。迁移前后的逐用例位置和要求见 `tests/live/inventory.json`。

## 旧变量迁移

旧变量在一个迁移阶段内由中央兼容层支持；没有配置文件时才读取，不补入供应商默认地址。完整的 `LLM_API_FORMAT`、`LLM_BASE_URL`、`LLM_MODEL` 组可配合 `LLM_API_KEY` 使用，协议简称仅兼容层接受。协议专用旧变量作为该组的明确替代字段，有冲突时报错。

| 旧输入 | 新位置 |
| --- | --- |
| `LLM_*`、`ALIYUN_LIVE_*`、`ANTHROPIC_*`、`ANTHROPIC_LIVE_*` 的连接字段 | 具名 profile；凭据变量仍可由 `api_key_env` 引用 |
| `LLM_MODEL_ALTERNATIVE` | `roles.secondary` 指向完整 profile |
| `LLM_LIVE_TIMEOUT` | `timeouts.request_seconds` |
| 缓存等待与轮询变量 | `[cache]` |
| `IMAGE_URL` | `assets.image_url_env` |
| Langfuse host 与密钥变量 | `[services.langfuse]` |
| `ECS_P1_LIVE`、`ECS_P2_LIVE`、`ECS_AUTHORING_LIVE` | `--live -m "live and authoring"` |
| `RUN_LANGFUSE_LIVE_TESTS` | `--live` 加 Langfuse 文件或观测场景选择 |

旧开关不能启用 live。选择新配置文件后，旧连接变量不再覆盖它；终端提示被忽略的变量名。仓库场景已迁移，兼容层仍保留供外部调用迁移，删除时点另行记录。
