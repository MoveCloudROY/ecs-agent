# 测试配置统一与执行流程整理规划

日期：2026-10-21\
状态：已按用户确认完成重构，离线回归与指定 DeepSeek Responses 场景验证通过\
现状基线：`e555aa5`\
关联记录：[Agent 编写接口修复与 CI 验证](2026-10-20-agent-authoring-remaining-fix-results.md)

本规划统一测试中的 LLM 连接配置、凭据来源、运行开关、超时和资源清理，使本地测试、CI 与 Release 使用相同的配置和执行规则。建议采用具名模型配置 profile、默认离线、显式启用 live、配置错误提前失败，并按场景和能力组织真实服务测试。

用户于 2026-10-22 确认按本规划实施。测试、CI/Release 和说明已迁移；最终接口与验证结果见[实施与验证记录](2026-10-23-test-configuration-unification-results.md)，运行方式见[测试指南](../testing.md)。补充有效凭据后，21 项核心必测及 62 项扩展场景均已通过；其他协议、能力与外部服务的验证边界见实施记录。下文保留原设计要求。

## 当前问题

对 `tests/live/*.py`、`tests/test_real_llm*.py` 和 `tests/conftest.py` 的静态扫描发现，25 个文件直接读取了 25 个不同的环境变量名。该统计仅包含字面量变量名，不包含通过函数参数动态读取的变量。

| 问题 | 现有位置与行为 | 影响 |
| --- | --- | --- |
| 连接配置存在多套命名 | `LLM_*`、`ALIYUN_LIVE_*`、`ANTHROPIC_*`、`ANTHROPIC_LIVE_*` 并存 | 难以确认一个测试实际使用哪个服务和模型 |
| 默认值与优先级不一致 | 原共享解析器 `tests/live/api_format.py`（已迁入测试支持层） 含 DashScope 默认地址，部分文件自行读取配置；两个 Anthropic live 文件使用不同默认地址和模型 | 同一组环境变量在不同测试中可能产生不同连接结果 |
| 启用规则分散 | [根 fixture](../../tests/conftest.py) 按密钥是否存在决定跳过；authoring 使用三个 `ECS_*_LIVE` 开关，Langfuse 使用独立开关 | 设置密钥可能意外启用测试，执行 live 目录也不代表覆盖全部场景 |
| 配置错误与服务失败被跳过 | 未知协议、缺少配置和部分请求超时进入 `skip` | 通过结果不能充分说明配置有效或功能已验证 |
| live 用例分布不统一 | 三个 `test_real_llm*.py` 位于 `tests/` 根目录 | 按目录执行容易遗漏真实服务测试 |
| 模型创建与清理重复 | 多处直接组装 `Model`、`ProviderConfig`、registry，超时与关闭方式不一致 | 修改成本高，异常路径容易遗留连接或异步任务 |
| Release 覆盖范围不明确 | [Release workflow](../../.github/workflows/release.yml) 运行 `pytest tests/live -v`，未启用 authoring 和 Langfuse 开关 | “full live test suite” 与实际执行范围不一致，缺少必测覆盖校验 |

需要保留的基础包括：现有 [ApiFormat 与 ProviderConfig](../../src/ecs_agent/providers/config.py)、`Model`、`ProviderRegistry`，以及 [CI](../../.github/workflows/ci.yml) 已有的 base/MCP 依赖矩阵和锁定依赖安装。

## 范围与设计原则

本次范围包括测试支持代码、全部真实 LLM 测试、pytest 配置、CI/Release 和使用说明。Langfuse 等外部服务纳入统一执行策略，但保留自己的服务配置结构。

- 测试配置集中解析，场景文件只表达所需能力、输入和断言。
- 默认 `pytest` 不连接真实 LLM 或外部观测服务，即使宿主环境已有密钥。
- 显式选择 live 后，对实际选中场景所需的配置和依赖进行预检；配置错误必须在请求前暴露。
- 端点、模型、协议和凭据引用属于同一个 profile，不能从多个供应商配置中隐式拼接。
- 复用生产 provider 接口；测试辅助层不成为新的公共 SDK。
- 保留场景断言、参数组合与被测入口，不以减少覆盖或放宽断言完成迁移。

本次不统一应用和 examples 的全部配置，不修改生产 provider 的重试语义，不处理已有四个 conversation-tree 预期失败。单元测试中的虚拟 URL、模型名、非法值和环境变量输入属于测试数据，继续留在对应测试中。

## 四层职责

| 层次 | 职责 | 拟议位置 |
| --- | --- | --- |
| 配置层 | 解析 profile、服务配置、超时、旧变量映射，验证并记录配置来源 | `tests/support/config.py` |
| 执行层 | 注册 CLI/marker，选择 live 场景，预检所需能力与依赖，输出覆盖摘要 | `tests/support/pytest_plugin.py` |
| 资源层 | 将配置适配到现有 provider，提供模型工厂，统一关闭资源和子进程环境 | `tests/support/models.py`、`tests/support/environment.py` |
| 场景层 | 声明协议与能力要求，构造输入，调用被测 API，验证行为 | `tests/live/` 及普通测试文件 |

根 `tests/conftest.py` 负责加载插件和保留已有全局隔离 fixture。集中解析避免在测试模块导入时读取真实服务环境；配置解析器接收显式环境映射，便于离线验证不同输入。

测试支持类型采用有类型标注的 dataclass，并遵循项目代码规范。配置数据与 pytest 决策分离，使配置校验可以通过普通单元测试完成。

## 配置契约

### 具名模型配置

每个 profile 表示一组确定的连接参数。以下示例沿用此前测试使用的模型名称，用来说明配置结构；能力字段用于声明预期支持范围，仍需相应测试验证。

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

| 字段 | 约束 |
| --- | --- |
| `base_url` | 显式提供，符合对应 provider 的基础地址语义；不补入供应商默认地址 |
| `model` | 非空模型标识；不按测试文件指定不同隐含默认值 |
| `api_format` | 使用既有 `ApiFormat` 中支持的 LLM 协议值；新配置使用规范名称 |
| `api_key_env` | 环境变量名称，运行时解析；配置文件不保存密钥值 |
| `capabilities` | 使用集中定义的能力词表，例如文本、工具、流式、图像、缓存、响应链 |
| 超时 | 必须为有限正数；统一默认值允许在明确的 profile 或场景配置中覆盖 |

Chat、Responses 和 Anthropic Messages 分别建立 profile。即使模型名或地址相同，也不因某个测试需要另一协议而临时替换 `api_format`。

`primary` 表示当前主要模型；需要第二模型或图像模型时，通过 `secondary`、`vision` 角色显式引用其他 profile。仅实际选中的测试要求对应角色，不为普通文本测试强制要求图像或第二模型配置。协议专用场景必须校验角色绑定的协议。

### 配置来源与优先级

拟提供受版本控制的 `tests/live/profiles.example.toml`，以及加入忽略规则的本地 `tests/live/profiles.local.toml`。CI/Release 使用显式指定的非敏感配置文件，密钥从 Actions secrets 注入。

解析顺序如下：

1. `--live-config PATH` 指定文件；未指定时查找约定的本地文件。示例文件不自动作为有效运行配置。
2. `--llm-profile NAME` 选择主要 profile；未指定时使用文件中的 `roles.primary`。
3. 读取被选中 profile 的完整连接参数，并从其 `api_key_env` 引用解析密钥。
4. 没有新配置时，迁移期允许中央兼容层把完整旧变量组转换成 profile；输出弃用提示与来源。

不自动搜索或加载 `.env`。显式指定的文件不存在、profile 不存在或字段不合法时直接失败，不回退到旧变量或默认供应商。使用新配置时，不再把旧端点、模型、协议变量当作逐字段覆盖；被忽略的旧配置仅提示变量名，不能改变连接结果。密钥环境变量仍可按 `api_key_env` 正常引用。

兼容路径中，多个旧变量组对同一角色给出冲突值时明确报错，不能静默采用历史文件各自的优先级。未知协议必须报错；旧的协议简称仅在兼容层转换。

### 超时和场景参数

请求超时、流式读取超时和整个测试场景的截止时间分别建模。默认值集中维护，上例中的 60/60/180 秒作为初始建议，实施时按既有场景预算确认。

缓存写入等待、缓存轮询间隔、工具执行预算等属于场景参数，不混入模型连接字段。缓存实验放入独立场景配置，显式声明较长总预算；工厂负责将可支持的超时参数映射到对应 provider，不能接受参数后静默丢弃。

### 外部服务与测试素材

- Langfuse 单独保存 host 和凭据环境变量引用，不把其密钥合并进 LLM profile。
- 图像输入通过统一素材配置或 fixture 提供，替代场景内直接读取 `IMAGE_URL`；仅图像场景要求该配置。
- MCP、Langfuse SDK、tmux 等作为依赖要求声明，与模型 profile 分开。
- 子进程通过环境适配器获得必要配置，默认不向离线子测试传递 live 开关和真实密钥；不靠修改全局 `os.environ` 临时切换模型。

### 脱敏与配置来源

启动摘要显示 profile、模型、协议、经过脱敏的端点、超时和配置来源。凭据对象的 repr 排除密钥，不直接输出包含密钥的 `ProviderConfig`、完整环境映射或认证头。URL 的用户信息、查询参数等敏感内容也需要隐藏。

诊断信息应能回答“哪个字段来自哪里、为什么未执行某场景”，而不要求开发者打印密钥确认配置。兼容层提示仅列变量名和冲突字段。

## 执行与结果语义

### 默认离线与显式 live

拟议命令如下；迁移完成后才能使用这些新增参数和 marker。

```bash
# 默认运行离线测试
uv run pytest

# 运行选定模型的基础真实服务验证
uv run pytest --live --llm-profile deepseek-responses -m "live and smoke"

# 运行 agent 编写接口真实服务验证
uv run pytest --live --llm-profile deepseek-responses -m "live and authoring"

# 显式选择配置文件和图像场景
uv run pytest --live --live-config /path/to/profiles.toml -m "live and vision"
```

`--live` 表示允许执行 live，`-m` 表示选择场景；没有 `--live` 时，真实服务用例统一 deselect，并说明数量。即使显式传入某个 live 文件路径，也不能绕过开关。`--live` 不自动替代 `-m`：不提供选择表达式时，允许离线和 live 一起运行。

真实服务测试统一标记 `live`，场景使用 `smoke`、`authoring`、`vision`、`cache`、`observability` 等 marker；具体目录名不再决定是否需要真实凭据。协议与能力要求单独声明，不能用供应商品牌名推断能力。

pytest 先完成路径、`-k`、`-m` 等用户选择，再预检选中 live 用例的配置和依赖，之后创建资源。未选中场景不要求其服务凭据。配置检查不发出网络探测请求。

### 结果分类

| 情况 | 预期结果 |
| --- | --- |
| 未启用 live | live 用例 deselect，不创建真实客户端、不发送请求 |
| 选中场景缺少密钥、角色、素材或必要依赖 | 预检失败，包含补齐方法 |
| 配置文件不存在、未知协议、非法超时、冲突旧配置 | 配置错误，首个请求前失败 |
| profile 明确不支持所选集合中的某项能力或协议 | 标记为不适用并说明原因，单独汇总 |
| 声明支持能力后收到不支持响应 | 测试失败，不临时修改能力或转为不适用 |
| 已发出的请求发生认证错误、超时、限流、连接错误或 5xx | 测试失败，保留脱敏后的错误分类 |
| 断言不满足 | 测试失败，保留原有行为验证 |
| 显式要求执行 live，但没有任何适用 live 用例执行 | 非零退出，不视为有效验证 |

不适用可以映射为带统一原因的 pytest skip，但报告需要与未启用、配置失败、服务失败区分。Release 必测场景不允许通过“不适用”满足门槛。

移除通过错误文本识别超时、限流、5xx 后转为 skip 的做法。保留生产 provider 已有重试语义，测试框架不增加整个 agent 场景的自动重试，避免重复执行带副作用的工具。

## Fixture 与资源生命周期

拟提供以下 fixture：

| Fixture | 责任 |
| --- | --- |
| `live_settings` | 会话级解析结果、超时和服务配置，保持只读 |
| `live_profile` | 当前场景选中的模型角色与能力 |
| `live_model_factory` | 创建现有 provider 实例，登记并在 teardown 关闭 |
| `live_model` | 普通单模型场景的便捷入口 |
| `live_image` | 已校验的图像素材 |

工厂复用 `Model`、`ProviderConfig` 与 registry 的既有能力。配置可以跨测试共享，带连接和可变状态的客户端默认按测试函数创建，避免 response chaining、缓存状态或异步事件循环互相污染。

关闭责任由创建入口明确承担，异常、取消和断言失败路径同样执行。需要直接构造对象的测试，通过资源登记机制交由 fixture 清理，避免重复关闭和清理遗漏。

`test_model_constructor_live.py` 和 registry 相关测试继续直接调用各自被测 API，只共享配置和资源清理。不能统一经过一个构造工厂后丢失不同构造分支的覆盖。

## 迁移清单

| 现有内容 | 迁移方式 |
| --- | --- |
| `LLM_API_KEY` | 可继续作为 profile 的凭据环境变量名；去除“存在即启用”的语义 |
| `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_FORMAT` | 迁移到 profile 字段，旧组由中央兼容层读取 |
| `ALIYUN_LIVE_*`、`LLM_RESPONSES_BASE_URL` | 转换成明确的 Chat/Responses profile，取消隐含默认值 |
| `ANTHROPIC_*`、`ANTHROPIC_LIVE_*` | 统一成显式 Anthropic profile 和凭据引用，检查重复组冲突 |
| `LLM_MODEL_ALTERNATIVE` | 第二模型角色引用；不得只替换模型名却隐含沿用不兼容连接 |
| `ECS_P1_LIVE`、`ECS_P2_LIVE`、`ECS_AUTHORING_LIVE` | 迁移到 `--live` 与 authoring 场景选择 |
| `RUN_LANGFUSE_LIVE_TESTS` | 迁移到 `--live` 与 observability 场景选择，服务配置独立验证 |
| `IMAGE_URL` | 统一素材配置；兼容层保留旧输入映射 |
| `LLM_LIVE_TIMEOUT` | 中央超时配置 |
| 缓存等待和轮询变量 | 集中到缓存场景配置 |
| `tests/live/api_format.py` | 解析能力并入支持层，迁移后移除重复解析与失败转 skip 逻辑 |
| 三个根目录 `test_real_llm*.py` | 逐用例分类；真实服务部分迁入 live，混合文件中的离线测试保留 |

旧变量兼容保留一个迁移阶段：所有读取进入中央兼容层，给出迁移提示和冲突检查。旧启用开关不能绕过新的 `--live`；必要时仅提示等价的场景选择命令。

兼容层退出条件是：仓库测试、CI/Release 和使用说明均已使用新入口，旧输入有完整迁移说明及回归验证。删除前确认实际外部使用情况，并在变更记录中明确删除时点，不按一个未经约定的版本号自动移除。

## CI 与 Release

普通 CI 保留 base/MCP 依赖矩阵，继续使用 `uv sync --locked` 和 `uv run --no-sync pytest`。两组都执行离线测试；MCP 组验证实际安装 SDK 的路径，base 组验证可选依赖缺失时的行为。安装 MCP 不意味着开启真实 LLM。

Release 使用同一配置解析器、marker 和命令参数，依赖安装与普通 CI 对齐。将“全量 live”改为显式的验证矩阵，区分协议、能力和外部服务：

| 验证维度 | 配置方式 | 完成条件 |
| --- | --- | --- |
| 普通离线回归 | base/MCP 依赖组 | 必须通过 |
| 基础 LLM 与 authoring | 指定 profile 与场景组 | 必测用例实际执行并通过 |
| 协议专用验证 | Chat、Responses、Anthropic 对应 profile | 每个列为必测的协议均有有效执行结果 |
| 图像、缓存、响应链 | 声明相应能力的 profile 与素材 | 按明确列入的扩展矩阵验收 |
| Langfuse 等观测集成 | LLM profile 加服务配置与可选 SDK | 独立报告，不因缺少环境静默消失 |

阶段 A 需列出实际可用 profile 和必测场景清单，确认哪些扩展进入发布门槛；未纳入的部分明确标为未覆盖。必测场景不能由“所选 profile 恰好支持什么”反向决定。

Release 对每个必测场景核对预期用例或参数组合，任何必测缺失、不适用、skip、xfail 或失败都不能满足门槛。仅“有一个用例通过”不足以证明该场景完成；同时防止零选择和全部跳过的任务返回成功。

每次运行输出 profile/场景维度的通过、失败、不适用和未选择数量，CI 保存脱敏摘要和测试报告。首次真实服务验证由小范围 smoke 开始，再运行协议与功能矩阵；正式门槛必须覆盖清单全部必测项。

## 实施阶段与交付物

### 阶段 A 盘点与冻结契约

- 建立逐用例清单：是否联网、协议、能力、外部依赖、特殊预算、原有开关和断言目标。
- 明确 profile 字段、角色、超时默认值、旧变量冲突规则以及 marker 词表。
- 确认 Release 必测范围和实际模型/服务配置来源。
- 交付配置样例、变量映射和迁移前覆盖基线。

### 阶段 B 共享基础设施

- 新增 `tests/support/` 配置、pytest 插件、工厂和环境适配器。
- 接入默认离线、显式 live、配置预检、脱敏及结果摘要。
- 增加针对配置决策、pytest 行为和资源清理的离线测试。
- 先迁移代表性的普通模型、直接构造和 authoring 场景，验证抽象边界。

### 阶段 C 全部场景迁移

- 迁移 `tests/live/` 和根目录真实 LLM 用例，保留对应断言和参数组合。
- 清除场景内的连接环境读取、供应商默认值和错误转 skip 逻辑。
- 统一模型、流式连接、服务客户端和子进程清理。
- 对照阶段 A 清单确认每个原有行为都有对应测试，记录合理拆分或移动。

### 阶段 D CI Release 与说明收尾

- 更新普通 CI 和 Release，配置必测覆盖检查与报告产物。
- 更新 README、模型配置、metrics、Langfuse、prompt caching、logging 等测试说明，形成一处权威运行指南，其余位置引用该指南。
- 历史 RFC 的执行记录保留原有命令与结果，通过迁移说明链接提供新入口。
- 完成离线全量回归与已配置 live 矩阵验证，记录结果与未覆盖项，再按退出条件处理旧变量兼容层。

## 验收标准

1. 宿主环境设置真实密钥时，普通 `pytest` 仍不执行真实 LLM 或外部观测请求；收集阶段也不创建真实连接。
2. 同一 profile 在不同场景中的端点、协议、模型和凭据来源一致；场景不再自行读取这些环境变量。
3. 缺少必需配置、非法协议/超时、旧变量冲突在首个请求前失败，未选中的外部服务不要求凭据。
4. 服务错误导致失败，不再转为跳过；能力不适用有可追踪原因，必测覆盖不能通过能力声明被削弱。
5. 正常、断言失败、取消和部分初始化失败时，已创建资源均被清理；测试间客户端状态不泄漏。
6. 所有现有真实服务场景都进入清单并完成迁移，构造器和 registry 测试仍覆盖原来的直接调用分支。
7. base/MCP 离线矩阵通过，既有预期失败与环境差异单独说明，不以新增 skip/xfail 掩盖迁移回归。
8. live smoke、authoring 及约定协议/扩展矩阵有明确结果；必测项缺失、零选择或全部跳过时，Release 门槛失败。
9. 日志、终端摘要和 CI 报告可解释配置来源且不暴露凭据；文档命令与实际 CLI 一致。

基础设施验证使用虚拟凭据、受控环境映射和本地模拟请求，重点覆盖选择、校验、脱敏和清理的可观察行为。真实接口验证用于确认最终连接映射与功能行为，不替代这些确定性检查。

## 确认与实施状态

建议整体采用以下边界：测试代码、CI/Release 和相关文档共同迁移；默认离线，显式启用 live；配置错误提前失败，实际服务错误作为失败报告；旧变量集中兼容一个迁移阶段。

用户已确认实施。当前 Release 选择用户指定的 DeepSeek Responses 配置，必测清单包含 2 个基础用例和 19 个 authoring 参数组合。其他协议、图像、缓存、第二模型和外部 Langfuse 服务未纳入本次发布必测范围，详见测试指南与实施记录。
