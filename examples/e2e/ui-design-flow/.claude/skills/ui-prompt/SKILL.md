---
name: ui-prompt
description: '根据用户提供的界面需求草稿，结构化地生成适用于 Nano Banana (以及 Nano Banana 2) 的英文图像生成提示词，帮助用户快速生成高质量的UI设计视觉稿。在用户需要从UI设计草稿生成视觉稿时使用。
user-invokable: true
metadata: {"version": "0.0.1", "updated": "2026-03-09", "author":"Mancitrus"}
---

# ui-prompt

接收用户提供的 UI 设计草稿，深度分析其布局、功能和设计意图，拆解为清晰的页面层级，并针对每一个独立页面，深度刻画其中的交互组件样式，最终输出一套可直接用于生成高保真原型的英文 Nano Banana 提示词集。

## 核心规则 (Golden Rules)

1. **Page-by-page generate**: After receiving the design draft, do not generate a mixed prompt all at once. You must conduct a  analysis, identifying the interactive components contained in each page and their specific visual styles, and finally output a precise English prompt independently for each page.

2. **Natural language over tag soup**: Write as if briefing a human artist, not listing keywords.
   - BAD: "dog, park, sunset, 4k, realistic, cinematic"
   - GOOD: "A golden retriever bounding through a sun-dappled park at golden hour, shot from a low angle with shallow depth of field"

3. **Specificity matters**: Define subjects precisely with materiality, texture, and detail.
   - Instead of "a woman": "a sophisticated elderly woman wearing a vintage Chanel-style tweed suit"
   - Include materials: "matte finish," "brushed steel," "soft velvet," "weathered leather"

4. **Provide context about purpose**: Mention the use case or audience.
   - "Create a hero image for a premium coffee brand's website" helps the model infer professional lighting, composition, and mood.

5. **Must use English**

## 工作流程 (Workflow)

### Stage 1：全局解析与页面清单 (Global Mapping)

- 分析用户的草稿，提取出应用类型、整体风格和主色调。
- 将草稿拆解为独立的页面清单（如：首页/Dashboard、商品详情页、个人中心、登录注册页等）。

### Stage 2：单页聚焦与组件样式定义 (Page Focus & Component Styling)

- 针对当前正在处理的单一页面，梳理其核心布局（Top/Middle/Bottom）。
- 重点提取交互组件，并强制为其赋予视觉表现词汇：
   - 按钮 (Buttons)：例如 glowing CTA button, neumorphic toggle, pill-shaped button with subtle drop shadow.
   - 导航 (Navigation)：例如 glassmorphism bottom tab bar, floating action button (FAB), sticky frosted glass header.
   - 卡片/列表 (Cards/Lists)：例如 rounded corner cards with soft gradient, swipeable carousel, grid layout with hover state indications.
   - 输入/表单 (Inputs)：例如 minimalist search bar with outlined icon, sleek input fields.

### Stage 3：生成 Nano Banana 提示词 (Prompt Generation)

- 将第二步的分析转化为纯英文的 Nano Banana 专用提示词。

- 提示词公式：`[UI/UX Base], [Aspect Ratio], [Page Type], [Core Layout], [Detailed Interactive Components Styling], [Color Palette], [Lighting/Texture], [Quality Modifiers]`。

   1. `[UI/UX Base]`: eg. UI design, user interface, mobile app design, web interface
   2. `[Aspect Ratio]`: eg. 16:9, 9:16, 1:1, 4:3, vertical orientation, horizontal orientation
   3. `[Page Type]`: eg. dashboard, login screen, e-commerce product page, user profile
   4. `[Core Layout]`: eg. split screen, masonry grid, centered layout, top heavy layout
   5. `[Detailed Components]`: eg. glowing CTA button, glassmorphism cards, floating action button (FAB)
   6. `[Color Palette]`: eg. dark mode, pastel colors, monochrome, neon blue accents
   7. `[Lighting/Texture]`: eg. soft drop shadow, frosted glass texture, clean gradient, matte finish
   8. `[Quality Modifiers]`: eg. highly detailed, high fidelity, 8k resolution, Dribbble style

### Stage 4：流转至下一页 (Iterate)

交付当前页面的提示词后，主动询问用户是否需要调整，或者直接进入下一个页面的提示词生成。

## 交付格式 (Output Format)
输出到 @../../../ui-design/nano-banana-prompts.md，格式如下：
```
### 页面 [N]：[页面名称，例如：首页 Dashboard]

**1. 视觉与组件架构**
* **页面布局**：[简述布局，如：顶部固定搜索，中部横向滑动卡片，底部悬浮导航]
* **关键交互组件样式**：
    * [组件 A]：[样式描述，如：带有微妙磨砂玻璃质感的底部 Tab 栏 (Glassmorphism bottom tab bar)]
    * [组件 B]：[样式描述，如：高对比度、带有柔和发光效果的“立即购买”胶囊按钮 (Glowing pill-shaped CTA button)]
    * [组件 C]：[样式描述]

**2. Nano Banana Prompt (可直接复制)**
**[此处输出纯英文提示词。必须包含具体的组件英文描述、整体风格、以及 UI design, high fidelity 等质量词。使用加粗显示]**

*Negative Prompt*: text, blurry, distorted, messy layout, overlapping components, bad proportions, cluttered interface, low resolution, complex background, realistic background, desk background, environmental shadows

```