---
name: ui-ux-reviewer
description: "UI/UX review checklist for landing pages and dashboards. Focus on hierarchy, accessibility, responsive layout, and interaction quality."
user-invocable: true
argument-hint: "<design-description>"
disable-model-invocation: false
---

# UI/UX Reviewer

Use this skill when reviewing or refining UI output. You are currently reviewing: $ARGUMENTS

## Review Priorities

1. Visual hierarchy is clear and consistent.
2. Accessibility is respected (contrast, focus states, semantics).
3. Layout works on mobile and desktop.
4. Interactions are predictable and provide feedback.

## Checklist

- Headline and CTA are visible without scrolling.
- Body text uses readable size and line-height.
- Interactive elements have visible hover/focus states.
- Buttons and tap targets are large enough for touch use.
- Spacing rhythm is consistent across sections.
- Motion is subtle and avoids distracting effects.

## Tool Usage

Use `build_ui_checklist` to generate a compact checklist for a specific page type.
