"""Textual application for the plan-and-task TUI.

Renders the ``PlanTaskViewModel`` and forwards submitted input to an
``InputSink`` (the event-bus bridge in production, a stub in tests). All ECS
updates arrive through :meth:`PlanTaskTuiApp.dispatch_change`, which posts a
``VmChanged`` message so widget mutation happens on Textual's message pump.
"""

from __future__ import annotations

from typing import Protocol

from rich.cells import cell_len
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.selection import Selection
from textual.strip import Strip
from textual.suggester import SuggestFromList
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

from examples.e2e.plan_and_task.tui.view_model import (
    PlanTaskViewModel,
    TranscriptEntry,
    UiChange,
)


class InputSink(Protocol):
    """Receiver for user actions raised by the app."""

    def submit_input(self, text: str) -> bool: ...

    def request_quit(self) -> None: ...


class CommandInput(Input):
    """Single-line prompt with slash-command suggestions."""


class SelectableRichLog(RichLog):
    """RichLog with screen text-selection support.

    ``RichLog`` stores pre-rendered ``Strip``s and implements neither text
    extraction nor the segment offset metadata the compositor uses to map
    the pointer to a text position, so screen selections over it degrade to
    all-or-nothing. This subclass follows the built-in ``Log`` widget:
    ``Strip.apply_offsets`` exposes per-character offsets for hit-testing,
    extraction joins the strip buffer's text, and the selection highlight
    is overlaid per line at render time. Selection offsets are string
    indices; cell widths are only computed when slicing the strip, so
    double-width (CJK) characters extract correctly.
    """

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Extract the selected text from the strip buffer."""
        text = "\n".join(strip.text.rstrip() for strip in self.lines)
        return selection.extract(text), "\n"

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        scroll_x, scroll_y = self.scroll_offset
        content_y = y + scroll_y
        if not self.lines:
            return strip
        if content_y >= len(self.lines):
            # Blank area below the content: anchor offsets to the last line so
            # a drag ending there selects to the end of the content instead of
            # falling back to the compositor's select-everything default.
            return strip.apply_offsets(scroll_x, len(self.lines) - 1)
        selection = self.text_selection
        if selection is not None:
            strip = self._apply_selection_style(strip, selection, content_y, scroll_x)
        return strip.apply_offsets(scroll_x, content_y)

    def _apply_selection_style(
        self, strip: Strip, selection: Selection, content_y: int, scroll_x: int
    ) -> Strip:
        span = selection.get_span(content_y)
        if span is None:
            return strip
        start, end = span
        text = self.lines[content_y].text
        if end == -1:
            end = len(text)
        # Span offsets are string indices; the strip is sliced by cell column.
        cell_start = max(0, cell_len(text[:start]) - scroll_x)
        cell_end = min(strip.cell_length, max(0, cell_len(text[:end]) - scroll_x))
        if cell_end <= cell_start:
            return strip
        selection_style = self.screen.get_component_rich_style("screen--selection")
        # Overlay only the selection background: Strip.apply_style layers
        # UNDER existing segment styles (the widget background would win),
        # and the theme's full selection style sets color == bgcolor, which
        # would render the selected text invisible.
        overlay = (
            RichStyle(bgcolor=selection_style.bgcolor)
            if selection_style.bgcolor is not None
            else selection_style
        )
        before, selected, after = strip.divide(
            [cell_start, cell_end, strip.cell_length]
        )
        highlighted = Strip(
            [
                Segment(seg_text, seg_style + overlay if seg_style else overlay)
                for seg_text, seg_style, _ in selected
            ],
            selected.cell_length,
        )
        return Strip.join([before, highlighted, after])


class VmChanged(Message):
    """A view-model section changed and should be re-rendered."""

    def __init__(self, change: UiChange) -> None:
        self.change = change
        super().__init__()


def _fmt_tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


_RUN_ICONS = {"running": "▶", "completed": "✓", "failed": "✗"}
_RUN_STYLES = {"running": "yellow", "completed": "green", "failed": "red"}
_WAITING_PLACEHOLDER = "agent is working… (Ctrl+Q to quit)"
_READY_PLACEHOLDER = "type a message or /command — Enter to send"

SPINNER_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
_STATUS_INTERVAL = 1 / 8

_ACTIVITY_LABELS = {
    "waiting": "waiting for model",
    "thinking": "thinking",
    "generating": "generating",
    "tool": "tool",
    "subagent": "subagent",
}


class PlanTaskTuiApp(App[None]):
    """Dashboard-style TUI for the plan-and-task workflow."""

    TITLE = "plan-and-task"

    CSS = """
    #body {
        height: 1fr;
    }
    #chat-pane {
        width: 1fr;
    }
    #transcript {
        height: 1fr;
        border: round $primary 30%;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }
    #live-tail {
        height: auto;
        max-height: 14;
        border: round $success 30%;
        padding: 0 1;
        display: none;
    }
    #sidebar {
        width: 38;
        border: round $primary 30%;
        padding: 0 1;
    }
    #phase-panel, #subagent-panel, #review-panel {
        height: auto;
    }
    #task-table {
        height: 1fr;
        min-height: 3;
    }
    #status-bar {
        height: 1;
        padding: 0 1;
        display: none;
    }
    #status-label {
        width: 1fr;
        color: $success;
    }
    #status-tokens {
        width: auto;
        color: $warning;
    }
    #usage-bar {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    CommandInput {
        border: round $accent;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit_session", "Quit"),
        ("ctrl+l", "clear_transcript", "Clear log"),
    ]

    def __init__(
        self,
        view_model: PlanTaskViewModel,
        sink: InputSink,
        commands: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._vm = view_model
        self._sink = sink
        self._commands = commands
        self._backlog: list[UiChange] = []
        self._last_copied: str | None = None
        self._spinner_frame = 0

    # -- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="chat-pane"):
                yield SelectableRichLog(
                    id="transcript", wrap=True, auto_scroll=True, min_width=0
                )
                yield Static(id="live-tail")
            with Vertical(id="sidebar"):
                yield Static(id="phase-panel")
                yield Static(id="review-panel")
                yield DataTable(id="task-table", cursor_type="row")
                yield Static(id="subagent-panel")
        with Horizontal(id="status-bar"):
            yield Static(id="status-label")
            yield Static(id="status-tokens")
        yield Static(id="usage-bar")
        suggester = (
            SuggestFromList(self._commands, case_sensitive=False)
            if self._commands
            else None
        )
        yield CommandInput(
            placeholder=_WAITING_PLACEHOLDER,
            suggester=suggester,
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.add_columns("task", "status", "retry")
        self._render_phases()
        self._render_reviews()
        self._render_subagents()
        self._render_usage()
        self.set_interval(_STATUS_INTERVAL, self._tick_status)
        self.query_one(CommandInput).focus()
        backlog, self._backlog = self._backlog, []
        for change in backlog:
            self.post_message(VmChanged(change))

    # -- bridge entry point -------------------------------------------------

    def dispatch_change(self, change: UiChange) -> None:
        """Entry point for the event-bus bridge (same asyncio loop)."""
        if self.is_running:
            self.post_message(VmChanged(change))
        else:
            self._backlog.append(change)

    # -- message/action handlers --------------------------------------------

    @on(VmChanged)
    def _handle_vm_changed(self, message: VmChanged) -> None:
        change = message.change
        match change.section:
            case "transcript":
                log = self.query_one("#transcript", SelectableRichLog)
                for entry in change.entries:
                    self._write_entry(log, entry)
            case "live":
                self._render_live_tail()
            case "phases":
                self._render_phases()
            case "tasks":
                self._render_tasks()
                self._render_reviews()
            case "usage":
                self._render_usage()
            case "subagents":
                self._render_subagents()
            case "input":
                self._render_input_state()
            case "status":
                self._render_status()
            case "notify":
                if change.notification is not None:
                    self.notify(change.notification, severity=change.severity)

    @on(Input.Submitted)
    def _handle_submit(self, event: Input.Submitted) -> None:
        if self._sink.submit_input(event.value):
            field = self.query_one(CommandInput)
            field.value = ""

    def on_text_selected(self, event: events.TextSelected) -> None:
        """Copy-on-select: mirror the selection to the clipboard (OSC 52).

        The screen posts ``TextSelected`` on every selection-ending mouse
        release; a plain click clears the selection first, so ``selected``
        is empty then and only real selections are copied. Deduplication
        keeps stale-selection mouse-ups (e.g. scrollbar clicks) from
        re-copying and re-notifying.
        """
        selected = self.screen.get_selected_text()
        if not selected:
            self._last_copied = None
            return
        if selected == self._last_copied:
            return
        self._last_copied = selected
        self.copy_to_clipboard(selected)
        self.notify(
            f"copied {len(selected)} characters",
            title="clipboard",
            timeout=1.5,
        )

    def action_quit_session(self) -> None:
        self._sink.request_quit()
        self.exit()

    def action_clear_transcript(self) -> None:
        self.query_one("#transcript", SelectableRichLog).clear()

    # -- renderers ----------------------------------------------------------

    def _write_entry(self, log: RichLog, entry: TranscriptEntry) -> None:
        match entry.kind:
            case "user":
                log.write(Text(""))
                log.write(Text(f"❯ {entry.text}", style="bold cyan"))
            case "assistant":
                log.write(Text(""))
                log.write(Markdown(entry.text))
            case "reasoning":
                log.write(Text(entry.text, style="dim italic"))
            case "command":
                title = entry.meta.get("command", "command")
                log.write(Panel(Text(entry.text), title=title, border_style="cyan"))
            case "tool_call":
                log.write(Text(f"⚙ {entry.text}", style="yellow"))
            case "tool_result":
                style = "green" if entry.meta.get("success") == "true" else "red"
                log.write(Text(f"  └ {entry.text}", style=style))
            case "subagent":
                log.write(Text(entry.text, style="magenta"))
            case "system":
                log.write(Text(f"· {entry.text}", style="dim"))
            case "error":
                log.write(Text(f"✗ {entry.text}", style="bold red"))

    def _render_live_tail(self) -> None:
        tail = self.query_one("#live-tail", Static)
        parts: list[Text] = []
        if self._vm.live_reasoning:
            parts.append(Text(self._vm.live_reasoning[-1500:], style="dim italic"))
        if self._vm.live_content:
            parts.append(Text(self._vm.live_content[-3000:]))
        if parts:
            combined = Text("\n").join(parts) if len(parts) > 1 else parts[0]
            tail.update(combined)
            tail.display = True
        else:
            tail.update("")
            tail.display = False

    def _render_phases(self) -> None:
        lines = Text()
        lines.append("Phases\n", style="bold")
        for phase_id in self._vm.phase_ids:
            if phase_id == self._vm.current_phase:
                lines.append(f"▸ {phase_id}\n", style="bold green")
            else:
                lines.append(f"· {phase_id}\n", style="dim")
        self.query_one("#phase-panel", Static).update(lines)

    def _render_reviews(self) -> None:
        panel = self.query_one("#review-panel", Static)
        if not self._vm.review_verdicts:
            panel.update("")
            return
        lines = Text()
        lines.append("Reviews\n", style="bold")
        for phase, verdict in self._vm.review_verdicts:
            style = "green" if verdict == "approved" else "yellow"
            lines.append(f"{phase}: {verdict}\n", style=style)
        panel.update(lines)

    def _render_tasks(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.clear()
        for task in self._vm.tasks:
            marker = "▸ " if task.task_id == self._vm.current_task_id else ""
            table.add_row(
                f"{marker}{task.task_id}", task.status, str(task.retry_count)
            )
        if self._vm.workflow_id is not None:
            self.sub_title = (
                f"{self._vm.workflow_id} · {self._vm.current_phase}"
                f" · {self._vm.workflow_status}"
            )

    def _render_subagents(self) -> None:
        panel = self.query_one("#subagent-panel", Static)
        if not self._vm.subagent_runs:
            panel.update("")
            return
        lines = Text()
        lines.append("Subagents\n", style="bold")
        for run in self._vm.subagent_runs[-6:]:
            icon = _RUN_ICONS.get(run.status, "?")
            style = _RUN_STYLES.get(run.status, "")
            suffix = (
                f" {_fmt_tokens(run.stream_chars)} chars"
                if run.status == "running" and run.stream_chars
                else ""
            )
            lines.append(f"{icon} {run.name}{suffix}\n", style=style)
        panel.update(lines)

    def _render_usage(self) -> None:
        usage = self._vm.usage
        parts = [
            f"{usage.invocations} calls",
            f"in {_fmt_tokens(usage.prompt_tokens)}",
            f"out {_fmt_tokens(usage.completion_tokens)}",
        ]
        if usage.cache_read_tokens:
            parts.append(f"cache {usage.cache_hit_rate:.0%}")
        if usage.total_cost:
            parts.append(f"${usage.total_cost:.4f}")
        if usage.last_model:
            parts.append(usage.last_model)
        self.query_one("#usage-bar", Static).update(" · ".join(parts))

    def _tick_status(self) -> None:
        """Animate the spinner and the live token count while busy."""
        if not self._vm.busy:
            return
        self._spinner_frame += 1
        self._render_status()

    def _status_label_text(self) -> str:
        frame = SPINNER_FRAMES[self._spinner_frame % len(SPINNER_FRAMES)]
        label = _ACTIVITY_LABELS.get(self._vm.activity, self._vm.activity)
        if self._vm.activity_detail:
            label = f"{label} {self._vm.activity_detail}"
        return f"{frame} {label}…"

    def _status_tokens_text(self) -> str:
        marker = "~" if self._vm.turn_tokens_estimated else ""
        return f"▲ {marker}{self._vm.turn_output_tokens} tok"

    def _render_status(self) -> None:
        bar = self.query_one("#status-bar")
        bar.display = self._vm.busy
        if not self._vm.busy:
            return
        self.query_one("#status-label", Static).update(
            Text(self._status_label_text())
        )
        self.query_one("#status-tokens", Static).update(
            Text(self._status_tokens_text())
        )

    def _render_input_state(self) -> None:
        field = self.query_one(CommandInput)
        # input_pending lives on the bridge; the vm-level signal is simply
        # that a change arrived — reflect readiness via placeholder text.
        field.placeholder = (
            _READY_PLACEHOLDER if self._sink_ready() else _WAITING_PLACEHOLDER
        )

    def _sink_ready(self) -> bool:
        pending = getattr(self._sink, "input_pending", True)
        return bool(pending)
