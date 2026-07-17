"""Textual application for the plan-and-task TUI.

Renders the ``PlanTaskViewModel`` and forwards submitted input to an
``InputSink`` (the event-bus bridge in production, a stub in tests). All ECS
updates arrive through :meth:`PlanTaskTuiApp.dispatch_change`, which posts a
``VmChanged`` message so widget mutation happens on Textual's message pump.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from rich.cells import cell_len
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.selection import Selection
from textual.strip import Strip
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    OptionList,
    RadioSet,
    RichLog,
    SelectionList,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option, OptionDoesNotExist
from textual.widgets.selection_list import Selection as ListSelection

from examples.e2e.plan_and_task.ask_tool import AskQuestion, QuestionAnswer
from examples.e2e.plan_and_task.tui.view_model import (
    PlanTaskViewModel,
    SubagentRun,
    TranscriptEntry,
    UiChange,
)


class InputSink(Protocol):
    """Receiver for user actions raised by the app."""

    def submit_input(self, text: str) -> bool: ...

    def submit_answers(self, answers: list[QuestionAnswer] | None) -> bool: ...

    def request_quit(self) -> None: ...


class CommandInput(TextArea):
    """Multi-line prompt with slash-command suggestions and completion.

    Enter submits the whole buffer; Ctrl+J inserts a newline, so multi-line
    messages can be composed before sending. Completion-list navigation keys
    (up/down/tab/enter/escape) are offered to the app first; anything the
    completion list does not consume falls through to the regular ``TextArea``
    behavior (arrows move between lines, printable keys insert).

    The widget keeps an ``Input``-compatible surface — a ``value`` alias for
    ``text``, a linear ``cursor_position``, and a ``Submitted`` message — so
    the surrounding app and its tests treat it like the single-line prompt it
    replaced.
    """

    class Submitted(Message):
        """Posted when the user presses Enter to send the buffer."""

        def __init__(self, input: CommandInput, value: str) -> None:
            self.input = input
            self.value = value
            super().__init__()

        @property
        def control(self) -> CommandInput:
            return self.input

    def __init__(self, *, placeholder: str = "") -> None:
        # tab_behavior="focus" keeps Tab free for command completion instead of
        # indenting; soft_wrap lets long lines wrap inside the bordered box.
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="focus",
            placeholder=placeholder,
        )

    @property
    def value(self) -> str:
        """The buffer text (``Input``-compatible alias for ``text``)."""
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        # ``TextArea.load_text`` parks the cursor at (0, 0); ``Input`` left it at
        # the end. Restore end-of-buffer placement so typing/Ctrl+J after a
        # programmatic set appends instead of inserting at the front.
        self.text = new_value
        self.cursor_position = len(new_value)

    @property
    def cursor_position(self) -> int:
        """Cursor offset as a linear character index into ``value``."""
        row, column = self.cursor_location
        lines = self.text.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + column

    @cursor_position.setter
    def cursor_position(self, index: int) -> None:
        text = self.text
        index = max(0, min(index, len(text)))
        prefix = text[:index]
        row = prefix.count("\n")
        column = index - (prefix.rfind("\n") + 1)
        self.move_cursor((row, column))

    async def _on_key(self, event: events.Key) -> None:
        app = self.app
        if isinstance(app, PlanTaskTuiApp) and app.handle_completion_key(event.key):
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            # Enter sends the message; Ctrl+J is the newline key (handled below).
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        if event.key == "ctrl+j":
            # Mirror TextArea's own Enter handling so a selection is replaced
            # and the cursor lands after the inserted newline.
            event.stop()
            event.prevent_default()
            start, end = self.selection
            self._replace_via_keyboard("\n", start, end)
            return
        await super()._on_key(event)


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


class QuestionScreen(ModalScreen[list[QuestionAnswer] | None]):
    """Modal that puts the agent's ``ask_question`` questions to the user.

    Renders each question as either a radio group (single choice), a selection
    list (multi-select), or a free-text field, plus a free-text field on option
    questions so the user can add an "other" answer. Dismisses with the list of
    :class:`QuestionAnswer` on submit, or ``None`` when cancelled.
    """

    CSS = """
    QuestionScreen {
        align: center middle;
    }
    #question-dialog {
        width: 80;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    #question-title {
        height: auto;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    /* The body is the only flexible region: it takes the space left between the
       title and the pinned footer/buttons and scrolls when content overflows,
       so the Submit/Cancel buttons are always on screen. */
    #question-body {
        height: 1fr;
        min-height: 3;
    }
    .question-header {
        height: auto;
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }
    .question-text {
        height: auto;
        margin-bottom: 1;
    }
    /* Option descriptions render here as full, wrapping text (radio/selection
       labels are single-line and would truncate long descriptions). */
    .option-legend {
        height: auto;
        color: $text-muted;
        margin: 0 0 1 1;
    }
    #question-body RadioSet, #question-body SelectionList {
        height: auto;
        width: 1fr;
    }
    #question-body Input {
        margin-bottom: 1;
    }
    #question-hint {
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    #question-buttons {
        height: auto;
        align-horizontal: right;
    }
    #question-buttons Button {
        margin-left: 2;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("ctrl+s", "submit", "Submit"),
    ]

    def __init__(self, questions: list[AskQuestion]) -> None:
        super().__init__()
        self._questions = questions

    def compose(self) -> ComposeResult:
        with Vertical(id="question-dialog"):
            yield Static("The agent needs your input", id="question-title")
            with VerticalScroll(id="question-body"):
                for index, question in enumerate(self._questions):
                    yield Static(question.header, classes="question-header")
                    yield Static(question.question, classes="question-text")
                    if question.options and question.multi_select:
                        yield SelectionList[int](
                            *[
                                ListSelection(option.label, opt_index)
                                for opt_index, option in enumerate(question.options)
                            ],
                            id=f"q{index}-select",
                        )
                        yield from self._option_legend(question)
                    elif question.options:
                        yield RadioSet(
                            *[option.label for option in question.options],
                            id=f"q{index}-radio",
                        )
                        yield from self._option_legend(question)
                    yield Input(
                        placeholder=(
                            "type a custom answer…"
                            if question.options
                            else "type your answer…"
                        ),
                        id=f"q{index}-input",
                    )
            yield Static(
                "Submit button or Ctrl+S to send · Esc to cancel", id="question-hint"
            )
            with Horizontal(id="question-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Submit", variant="primary", id="submit")

    @staticmethod
    def _option_legend(question: AskQuestion) -> Iterator[Static]:
        """Full, wrapping descriptions for a question's options.

        Radio/selection labels are single-line and truncate, so descriptions
        are shown here in a wrapping ``Static`` legend instead of on the
        control itself. Yields nothing when no option has a description.
        """
        if not any(option.description for option in question.options):
            return
        legend = Text()
        for option in question.options:
            if not option.description:
                continue
            legend.append(f"• {option.label}: ", style="bold")
            legend.append(f"{option.description}\n")
        yield Static(legend, classes="option-legend")

    def on_mount(self) -> None:
        # Focus the first interactive control so keyboard use works immediately
        # (arrows to pick options, Enter to submit); fall back to the input.
        focusables = self.query("RadioSet, SelectionList, Input")
        if focusables:
            focusables.first().focus()

    @on(Button.Pressed, "#submit")
    def _on_submit_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_submit()

    @on(Button.Pressed, "#cancel")
    def _on_cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_cancel()

    @on(Input.Submitted)
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter inside a field submits the whole form; keep it from bubbling to
        # the main command input underneath the modal.
        event.stop()
        self.action_submit()

    def action_submit(self) -> None:
        # Always dismiss, even if answer collection hits an unexpected widget
        # error: a modal that fails to dismiss would leave the ask_question
        # future unresolved and hang the agent loop forever.
        self.dismiss(self._collect_answers())

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _collect_answers(self) -> list[QuestionAnswer]:
        answers: list[QuestionAnswer] = []
        for index, question in enumerate(self._questions):
            selected = self._collect_selection(index, question)
            try:
                custom = self.query_one(f"#q{index}-input", Input).value.strip()
            except NoMatches:
                custom = ""
            answers.append(
                QuestionAnswer(
                    header=question.header,
                    question=question.question,
                    selected=selected,
                    custom_text=custom or None,
                )
            )
        return answers

    def _collect_selection(self, index: int, question: AskQuestion) -> list[str]:
        try:
            if question.options and question.multi_select:
                widget = self.query_one(f"#q{index}-select", SelectionList)
                return [
                    question.options[value].label for value in sorted(widget.selected)
                ]
            if question.options:
                radio = self.query_one(f"#q{index}-radio", RadioSet)
                if radio.pressed_index >= 0:
                    return [question.options[radio.pressed_index].label]
        except (NoMatches, IndexError):
            return []
        return []


class SubagentInspectorScreen(ModalScreen[None]):
    """Read-only inspector for delegated subagents.

    Lists every subagent run on the left and, for the selected run, shows its
    streamed reasoning, streamed content, and full final result on the right.
    The screen reads the live :class:`PlanTaskViewModel` directly, so while it
    stays open :meth:`refresh` repaints the currently-streaming run token by
    token — this is where a subagent's *thinking process* becomes visible,
    which the one-line transcript summary can never show.
    """

    CSS = """
    SubagentInspectorScreen {
        align: center middle;
    }
    #inspector-dialog {
        width: 96;
        max-width: 96%;
        height: 90%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    #inspector-title {
        height: auto;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    #inspector-body {
        height: 1fr;
    }
    #inspector-list {
        width: 32;
        height: 1fr;
        border: round $primary 30%;
        margin-right: 1;
    }
    #inspector-detail {
        width: 1fr;
        height: 1fr;
        border: round $primary 30%;
        padding: 0 1;
    }
    #inspector-detail Static {
        height: auto;
        margin-bottom: 1;
    }
    #inspector-hint {
        height: auto;
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "close", "Close")]

    def __init__(
        self, view_model: PlanTaskViewModel, initial_id: str | None = None
    ) -> None:
        super().__init__()
        self._vm = view_model
        self._selected_id = initial_id

    def compose(self) -> ComposeResult:
        with Vertical(id="inspector-dialog"):
            yield Static("Subagent inspector", id="inspector-title")
            with Horizontal(id="inspector-body"):
                run_list: OptionList = OptionList(id="inspector-list")
                yield run_list
                with VerticalScroll(id="inspector-detail"):
                    yield Static(id="inspector-reasoning")
                    yield Static(id="inspector-content")
            yield Static(
                "↑↓ pick a subagent · Esc to close", id="inspector-hint"
            )

    def on_mount(self) -> None:
        self._rebuild_list()

    @staticmethod
    def _run_id(index: int, run: SubagentRun) -> str:
        # Correlation ids are unique per delegation; fall back to the index for
        # the rare run created without one so option ids never collide.
        return run.correlation_id or f"idx-{index}"

    def _selected_run(self) -> SubagentRun | None:
        for index, run in enumerate(self._vm.subagent_runs):
            if self._run_id(index, run) == self._selected_id:
                return run
        return None

    def _default_selection(self) -> str | None:
        """Prefer the latest running run, else the most recent one."""
        runs = self._vm.subagent_runs
        if not runs:
            return None
        for index in range(len(runs) - 1, -1, -1):
            if runs[index].status == "running":
                return self._run_id(index, runs[index])
        return self._run_id(len(runs) - 1, runs[-1])

    def _rebuild_list(self) -> None:
        option_list = self.query_one("#inspector-list", OptionList)
        highlighted_before = self._selected_id
        option_list.clear_options()
        for index, run in enumerate(self._vm.subagent_runs):
            option_list.add_option(
                Option(self._run_label(run), id=self._run_id(index, run))
            )
        if not self._vm.subagent_runs:
            self._render_detail()
            return
        target = highlighted_before or self._default_selection()
        self._select(option_list, target)

    def _select(self, option_list: OptionList, run_id: str | None) -> None:
        if run_id is None:
            return
        try:
            index = option_list.get_option_index(run_id)
        except OptionDoesNotExist:
            index = 0
        self._selected_id = run_id
        option_list.highlighted = index
        self._render_detail()

    @staticmethod
    def _run_label(run: SubagentRun) -> Text:
        icon = _RUN_ICONS.get(run.status, "?")
        style = _RUN_STYLES.get(run.status, "")
        label = Text(f"{icon} {run.name}\n", style=style)
        detail = (
            f"  {_fmt_tokens(run.stream_chars)} chars"
            if run.status == "running"
            else f"  {run.status}"
        )
        label.append(detail, style="dim")
        return label

    @on(OptionList.OptionHighlighted, "#inspector-list")
    def _on_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        if event.option_id is not None:
            self._selected_id = event.option_id
            self._render_detail()

    @on(OptionList.OptionSelected, "#inspector-list")
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option_id is not None:
            self._selected_id = event.option_id
            self._render_detail()

    def _render_detail(self) -> None:
        reasoning_widget = self.query_one("#inspector-reasoning", Static)
        content_widget = self.query_one("#inspector-content", Static)
        run = self._selected_run()
        if run is None:
            reasoning_widget.update(
                Text("No subagents have run yet.", style="dim italic")
            )
            content_widget.update("")
            return
        reasoning_widget.update(self._reasoning_block(run))
        content_widget.update(self._content_block(run))
        if run.status == "running":
            # Follow the live stream to the newest tokens.
            self.query_one("#inspector-detail", VerticalScroll).scroll_end(
                animate=False
            )

    @staticmethod
    def _reasoning_block(run: SubagentRun) -> Text:
        header = Text()
        header.append(f"{run.name}", style="bold magenta")
        header.append(f"  ({run.status})\n", style="dim")
        if run.task:
            header.append(f"task: {run.task}\n", style="dim")
        header.append("\n")
        if run.reasoning:
            header.append("Reasoning\n", style="bold")
            header.append(run.reasoning, style="dim italic")
        else:
            header.append(
                "No reasoning streamed"
                if run.status != "running"
                else "waiting for the subagent to think…",
                style="dim italic",
            )
        return header

    @staticmethod
    def _content_block(run: SubagentRun) -> Text:
        block = Text()
        if run.content:
            block.append("\nOutput\n", style="bold")
            block.append(run.content)
        # The final result may be an envelope distinct from the raw stream, so
        # show it explicitly once the delegation reports back.
        if run.result and run.result != run.content:
            block.append("\n\nResult\n", style="bold green")
            block.append(run.result)
        if run.error:
            block.append("\n\nError\n", style="bold red")
            block.append(run.error, style="red")
        return block

    def refresh_runs(self) -> None:
        """Repaint the list and detail from the live view model (app-driven)."""
        self._rebuild_list()

    def action_close(self) -> None:
        self.dismiss(None)


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
_READY_PLACEHOLDER = "type a message or /command — Enter to send · Ctrl+J newline"

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
    #command-list {
        display: none;
        height: auto;
        max-height: 8;
        border: round $accent 50%;
        background: $surface;
    }
    CommandInput {
        border: round $accent;
        height: auto;
        min-height: 3;
        max-height: 10;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit_session", "Quit"),
        ("ctrl+l", "clear_transcript", "Clear log"),
        ("ctrl+o", "inspect_subagents", "Subagents"),
    ]

    def __init__(
        self,
        view_model: PlanTaskViewModel,
        sink: InputSink,
        commands: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """``commands`` holds ``(pattern, hint)`` pairs for the completion list."""
        super().__init__()
        self._vm = view_model
        self._sink = sink
        self._commands = commands
        self._backlog: list[UiChange] = []
        self._last_copied: str | None = None
        self._spinner_frame = 0
        self._inspector: SubagentInspectorScreen | None = None

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
        yield OptionList(id="command-list")
        yield CommandInput(placeholder=_WAITING_PLACEHOLDER)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#task-table", DataTable)
        table.add_columns("task", "status", "retry")
        # The list is driven from the input; keeping it unfocusable stops tab
        # cycling and option clicks from stealing focus from the prompt.
        self.query_one("#command-list", OptionList).can_focus = False
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
                if self._inspector is not None:
                    self._inspector.refresh_runs()
            case "input":
                self._render_input_state()
            case "status":
                self._render_status()
            case "question":
                self._push_question(change.questions)
            case "notify":
                if change.notification is not None:
                    self.notify(change.notification, severity=change.severity)

    @on(CommandInput.Submitted)
    def _handle_submit(self, event: CommandInput.Submitted) -> None:
        if self._sink.submit_input(event.value):
            field = self.query_one(CommandInput)
            field.value = ""

    # -- ask_question modal --------------------------------------------------

    def _push_question(self, questions: list[AskQuestion]) -> None:
        """Open the question modal; its result flows back to the sink."""
        if not questions:
            return
        self.push_screen(QuestionScreen(questions), self._on_question_answered)

    def _on_question_answered(self, answers: list[QuestionAnswer] | None) -> None:
        self._sink.submit_answers(answers)

    # -- slash-command completion ---------------------------------------------

    @on(CommandInput.Changed)
    def _handle_input_changed(self, event: CommandInput.Changed) -> None:
        self._refresh_command_list(event.text_area.text)

    @on(OptionList.OptionSelected, "#command-list")
    def _handle_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            self._accept_completion(event.option_id)

    def handle_completion_key(self, key: str) -> bool:
        """Handle a navigation key for the completion list.

        Returns True when the key was consumed. Called by ``CommandInput``
        before its own key handling so enter can complete instead of submit.
        """
        option_list = self.query_one("#command-list", OptionList)
        if not option_list.display:
            return False
        if key == "escape":
            option_list.display = False
            return True
        if key in ("down", "up"):
            if option_list.option_count:
                step = 1 if key == "down" else -1
                current = option_list.highlighted or 0
                option_list.highlighted = (
                    current + step
                ) % option_list.option_count
            return True
        if key in ("tab", "enter"):
            pattern = self._highlighted_command(option_list)
            if pattern is None:
                option_list.display = False
                return key == "tab"
            if key == "enter" and self.query_one(CommandInput).value.strip() == pattern:
                # The command is already fully typed: let the submit proceed.
                option_list.display = False
                return False
            self._accept_completion(pattern)
            return True
        return False

    def _matching_commands(self, text: str) -> list[tuple[str, str]]:
        if not text.startswith("/"):
            return []
        lowered = text.lower()
        return [
            (pattern, hint)
            for pattern, hint in self._commands
            if pattern.startswith(lowered)
        ]

    def _refresh_command_list(self, text: str) -> None:
        option_list = self.query_one("#command-list", OptionList)
        matches = self._matching_commands(text)
        option_list.clear_options()
        if not matches:
            option_list.display = False
            return
        for pattern, hint in matches:
            label = Text(pattern, style="bold")
            if hint:
                label.append(f"  {hint}", style="dim")
            option_list.add_option(Option(label, id=pattern))
        option_list.highlighted = 0
        option_list.display = True

    def _highlighted_command(self, option_list: OptionList) -> str | None:
        index = option_list.highlighted
        if index is None or not option_list.option_count:
            return None
        return option_list.get_option_at_index(index).id

    def _accept_completion(self, pattern: str) -> None:
        field = self.query_one(CommandInput)
        # Trailing space so argument-taking commands are ready for input; the
        # completed value is no longer a prefix of any pattern, which closes
        # the list via the CommandInput.Changed refresh.
        field.value = f"{pattern} "
        field.cursor_position = len(field.value)
        field.focus()

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

    def action_inspect_subagents(self, initial_id: str | None = None) -> None:
        """Open the subagent inspector (no-op with a hint when none have run)."""
        if not self._vm.subagent_runs:
            self.notify("No subagents have run yet.", timeout=2.0)
            return
        if self._inspector is not None:
            return
        inspector = SubagentInspectorScreen(self._vm, initial_id=initial_id)
        self._inspector = inspector
        self.push_screen(inspector, self._on_inspector_closed)

    def _on_inspector_closed(self, _result: None) -> None:
        self._inspector = None

    @on(events.Click, "#subagent-panel")
    def _handle_subagent_click(self, event: events.Click) -> None:
        """Clicking the sidebar subagent summary opens the inspector."""
        event.stop()
        self.action_inspect_subagents()

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
        lines.append("^O / click to inspect", style="dim")
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
        try:
            bar = self.query_one("#status-bar")
        except NoMatches:
            # A modal (e.g. the subagent inspector) can cover the base screen
            # while a subagent is still running; the recurring status tick must
            # not crash when the status bar is transiently unreachable.
            return
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
