from ast import literal_eval
from dataclasses import dataclass
from io import StringIO
from itertools import zip_longest
from token import tok_name
from tokenize import generate_tokens, INDENT, NAME, OP, NUMBER
from types import EllipsisType
from typing import Union

SchedulePattern = list[Union[int, EllipsisType]]


@dataclass
class Schedule:
    """Store and apply a schedule pattern.

    This class encapsulates
    1. a list of ints and Ellipses which define on which steps to schedule a task
    2. the number of the step on which this schedule should start

    The list of ints and Ellipses is called the schedule pattern.
    The number of the step on which this schedule should start is called the start step.
    The ints are the steps on which to schedule the task.
    An Ellipsis continues the arithmetic series defined by the previous two ints until
    the next int.

    """

    pattern: SchedulePattern
    start: int = 0

    def __post_init__(self) -> None:
        """Validate the schedule pattern when instantiating the class."""
        self._validate_pattern()

    def _validate_pattern(self) -> None:
        """Validate the schedule pattern."""
        if not self.pattern:
            self.pattern = [0]
            return
        previous_int = -1
        for prev, item, next_ in zip(
            [..., ...] + self.pattern, [...] + self.pattern, self.pattern
        ):
            if next_ is Ellipsis:
                if not isinstance(item, int) or not isinstance(prev, int):
                    raise ValueError("An Ellipsis must be preceded by two integers.")
                continue
            elif not isinstance(next_, int):
                raise ValueError(
                    "The schedule pattern must consist of ints and Ellipses."
                )
            if next_ <= previous_int:
                raise ValueError(
                    "Schedule pattern ints must be ascending and non-negative."
                )
            previous_int = next_

    def should_send(self, step: int) -> bool:
        """Return whether the task should execute at the given step.

        :param step: The step at which to check whether to execute the task.
        :return: Whether the task should execute at the given step.

        """
        if step < self.start:
            return False
        step -= self.start
        series_offset = 0
        common_difference = 0
        # Iterate over the pattern until we either find a match or know there is no
        # match since the current value in the pattern is later than the given step.
        for this, next_ in zip_longest(self.pattern, self.pattern[1:]):
            if isinstance(this, int):
                if isinstance(next_, int):
                    # an arithmetic series may follow, remember the difference
                    common_difference = next_ - this
                elif next_ is Ellipsis:
                    # an arithmetic series begins, remember the offset
                    series_offset = this + common_difference
                elif next_ is not None:
                    raise ValueError("Invalid value {next_!r} in schedule pattern.")
                if step > this:  # the given step is later than the current value
                    continue
                # Either we found a match, or the current value is later than the given
                # step and there was no match.
                return this == step
            elif this is Ellipsis:
                if step < series_offset:  # the series begins later than the given step
                    return False
                if (
                    isinstance(next_, int) and step >= next_
                ):  # step later than the series
                    continue
                if (step - series_offset) % common_difference == 0:
                    # the step matches one of the values in the arithmetic series
                    return True
                return False  # step is in the range of the series, but not a member
            else:
                raise ValueError("Invalid value {this!r} in schedule pattern.")
        return False


class Action:
    pass


class Goto(int, Action):
    pass


class Emit:
    pass


class EmitAndGoto(Goto, Emit):
    pass


class Cancel(Action):
    pass


END = -1
ANY = ""
GRAMMAR: dict[int, dict[int, dict[str, Action]]] = {
    0: {INDENT: {ANY: Goto(1)}, NAME: {"schedule": Goto(2), ANY: Cancel()}},
    1: {NAME: {"schedule": Goto(2)}},
    2: {OP: {"=": Goto(3)}},
    3: {OP: {"[": Goto(4)}},
    4: {NUMBER: {ANY: EmitAndGoto(5)}, OP: {"...": EmitAndGoto(5), "]": Goto(END)}},
    5: {OP: {",": Goto(4), "]": Goto(END)}},
}


def parse_schedule(string: str, start: int) -> tuple[Schedule, str]:
    """Create a schedule from a string.

    :param string: The string to parse.
    :param start: The start step.
    :return: The schedule parsed from the string, and the remaining string.

    """

    state = 0
    pattern: SchedulePattern = []
    for token in generate_tokens(StringIO(string).readline):
        try:  # [token.type] and [ANY] may raise a KeyError
            grammar_strings = GRAMMAR[state][token.type]
            if token.string in grammar_strings:
                action = grammar_strings[token.string]
            else:
                action = grammar_strings[ANY]
        except KeyError as exc:
            raise ValueError(
                f"Unexpected {tok_name[token.type]} {token.string!r} at pos"
                f" {token.start[1]} in {string!r}."
            ) from exc
        if isinstance(action, Cancel):
            return Schedule([0], start), string[token.start[1] :]
        if isinstance(action, Goto):
            state = int(action)
        if isinstance(action, Emit):
            pattern.append(literal_eval(token.string))
        if state == END:
            break
    else:
        raise ValueError(f"Unexpected end of string {string!r}.")
    return Schedule(pattern, start), string[token.end[1] :]
