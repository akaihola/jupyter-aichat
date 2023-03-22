import re
from types import EllipsisType
from typing import Union, Optional

import pytest

from jupyter_aichat.schedule import Schedule, SchedulePattern, parse_schedule
from tests.assertion import raises_or_matches


@pytest.mark.kwparametrize(
    dict(pattern=[], expect=[0]),
    dict(pattern=[0], expect=[0]),
    dict(pattern=[1], expect=[1]),
    dict(pattern=[0, 2], expect=[0, 2]),
    dict(pattern=[0, 2, ...], start=5, expect=[5, 7, 9, 11, 13, 15, 17, 19]),
    dict(pattern=[1, 2, 4, ..., 8], start=1, expect=[2, 3, 5, 7, 9]),
    dict(pattern=[1, 3, ..., 9, 12, ...], start=4, expect=[5, 7, 9, 11, 13, 16, 19]),
    start=None,
)
def test_should_execute(
    pattern: list[Union[int, EllipsisType]], start: Optional[int], expect: list[int]
) -> None:
    """Test `Schedule.should_send`."""
    schedule = Schedule(pattern, *[] if start is None else [start])

    result = [step for step in range(20) if schedule.should_send(step)]

    assert result == expect


@pytest.mark.kwparametrize(
    dict(pattern=[...], expect="An Ellipsis must be preceded by two integers."),
    dict(pattern=[42, ...], expect="An Ellipsis must be preceded by two integers."),
    dict(
        pattern=[1, 2, ..., ...], expect="An Ellipsis must be preceded by two integers."
    ),
    dict(
        pattern=[1, 2, ..., 3, ...],
        expect="An Ellipsis must be preceded by two integers.",
    ),
    dict(
        pattern=[-1],
        expect="Schedule pattern ints must be ascending and non-negative.",
    ),
    dict(
        pattern=[42, 1],
        expect="Schedule pattern ints must be ascending and non-negative.",
    ),
    dict(
        pattern=[1, 2, 3, ..., 8, 7, 9, ..., 11],
        expect="Schedule pattern ints must be ascending and non-negative.",
    ),
    dict(
        pattern=[1, 2, None, 3],
        expect="The schedule pattern must consist of ints and Ellipses.",
    ),
)
def test_validate_pattern(pattern: list[Union[int, EllipsisType]], expect: str) -> None:
    """Test `Schedule._validate_pattern`."""
    with pytest.raises(ValueError, match=f"^{re.escape(expect)}$"):
        Schedule(pattern)


@pytest.mark.kwparametrize(
    dict(expression="schedule=[]", expect=(Schedule(pattern=[0], start=0), "")),
    dict(expression=" schedule=[0]", expect=(Schedule(pattern=[0], start=0), "")),
    dict(expression="schedule =[1]", expect=(Schedule(pattern=[1], start=0), "")),
    dict(expression="schedule= [0, 2]", expect=(Schedule(pattern=[0, 2], start=0), "")),
    dict(
        expression="schedule=[0, 2, ...,]",
        expect=(Schedule(pattern=[0, 2, ...], start=0), ""),
    ),
    dict(
        expression="schedule=[1, 2, 4, ..., 8] followed by the system message",
        expect=(
            Schedule(pattern=[1, 2, 4, ..., 8], start=0),
            " followed by the system message",
        ),
    ),
    dict(
        expression=" schedule = [ 1 , 3 , ... , 9 , 12 , ... ]  remaining text  ",
        start=9,
        expect=(
            Schedule(pattern=[1, 3, ..., 9, 12, ...], start=9),
            "  remaining text  ",
        ),
    ),
    dict(
        expression="No schedule at all for this message.",
        start=9,
        expect=(
            Schedule(pattern=[0], start=9),
            "No schedule at all for this message.",
        ),
    ),
    dict(
        expression="schedule bla bla bla",
        start=9,
        expect=(Schedule(pattern=[0], start=9), "schedule bla bla bla"),
    ),
    dict(
        expression="schedule= bla bla bla",
        start=9,
        expect=(Schedule(pattern=[0], start=9), "schedule= bla bla bla"),
    ),
    dict(
        expression="schedule=[ bla bla",
        start=9,
        expect=ValueError("Unexpected NAME 'bla' at pos 11 in 'schedule=[ bla bla'."),
    ),
    start=0,
)
def test_parse_schedule(
    expression: str, start: int, expect: tuple[Schedule, str]
) -> None:
    """Test `parse_schedule`."""
    with raises_or_matches(expect):
        # end of test setup

        result = parse_schedule(expression, start)

        assert result == expect
