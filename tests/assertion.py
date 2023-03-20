from contextlib import nullcontext
from typing import Union, Any

import pytest
from _pytest.python_api import RaisesContext


def raises_or_matches(  # type: ignore[misc]
    exception: Union[BaseException, Any]
) -> Union[nullcontext[None], RaisesContext[BaseException]]:
    if isinstance(exception, type) and issubclass(exception, BaseException):
        return pytest.raises(exception)
    return nullcontext()
