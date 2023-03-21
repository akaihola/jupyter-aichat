import re
from contextlib import nullcontext
from typing import Union, Any, Type

import pytest
from _pytest.python_api import RaisesContext


def raises_or_matches(  # type: ignore[misc]
    exception: Union[BaseException, Type[BaseException], Any]
) -> Union[nullcontext[None], RaisesContext[BaseException]]:
    if isinstance(exception, type) and issubclass(exception, BaseException):
        return pytest.raises(exception)
    if isinstance(exception, BaseException):
        return pytest.raises(type(exception), match=f"^{re.escape(str(exception))}$")
    return nullcontext()
