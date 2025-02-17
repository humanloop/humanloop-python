from contextlib import contextmanager, redirect_stdout
from typing import ContextManager
import io
from typing import TextIO
import pytest


@pytest.fixture()
def capture_stdout() -> ContextManager[TextIO]:
    @contextmanager
    def _context_manager():
        f = io.StringIO()
        with redirect_stdout(f):
            yield f

    return _context_manager  # type: ignore [return-value]
