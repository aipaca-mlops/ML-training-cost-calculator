# dummy_context_mgr is used for when with statement need to have condition
# Reference: https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
import contextlib


@contextlib.contextmanager
def dummy_context_mgr():
    yield None
