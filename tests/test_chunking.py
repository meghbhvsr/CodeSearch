import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from codesearch.chunking import extract_functions  # noqa: E402

SOURCE = b'''\
import sys

def top(a):
    """Doc."""
    return a

class Adapter(object):
    @staticmethod
    @cached
    def send(req):
        def retry():
            pass
        return retry()

if sys.version_info[0] == 2:
    def compat():
        print "py2"
'''


def test_extracts_functions_methods_and_nested():
    chunks = {c.name: c for c in extract_functions(SOURCE, "pkg/mod.py")}
    assert set(chunks) == {"top", "Adapter.send", "Adapter.send.retry", "compat"}


def test_lines_and_code_shape():
    send = next(c for c in extract_functions(SOURCE, "pkg/mod.py") if c.name == "Adapter.send")
    assert send.start_line == 8  # first decorator
    assert send.end_line == 13
    assert send.code.startswith("def send(req):")  # decorators aren't embedded
    assert send.code.splitlines()[1].startswith("        def retry")  # body keeps its indentation
    assert send.path == "pkg/mod.py"


def test_python2_and_broken_files_still_yield_functions():
    chunks = extract_functions(b"def ok():\n    print 'x'\n\ndef broken(:\n    pass\n\ndef after():\n    return 1\n", "x.py")
    names = [c.name for c in chunks]
    assert "ok" in names and "after" in names


def test_crlf_normalized():
    (chunk,) = extract_functions(b"def f():\r\n    return 1\r\n", "x.py")
    assert "\r" not in chunk.code and chunk.end_line == 2
