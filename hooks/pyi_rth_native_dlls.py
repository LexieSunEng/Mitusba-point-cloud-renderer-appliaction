# Runtime hook: make Dr.Jit / Mitsuba native DLLs discoverable on Windows.
import os
import sys


def _add(path):
    if not path or not os.path.isdir(path):
        return
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass


if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    root = sys._MEIPASS
    _add(root)
    _add(os.path.join(root, "drjit"))
    _add(os.path.join(root, "mitsuba"))
    _add(os.path.join(root, "mitsuba", "plugins"))
