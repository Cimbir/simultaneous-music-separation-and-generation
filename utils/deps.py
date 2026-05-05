from zipfile import Path
import sys

def add_msg_ld_to_path():
    """Add MSG-LD source directory to sys.path for imports."""
    MSG_LD_ROOT = Path("../MSG-LD").resolve()
    MSG_LD_SRC = MSG_LD_ROOT / "src"
    for p in [str(MSG_LD_SRC), str(MSG_LD_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)
