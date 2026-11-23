"""Run the LAN API without a console and retain the current run's log."""

from contextlib import redirect_stderr, redirect_stdout
import os
from pathlib import Path
import traceback


if __name__ == "__main__":
    log_dir = Path(os.environ["LOCALAPPDATA"]) / "ProceduralNebula"
    log_dir.mkdir(exist_ok=True)
    with (log_dir / "server.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                from api import main

                main()
            except Exception:
                traceback.print_exc()
                raise SystemExit(1) from None
