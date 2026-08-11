#!/usr/bin/env python
import sys

try:
    from gto_cli.cli import main
except ModuleNotFoundError as error:
    if error.name == "cv2":
        print(
            "OpenCV is missing from this Python. Run this project with:\n"
            "  E:\\code\\conda\\python.exe gto.py <command>",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    raise


if __name__ == "__main__":
    sys.exit(main())
