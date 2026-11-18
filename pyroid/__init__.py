"""PyRoid - Radially Symmetrical Gyroid Generator & STL Volume Infill Tool."""

import sys

__version__ = "1.1.0"
__author__ = "PyRoid Team"


def main() -> int:
    """Main application entry point dispatcher for PyRoid."""
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from pyroid.cli import main as cli_main

        return cli_main()
    from pyroid.gui import main as gui_main

    return gui_main()
