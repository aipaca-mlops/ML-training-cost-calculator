import sys
from tools.util.stdout import print_err


def exit_with_err(s: str):
    print_err(s)
    sys.exit()