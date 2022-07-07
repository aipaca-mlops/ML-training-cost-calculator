class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def print_err(s: str):
    print(f"{bcolors.FAIL}ERROR: {s.capitalize()}{bcolors.ENDC}")

def print_warning(s: str):
    print(f"{bcolors.WARNING}WARNING: {s.capitalize()}{bcolors.ENDC}")


def print_ok(s: str):
    print(f"{bcolors.OKGREEN} {s.capitalize()}{bcolors.ENDC}")
