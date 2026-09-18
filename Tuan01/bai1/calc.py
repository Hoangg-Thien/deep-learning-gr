"""Simple command-line calculator used by main_calc.ipynb."""

from __future__ import annotations

import operator
import sys
from collections.abc import Callable


OPERATIONS: dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
}


def calculate(left: float, operation: str, right: float) -> float:
    """Return the result of one binary operation."""
    if operation == "/":
        if right == 0:
            raise ValueError("Khong the chia cho 0.")
        return left / right

    try:
        function = OPERATIONS[operation]
    except KeyError as error:
        raise ValueError(f"Phep tinh khong hop le: {operation}") from error
    return function(left, right)


def main() -> None:
    """Read arguments and print the calculation result."""
    if len(sys.argv) != 4:
        print("Cach dung: python calc.py <so thu nhat> <phep tinh> <so thu hai>")
        print("Vi du: python calc.py 12 + 5")
        raise SystemExit(1)

    try:
        left = float(sys.argv[1])
        operation = sys.argv[2]
        right = float(sys.argv[3])
        result = calculate(left, operation, right)
    except ValueError as error:
        print(f"Loi: {error}")
        raise SystemExit(1) from error

    print(f"{left:g} {operation} {right:g} = {result:g}")


if __name__ == "__main__":
    main()
