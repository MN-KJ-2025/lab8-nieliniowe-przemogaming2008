# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
from typing import Callable


def func(x: int | float | np.ndarray) -> int | float | np.ndarray:
    """Funkcja wyliczająca wartości funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.exp(-2 * x) + x**2 - 1


def dfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości pierwszej pochodnej (df(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    df(x) = -2 * e^(-2x) + 2x

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return -2 * np.exp(-2 * x) + 2 * x


def ddfunc(x: np.ndarray) -> np.ndarray:
    """Funkcja wyliczająca wartości drugiej pochodnej (ddf(x)) funkcji f(x).
    f(x) = e^(-2x) + x^2 - 1
    ddf(x) = 4 * e^(-2x) + 2

    Args:
        x (int | float | np.ndarray): Argumenty funkcji.

    Returns:
        (int | float | np.ndarray): Wartości funkcji f(x).

    Raises:
        TypeError: Jeśli argument x nie jest typu `np.ndarray`, `float` lub 
            `int`.
        ValueError: Jeśli argument x nie jest jednowymiarowy.
    """
    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return 4 * np.exp(-2 * x) + 2


def bisection(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    if not isinstance(epsilon, float):
        return None
    if not isinstance(max_iter, int):
        return None
    if not callable(f):
        return None
    if a >= b:
        return None
    if epsilon < 0:
        return None
    if max_iter <= 0:
        return None

    if f(a) * f(b) >= 0:
        return None

    counter = 0
    x0 = (a + b) / 2

    while abs(f(x0)) > epsilon and counter < max_iter:
        counter += 1

        if f(a) * f(x0) < 0:
            b = x0
        else:
            a = x0

        x0 = (a + b) / 2

    return x0, counter + 1

def secant(
    a: int | float,
    b: int | float,
    f: Callable[[float], float],
    epsilon: float,
    max_iters: int,
) -> tuple[float, int] | None:
    if not isinstance(a, (float, int)) or not isinstance(b, (float, int)):
        return None
    if not callable(f):
        return None
    if a >= b:
        return None
    if epsilon < 0:
        return None
    if max_iters <= 0:
        return None

    fa = f(a)
    fb = f(b)
    if fa * fb >= 0:
        return None

    x = (a * fb - b * fa) / (fb - fa)
    fx = f(x)
    counter = 1

    while abs(fx) > epsilon and counter < max_iters:
        if fa * fx < 0:
            b = x
            fb = fx
        else:
            a = x
            fa = fx

        x = (a * fb - b * fa) / (fb - fa)
        fx = f(x)
        counter += 1

    return x, counter

def difference_quotient(
    f: Callable[[float], float], x: int | float, h: int | float
) -> float | None:
    if not isinstance(x, (float, int)) or not isinstance(h, (float, int)):
        return None
    if not callable(f):
        return None
    if h == 0:
        return None
    return (f(x + h) - f(x)) / h


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    ddf: Callable[[float], float],
    a: int | float,
    b: int | float,
    epsilon: float,
    max_iter: int,
) -> tuple[float, int] | None:
    """Funkcja aproksymująca rozwiązanie równania f(x) = 0 metodą Newtona.

    Args:
        f (Callable[[float], float]): Funkcja, dla której poszukiwane jest 
            rozwiązanie.
        df (Callable[[float], float]): Pierwsza pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        ddf (Callable[[float], float]): Druga pochodna funkcji, dla której 
            poszukiwane jest rozwiązanie.
        a (int | float): Początek przedziału.
        b (int | float): Koniec przedziału.
        epsilon (float): Tolerancja zera maszynowego (warunek stopu).
        max_iter (int): Maksymalna liczba iteracji.

    Returns:
        (tuple[float, int]):
            - Aproksymowane rozwiązanie,
            - Liczba wykonanych iteracji.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not callable(f) : return None
    if not isinstance(a, (float, int)) or not isinstance(b, (float, int)): return None
    if f(a) * f(b) >= 0 : return None
    if a >= b : return None
    if not isinstance(epsilon ,float) : return None
    if not isinstance(max_iter ,int) : return None
    if epsilon < 0 : return None
    if max_iter <= 0 : return None

    fa = f(a)
    fb = f(b)
    dda = ddf(a)
    ddb = ddf(b)

    if fa * dda > 0:
        x = float(a)
    elif fb * ddb > 0:
        x = float(b)
    else:
        x = (a + b) / 2

    for it in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            return None

        x_next = x - fx / dfx

        if abs(f(x_next)) <= epsilon:
            return (x_next, it)

        x = x_next

    return (x, max_iter)
h = 1e-6
df = lambda x : difference_quotient(main.func,x,h)
ddf = lambda x : difference_quotient(df,x,h)

