import math
from itertools import islice
from time import ctime

print(ctime())


print(""
      "fibonacci algorithms.py")

print("Iterative positive and negative")

print(ctime())


def fib(n, x=None):
    if x is None:
        x = [0, 1]
    for i in range(abs(n) - 1):
        x = [x[1], sum(x)]
    return x[1] * math.pow(-1, abs(n) - 1) if n < 0 else x[1] if n else 0


print(ctime())
print("expected Output:")
print(
    "-832040 514229 -317811 196418 -121393 75025 -46368 28657 -17711 "
    "10946 -6765 4181 -2584 1597 -987 610 -377 233 -144 89 -55 34 -21 13"
    " -8 5 -3 2 -1 1 0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 "
    "2584 4181 6765 10946 17711 28657 46368 75025 121393 196418 317811 514229 832040")

for i in range(-30, 31):
    print(fib(i))

print(ctime())
print("Analytic")
print("Binget formula:")


def analytic_fibonacci(n):
    sqrt_5: float = math.sqrt(5)
    p = (1 + sqrt_5) / 2
    q = 1 / p
    return int((p ** n + q ** n) / sqrt_5 + 0.5)


for i in range(1, 31):
    print(analytic_fibonacci(i))

print("expected Output:")

print(
    "1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584 "
    "4181 6765 10946 17711 28657 46368 75025 121393 196418 317811 514229 832040")
print("Iterative")


def fib_iter(n):
    if n < 2:
        return n
    fib_prev = 1
    fib = 1
    for num in range(2, n):
        fib_prev, fib = fib, fib + fib_prev
    return fib


print("Recursive")


def fib_rec(n):
    if n < 2:
        return n
    else:
        return fib_rec(n - 1) + fib_rec(n - 2)


print("Recursive with Memoization")


def fib_memo():
    pad = {0: 0, 1: 1}

    def func(n):
        if n not in pad:
            pad[n] = func(n - 1) + func(n - 2)
        return pad[n]

    return func


fm = fib_memo()
for i in range(1, 31):
    print(fm(i))

print("expected Output:")

print("1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 "
      "2584 4181 6765 10946 17711 28657 46368 75025 121393 196418 "
      "317811 514229 832040")
print("Better Recursive doesn't need Memoization")
print("The recursive code as written two sections above is incredibly "
      "slow and inefficient"
      " due to the nested recursion calls. Although the memoization "
      "above makes the code "
      "run faster, it is at the cost of extra memory use. The below "
      "code is syntactically "
      "recursive but actually encodes the efficient iterative process, and thus doesn't "
      "require memoization: ")


def fib_fast_rec(n):
    def fib(prv_prv, prv, c):
        if c < 1:
            return prv_prv
        else:
            return fib(prv, prv_prv + prv, c - 1)

    return fib(0, 1, n)


print("However, although much faster and not requiring memory, the above "
      "code can only work to a limited 'n' due to the limit on stack "
      "recursion "
      "depth by Python; it is better to use the iterative code above or "
      "the generative one below.")

print("Generative")


def fib_gen(n):
    a, b = 0, 1
    while n > 0:
        yield a
        a, b, n = b, a + b, n - 1


for i in fib_gen(11):
    print(i)

print("Example use: "
      "for i in fibGen(11)")

print("expect: [0,1,1,2,3,5,8,13,21,34,55]")

print("Matrix-Based")
print("Translation of the matrix-based approach used in F#.")


def prev_pow_two(n):
    print("Gets the power of two that is less than or equal to the given input")
    if (n & -n) == n:
        return n
    else:
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1
        return n / 2


def crazy_fib(n):
    print("Crazy fast fibonacci number calculation")
    pow_two = prev_pow_two(n)

    q = r = i = 1
    s = 0

    while i < pow_two:
        i *= 2
        q, r, s = q * q + r * r, r * (q + s), (r * r + s * s)

    while i < n:
        i += 1
        q, r, s = q + r, q, r

    return q


print("Large step recurrence")
print("This is much faster for a single, large value of n: ")


def fib(n, c=None):
    if c is None:
        c = {0: 1, 1: 1}
    if n not in c:
        x = n // 2
        c[n] = fib(x - 1) * fib(n - x - 1) + fib(x) * fib(n - x)
    return c[n]


print("calculating it takes a few seconds, printing it takes eons... original ex. 100000000 now 1024")
print("fib(1024)")
print(fib(1024))

print("Same as above but slightly faster")
print("Putting the dictionary outside the function makes this about 2 seconds faster, could just make a wrapper:")

F = {0: 0, 1: 1, 2: 1}


def fib(n):
    if n in F:
        return F[n]
    f1 = fib(n // 2 + 1)
    f2 = fib((n - 1) // 2)
    F[n] = (f1 * f1 + f2 * f2 if n & 1 else f1 * f1 - f2 * f2)
    return F[n]


print(fib(1024))


print("Generative with Recursion")
print("This can get very slow and uses a lot of memory. Can be sped up by caching the generator results.")
print("Yield fib[n+1] + fib[n]")
print("yield 1  ;  have to start somewhere")
print("Yield fib[n+1] + fib[n]")


def fib():
    yield 1  # have to start somewhere
    lhs, rhs = fib(), fib()
    yield next(lhs)
    # move lhs one iteration ahead
    while True:
        yield next(lhs) + next(rhs)


f = fib()
for _ in range(1, 9):
    print(next(f))
print("expected Output:")
print("[1, 1, 2, 3, 5, 8, 13, 21, 34]")

print("Another version of recursive generators solution, starting from 0")


def fib2():
    yield 0
    yield 1
    a, b = fib2(), fib2()
    next(b)
    while True:
        yield next(a) + next(b)


print(tuple(islice(fib2(), 10)))
