# derivative of x**4 - 3*x**3 + 2
def df(x):
    return 4*x**3 - 9*x**2

current = 4.0
iterations = 20
alpha = 0.01

for _ in range(iterations):
    print(f'current = {current}')
    current = current - alpha * df(current)

print(f'Minimum at: {current}')

