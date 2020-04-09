from math import sin

precision = 0.0001
target = 0.1

l = 0  # lower bound sin(0) = 0
u = 1.5708  # upper bound sin(pi/2) = 1

while u - l > precision:
    m = (l+u)/2
    l, u = (l, m) if sin(m) > target else (m, u)

print(f'arcsin({target}) is in [{l}, {u}]')

