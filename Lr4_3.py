import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import math

# N

x = [3, 3.3, 2.5, 2]
y = [6, 6.5, 2.5, 2]

binom = lambda n, i: math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
Bt = lambda n, p, t: sum([binom(n, i) * ((1 - t) ** (n - i)) * (t ** i) * p[i] for i in range(0, n + 1)])

n = len(x) - 1
bx = [Bt(n, x, t) for t in np.arange(0, 1., 0.01)]
by = [Bt(n, y, t) for t in np.arange(0, 1., 0.01)]

plt.plot(bx, by)
plt.plot(x, y, 'o')
x = [2, 2.8, 5, 6]
y = [2, 3, 5.8, 6]
n = len(x) - 1
bx = [Bt(n, x, t) for t in np.arange(0, 1., 0.01)]
by = [Bt(n, y, t) for t in np.arange(0, 1., 0.01)]

x = [2, 2.8, 5, 6]
y = [2, 3, 5.8, 6]
n = len(x) - 1
bx = [Bt(n, x, t) for t in np.arange(0, 1., 0.01)]
by = [Bt(n, y, t) for t in np.arange(0, 1., 0.01)]
plt.plot(bx, by)
plt.plot(x, y, 'o')
x = [6, 5.8, 4.5, 5]
y = [6, 5, 2.5, 2]
n = len(x) - 1
bx = [Bt(n, x, t) for t in np.arange(0, 1., 0.01)]
by = [Bt(n, y, t) for t in np.arange(0, 1., 0.01)]

plt.plot(bx, by)
plt.plot(x, y, 'o')
x = [5, 5.5, 6.4, 6.5]
y = [2, 2.2, 2.8, 3]
n = len(x) - 1
bx = [Bt(n, x, t) for t in np.arange(0, 1., 0.01)]
by = [Bt(n, y, t) for t in np.arange(0, 1., 0.01)]

plt.plot(bx, by)
plt.plot(x, y, 'o')
plt.show()

# A
x = [2, 2.5, 3.5, 5]
y = [2, 4, 4, 7]

binom = lambda n, i: math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
Bt = lambda n, p, t: sum([binom(n, i) * ((1 - t) ** (n - i)) * (t ** i) * p[i] for i in range(0, n + 1)])

n = len(x) - 1
bx = [Bt(n, x, t) for t in np.arange(0, 1., 0.01)]
by = [Bt(n, y, t) for t in np.arange(0, 1., 0.01)]

plt.plot(bx, by)
plt.plot(x, y, 'o')
x = [5, 6.5, 7.5, 8]
y = [7, 4, 4, 2]
n = len(x) - 1
bx = [Bt(n, x, t) for t in np.arange(0, 1., 0.01)]
by = [Bt(n, y, t) for t in np.arange(0, 1., 0.01)]

plt.plot(bx, by)
plt.plot(x, y, 'o')
x = [3, 5, 5, 7]
y = [4, 4.5, 3.5, 4]
n = len(x) - 1
bx = [Bt(n, x, t) for t in np.arange(0, 1., 0.01)]
by = [Bt(n, y, t) for t in np.arange(0, 1., 0.01)]

plt.plot(bx, by)
plt.plot(x, y, 'o')
plt.show()
