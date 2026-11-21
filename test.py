import numpy as np
import matplotlib.pyplot as plt
import math 

x_vals = []
y_vals = []

steps = 100

for i in range(steps + 1): 
    x = (2 * math.pi * i)/steps
    x_vals.append(x)
    y_vals.append(math.sin(x))

plt.plot(x_vals, y_vals)
plt.xlabel("x (radians)")
plt.ylabel("y sin(x)")
plt.title("Test")
plt.grid(True)
plt.show()
