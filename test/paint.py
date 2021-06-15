import matplotlib.pyplot as plt
import math

def paint():
    x = range(60000)
    y = [1 / (1 + math.exp(-(float(i) - 60000) / 500.0)) for i in x]
    plt.figure()
    plt.plot(x, y)
    plt.show()

paint()