# coding=utf-8
# sigmoidPlot.py
# 简单的Sigmoid函数可视化

from pylab import *

t = arange(-100, 100, 0.01)
s = 1 / (1 + exp(-t))

ax = subplot(211)
ax.plot(t, s)
ax.axis([-5, 5, 0, 1])
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")

ax = subplot(212)
ax.plot(t, s)
ax.axis([-100, 100, 0, 1])
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")

show()
