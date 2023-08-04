# 1/(1+e^-(0.1*x - 1)) *6 - 2
import math


class Sigmoid:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x):
        return self.d + self.c * 1 / (1 + math.exp(-(self.a*x + self.b)))

if __name__ == '__main__':
    s = Sigmoid(a=0.1, b=-1, c=6, d=-2)
    print(s(10))