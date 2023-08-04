import sys
import os

def f():
    aa = sys.path[0]
    print(aa)
    print(__file__)
    print(os.path.abspath(__file__))
    dirname, filename = os.path.split(os.path.abspath(__file__))
    print(dirname)
    print(os.path.realpath(__file__))
    print(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    f()
