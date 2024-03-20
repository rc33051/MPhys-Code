from gab_large_cutoff import g_ab_parallel as g
from time import time


def main():
    start = time()
    print(g(1,0,[1,0,0], 0, 1e6, 0.01, 4))
    print('time:', start-time(), ' seconds')

main()