import os
import time
import multiprocessing
import numpy as np

def worker(n):
    factorial = np.math.factorial(90000)
    # rank = multiprocessing.current_process().name one can also use
    rank = multiprocessing.current_process()._identity[0]
    print(f'I am processor {rank}, got n={n}, and finished calculating the factorial.')


cpu_count = multiprocessing.cpu_count()

print(f'Number of CPUs: {cpu_count}')

# input_params_list = range(1, cpu_count+1)


# pool = multiprocessing.Pool(cpu_count)
# pool.map(worker, input_params_list)
# pool.close()
# pool.join()


