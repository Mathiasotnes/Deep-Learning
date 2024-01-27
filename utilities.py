import time
import math

def calculate_time(func):
    """
    Decorator to measure the runtime of an
    arbitrary function. 
    """

    def inner(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        runtime = end - begin
        print(f'Time of function: {func.__name__} is {runtime}s')

    return inner