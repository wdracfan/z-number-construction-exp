import datetime
import inspect

def log_with_timestamp(input_func):
    def output_func(*args, **kwargs):
        started = datetime.datetime.now()
        print(f'[{started}]: START {inspect.getmodule(input_func).__name__}.{input_func.__name__}')
        result = input_func(*args, **kwargs)
        ended = datetime.datetime.now()
        print(f'[{ended}]: FINISH {inspect.getmodule(input_func).__name__}.{input_func.__name__}')
        return result
    return output_func