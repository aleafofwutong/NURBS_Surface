from pathlib import Path
FILE=Path(__file__).resolve()
ROOT=FILE.parents[0]
import sys
sys.path.append(str(ROOT))
import time
import add_version
@add_version.add_version
class SecondsCounter:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        return time.time() - self.start_time
    
    def reset(self):
        self.start_time = None
        return
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = self.elapsed()
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        self.reset()
        return
    
    def __str__(self):
        if self.start_time is None:
            return "Timer has not been started."
        return f"Elapsed time: {self.elapsed():.2f} seconds"
    
    def __repr__(self):
        return self.__str__()
    
    def get_elapsed(self):
        return self.elapsed()
    
    def sleep(self, seconds):
        time.sleep(seconds)
        return
    def wait(self, seconds):
        time.sleep(seconds)
        return
    def pause(self, seconds):
        time.sleep(seconds)
        return
    
    def delay(self, seconds):
        time.sleep(seconds)
        return
    
    def hold(self, seconds):
        time.sleep(seconds)
        return
    
    def end(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        elapsed_time = self.elapsed()
        self.reset()
        return elapsed_time
    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        elapsed_time = self.elapsed()
        self.reset()
        return elapsed_time
    def finish(self):
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        elapsed_time = self.elapsed()
        self.reset()
        return elapsed_time
    
    
if __name__ == "__main__":
    counter = SecondsCounter()
    counter.start()
    time.sleep(2)
    print(counter)
    counter.reset()