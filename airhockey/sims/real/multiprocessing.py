import termios
import inspect
import sys
import tty
import multiprocessing
import select

class NonBlockingConsole(object):

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return False
    
class ProtectedArray:
    def __init__(self, array):
        self.array = array
        self.lock = multiprocessing.Lock()

    def __getitem__(self, index):
        with self.lock:
            return self.array[index]

    def __setitem__(self, index, value):
        with self.lock:
            self.array[index] = value