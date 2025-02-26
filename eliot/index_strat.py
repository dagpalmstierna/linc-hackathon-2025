import time
import multiprocessing
import sys
import os
import requests
from threading import Event

import hackathon_linc as lh

# Add the parent directory (where both 'eliot' and 'paddy' are located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the new API wrapper classes from paddy.
from paddy.Strategy_Execution import Strategy, DataCollection


def index_strategy_func():
    pass

if __name__ == "__main__":
    pass