from datetime import datetime
from typing import Any
from colorama import Fore, Style
import time

class Logger:
    """
    This class is a simple logger that prints the current time, the alias of the logger and the message. 
    """
    
    def __init__(self, alias: str):
        """
        This method initializes the instance of the class.

        Parameters
        ----------
        alias : str
            The alias of the logger. 
        """
        self.alias = alias

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        This method is called when the instance is called as a function.

        Parameters
        ----------
        args : Any
            The arguments to be printed.
        kwds : Any
            The keyword arguments to be printed.

        Returns
        -------
        None
        """
        print(datetime.fromtimestamp(time.time()), "::", Fore.GREEN + self.alias, Style.RESET_ALL, "::", *args, **kwds)