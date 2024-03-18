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

    def __call__(self, message):
        """
        This method is called to print a message.

        Parameters
        ----------
        message : str
            The message to be printed.

        Returns
        -------
        None
        """
        kind = "INFO"
        msg = f"{datetime.fromtimestamp(time.time())} :: {Style.BRIGHT + Fore.MAGENTA + self.alias + Style.RESET_ALL} :: {Fore.GREEN + kind + Style.RESET_ALL} :: {message}"

        print(msg)

    def warning(self, message):
        """
        This method is called to print a warning.

        Parameters
        ----------
        message : str


        Returns
        -------
        None
        """
        kind = "WARNING"
        msg = f"{datetime.fromtimestamp(time.time())} :: {Style.BRIGHT + Fore.MAGENTA + self.alias + Style.RESET_ALL} :: {Fore.YELLOW + kind + Style.RESET_ALL} :: {message}"

        print(msg)

    def error(self, message, ErrorType=None):
        """
        This method is called to raise an error

        Parameters
        ----------
        message : str
            The message to be printed.
        ErrorType : Exception
            The type of error to be raised.

        Returns
        -------
        None
        """
        kind = "ERROR"
        msg = f"{datetime.fromtimestamp(time.time())} :: {Style.BRIGHT + Fore.MAGENTA + self.alias + Style.RESET_ALL} :: {Fore.RED + kind + Style.RESET_ALL} :: {message}"

        if ErrorType is None:
            print(msg)
        else:
            raise(ErrorType(msg))