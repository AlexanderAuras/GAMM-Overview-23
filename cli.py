import plotext as plt
import readline
import sys

"""
i = 0
#plt.plot_size(plt._global.figure._width, plt._global.figure._height-1)
buffer = ""
while True:
    #plt.clt()
    #plt.cld()
    #y = plt.sin(periods=2, length=1000, phase=i/60)
    #plt.plot(y)
    #plt.show()
    buffer = readline.get_line_buffer()
    sys.stdout.write(buffer+"\n")
    #plt.sleep(1.0/60.0)
    time.sleep(1.0/20.0)
    i += 1
"""

import time
import readline
import threading
import sys

def print_diag():
    i = 0
    while True:
        plt.clt()
        plt.cld()
        #sys.stdout.write("\r"+" "*(len(readline.get_line_buffer())+2)+"\r")
        y = plt.sin(periods=2, length=1000, phase=i/60)
        plt.plot(y)
        plt.show()
        sys.stdout.write("> " + readline.get_line_buffer())
        sys.stdout.flush()
        plt.sleep(1.0/60.0)
        i += 1

threading.Thread(target=noisy_thread).start()
while True:
    s = input("> ")
    if s == "d":
        print_diag()