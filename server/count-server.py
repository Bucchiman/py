#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     count-server
# Author:       8ucchiman
# CreatedDate:  2023-07-04 23:23:54
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd

hostName = "127.0.0.1"
serverPort = 8171

class Counter:
    def __init__(self):
        self.count = 0

    def get(self):
        return self.count

    def set(self, i):
        self.count = i

counter = Counter()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global counter
        if self.path == "/":
            count = counter.get() + 1
            print(count)
            counter.set(count)

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Hello, world. {}".format(count), "utf-8"))

        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Not Found", "utf-8"))

    def log_message(self, format, *args):
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""



def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    server = ThreadedHTTPServer((hostName, serverPort), Handler)
    print("Server started http://%s:%s" %(hostName, serverPort))

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        pass

    server.server_close()
    print("Server stopped.")
    pass


if __name__ == "__main__":
    main()

