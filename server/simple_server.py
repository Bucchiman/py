#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     simple_server
# Author:       8ucchiman
# CreatedDate:  2023-07-04 21:32:53
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    書籍Interface 特集1 C言語と比べて理解する
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


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Hello, world.", "utf-8"))

        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes("Not Found", "utf-8"))

    def log_message(self, format, *args):
        return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handler requests in a separate thread."""


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    server = ThreadedHTTPServer((hostName, serverPort), Handler)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        pass
    server.server_close()
    print("Server stopped.")


if __name__ == "__main__":
    main()

