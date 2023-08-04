import os
from myapp import app

secret_key = os.environ.get('SECRET_KEY', None)

if not secret_key:
    raise ValueError('You must have "SECRET_KEY" variable')

app.config['SECRET_KEY'] = secret_key

import socket
myname = socket.getfqdn(socket.gethostname())
myaddr = socket.gethostbyname(myname)