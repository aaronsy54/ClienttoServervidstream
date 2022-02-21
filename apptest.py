from sre_constants import SUCCESS
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64,cv2
import numpy as np
import pyshine as ps
from flask_cors import CORS,cross_origin
import imutils
from engineio.payload import Payload
import torch
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
def get_yolov5():
    # local best.pt
    model = torch.hub.load('./yolov5', 'custom', path='E:/anaconda/envs/Demo/yolov5/models/best.pt', source = 'local')  # local repo
    model.conf = 0.85
    return model

model = get_yolov5()

def readb64(base64_string):
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    pimg = Image.open(sbuf)


    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

@socketio.on('image')
def image(data_image):
    frame = (readb64(data_image))
    imgencode = cv2.imencode('.jpeg', frame,[cv2.IMWRITE_JPEG_QUALITY,40])[1]    
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model.html')
def test():
    return render_template('model.html')

@app.route('/webcam.html',methods=['POST', 'GET'])
def webcam2():
    return render_template('webcam.html')

if __name__ == '__main__':
    socketio.run(app,port=9990 ,debug=True)
   
