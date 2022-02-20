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



Payload.max_decode_packets = 2048

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

def moving_average(x):
    return np.mean(x)

global fps,prev_recv_time,cnt,fps_array
fps=30
prev_recv_time = 0
cnt=0
fps_array=[0]
global frame
@socketio.on('image')
def image(data_image):
    global fps,cnt, prev_recv_time,fps_array
    recv_time = time.time()
    text  =  'FPS: '+str(fps)
    frame = (readb64(data_image))
    frame = ps.putBText(frame,text,text_offset_x=20,text_offset_y=30,vspace=20,hspace=10, font_scale=1.0,background_RGB=(10,20,222),text_RGB=(255,255,255))
    imgencode = cv2.imencode('.jpeg', frame,[cv2.IMWRITE_JPEG_QUALITY,40])[1]    
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)
    
    fps = 1/(recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)),1)
    prev_recv_time = recv_time
    #print(fps_array)
    cnt+=1
    if cnt==30:
        fps_array=[fps]
        cnt=0                




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
   
