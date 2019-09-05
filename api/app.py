import sys
import os
import uuid
import io
sys.path.append('cgan')
sys.path.append('api')
from flask import Flask, render_template, request, send_file
from api.commons import input_photo, load_photo, tensor_to_PIL, serve_pil_image, photo2painting, random_list_creator
from api.inference import get_painting_tensor
from pathlib import Path
from PIL import Image
import boto3

app = Flask(__name__)
app.config['RESULTS'] = 'api/static/esults/' 

S3 = boto3.client('s3')
BUCKET_NAME = 'photo2painting'
S3_LOCATION = 'http://{}.s3.amazonaws.com/'.format(BUCKET_NAME)

if __name__ == '__main__':
    app.run   
    
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/create/', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        ModelName = request.args.get('model')
        pics_list = random_list_creator(65, 6)
        print(len(pics_list))   

        return render_template('create.html', ModelName= ModelName, pics_list=pics_list)

    if request.method == 'POST':
                #get model and photo querys
                PhotoName = request.form['upload']
                ModelName = request.args.get('model')  

                #get input photo and resize it
                photo_path = os.path.join('api/static/images', PhotoName + '.jpg')
                photo = load_photo(photo_path) #load photo and scale it if necessary
                photo = input_photo(photo) #get tensor of photo (input of the model)

                #run inference on the photo
                painting = photo2painting(photo, ModelName) #load model from S3 and run inference
                photo = None
                painting = tensor_to_PIL(painting) #transform output tensor to PIL Image     
                #save painting output and update it to S3
                u_id = str(uuid.uuid4())
                save_path = os.path.join('api/static/esults/', u_id + '.jpg')
                painting.save(save_path)

                return render_template('result-download.html', key = u_id)    

@app.route('/result_download/', methods=['GET', 'POST'])
def result_download():
        if request.method == 'GET':
                return render_template('result-download.html')

@app.route('/download-files/', methods=['GET','POST'])
def download_files():
        if request.method == 'POST':
                key = request.form['download']
                key = str(key)
                img_path = os.path.join('api/static/esults/', key + '.jpg')
                img = Image.open(img_path)
                return serve_pil_image(img)

if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0')