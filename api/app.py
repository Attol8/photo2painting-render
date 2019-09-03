import sys
import os
import uuid
sys.path.append('cgan')
sys.path.append('api')
from flask import Flask, render_template, request, send_from_directory
from api.commons import input_photo, load_photo, tensor_to_PIL, upload_file_to_s3
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
    app.run(debug= False)   
    
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/create/', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        ModelName = request.args.get('model')      

        return render_template('create.html', ModelName= ModelName)

    if request.method == 'POST':
                #get model and photo querys
                PhotoName = request.form['upload']
                ModelName = request.args.get('model')  

                #get input photo and resize it
                photo_path = os.path.join('api/static/images', PhotoName + '.jpg')
                photo = load_photo(photo_path) #load photo and scale it if necessary
                photo = input_photo(photo) #get tensor of photo (input of the model)

                #run inference on the photo
                painting_tensor = get_painting_tensor(photo, ModelName).cpu() #load model from S3 and run inference
                painting_image = tensor_to_PIL(painting_tensor) #transform output tensor to PIL Image

                #save painting output and update it to S3
                u_id = str(uuid.uuid4())
                save_path = 'api/static/esults/' + u_id +'.jpg'
                painting_image.save(save_path)

                return render_template('result-download.html', key = u_id)    

@app.route('/result_download/', methods=['GET', 'POST'])
def result_download():
        if request.method == 'GET':
                return render_template('result-download.html')

@app.route('/download-files/', methods=['GET','POST'])
def download_files():
        try:
                key = request.form['download']
                key = str(key)
                print(key)
                return send_from_directory(app.config['RESULTS'], filename=key+'.jpg', as_attachment=True)
        except Exception as e:
                return str(e)

if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0')