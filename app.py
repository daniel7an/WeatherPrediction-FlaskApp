import os 
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from get_prediction import predict

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            print('file is not uploaded!')
        file = request.files['file']
        filename = secure_filename(file.filename)
        img_path = os.path.join('files', filename)
        file.save(img_path)
        prediction = predict(img_path)
        
        return prediction
if __name__ == '__main__':
    app.run(debug=True)