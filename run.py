import os, cv2
import base64
import inference_gfpgan_new
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/upload'
app.config['RESULT_FOLDER'] = 'static/result/restored_imgs'


@app.route('/upload', methods = ['GET', 'POST'])
def upload():

    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        height, width, channels = img.shape
        print(height, width)

        inference_gfpgan_new.main()
        out_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)

        img_1 = cv2.imread(out_path)
        out_img = cv2.resize(img_1, (width, height))
        cv2.imwrite(out_path, out_img)

        _, buffer = cv2.imencode('.jpg', img_1)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the base64 encoded image as JSON
        return jsonify({'image': img_base64})


if __name__ == '__main__':
    app.run()