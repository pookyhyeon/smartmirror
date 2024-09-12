from flask import Flask, render_template, request, send_file
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # HTML 파일로 기본 페이지 표시

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # photo_collage.jpg 파일 경로
        file_path = 'static/photo_collage.jpg'
        return send_file(file_path, mimetype='image/jpg')
    return "No image to display."

if __name__ == "__main__":
    app.run(debug=True)
