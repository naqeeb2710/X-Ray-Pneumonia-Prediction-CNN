from flask import Flask, render_template, request
from keras.models import load_model
import keras
import tensorflow as tf
import os ,shutil

app = Flask(__name__)


model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
	i = tf.keras.preprocessing.image.load_img(img_path, target_size=(128,128))
	i = tf.keras.preprocessing.image.img_to_array(i)/255.0
	i = i.reshape(1, 128,128,3)
	p = model.predict(i)
	p=p.reshape(1)
	if p>=0.51:
		return 'You have Pneumonia and Need to consult a doctor'
	else:
		return 'You are completely Healthy'


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)
	remove_file(img_path)
	return render_template("index.html", prediction = p, img_path = img_path)

def remove_file(img_path):
	folder = 'static/'
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		print(file_path)
		try:
			if file_path!=img_path:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
				elif os.path.isdir(file_path):
					shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
	



if __name__ =='__main__':
	#app.debug = True
	app.run(debug=True)