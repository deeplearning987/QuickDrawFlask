from tensorflow.keras.models import load_model
import numpy as np
from random import choice
from PIL import Image
import base64
import io

class QuickDraw:
	def __init__(self):
		self.model = model_try = load_model('keras.h5')
		f = open("categories.txt","r")
		self.classes = f.read().split('\n')[:-1]
		f.close()

	def puzzle(self):
		return choice(self.classes)
	
	def check(self, image, category):
		image = base64.b64decode(image)
		image = Image.open(io.BytesIO(image)).convert('L').resize((28, 28))
		image = np.array(image).reshape(28,28,1).astype('float32')/255.0
		prediction = self.model.predict(np.expand_dims(image, axis=0))[0]
		ind = (-prediction).argsort()[:5]
		result = [ self.classes[x] for x in ind]
		try:
			return result.index(category)+1
		except ValueError:
			return 0