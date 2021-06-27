from flask import request, Flask, render_template, jsonify
from quickdraw import QuickDraw

app = Flask(__name__)
quickdraw = QuickDraw()

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/puzzle')
def puzzle():
    return quickdraw.puzzle()

@app.route('/check', methods=['POST'])
def check():
    data = request.get_json()
    return jsonify({ 'result': quickdraw.check(data['image'], data['category']) })

if __name__ == '__main__':
	app.run(debug=True)