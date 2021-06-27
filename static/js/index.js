var lastX, lastY, mousePressed = false
var canvas = document.getElementById('canvas')
var ctx = canvas.getContext("2d")

document.getElementById('canvas').addEventListener('mousedown', function (e) {
	mousePressed = true
	Draw(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, false)
})

document.getElementById('canvas').addEventListener('mousemove', function (e) {
	if (mousePressed) {
		Draw(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true)
	}
})

document.getElementById('canvas').addEventListener('mouseup', function (e) {
	mousePressed = false
})
	document.getElementById('canvas').addEventListener('mouseleave', function (e) {
	mousePressed = false
})

function Draw(x, y, isDown) {
	if (isDown) {
		ctx.beginPath()
		ctx.strokeStyle = 'white'
		ctx.lineWidth = 5
		ctx.lineJoin = "round"
		ctx.moveTo(lastX, lastY)
		ctx.lineTo(x, y)
		ctx.closePath()
		ctx.stroke()
	}
	lastX = x
	lastY = y
}
	
function clearArea() {
	// Use the identity matrix while clearing the canvas
	ctx.setTransform(1, 0, 0, 1, 0, 0)
	ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
}

function submit(mode) {
	new Promise((resolve, reject) => {
		if(mode) {
			fetch('/check', {
				method: 'POST',
				headers: {
					'Accept': 'application/json',
					'Content-type': 'application/json',
				},
				body: JSON.stringify({
					image: canvas.toDataURL().replace(/^data:image\/[a-z]+;base64,/, ""),
					category: document.getElementById('category').innerHTML,
				})
			}).then(res => res.json())
			.then(response => {
				const result = [
					'<span class="text-danger">Failed</span>',
					'<span class="text-success">Perfect!</span>',
					'<span class="text-success">Excellent</span>',
					'<span class="text-warning">Very Good</span>',
					'<span class="text-warning">Good</span>',
					'<span class="text-warning">Average</span>'
				][response['result']]
				document.getElementById('result').innerHTML = result
				resolve()
			})
		}
		else {
			puzzle()
			resolve()
		}
	}).then(e => {
			document.querySelectorAll('.switch').forEach(elem => {
			elem.classList.toggle('d-none')
		})
	})
}

function puzzle() {
				clearArea()
				fetch('/puzzle').then(res => res.text())
		.then(response => document.getElementById('category').innerHTML = response)
}

puzzle()