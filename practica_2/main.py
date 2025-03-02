from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QMessageBox
from PyQt6.QtCore import Qt, QPointF, QTimer
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import random
import numpy as np

class Perceptron:
	def __init__(self):
		self.pointsRed: List[QPointF] = []
		self.pointsBlue: List[QPointF] = []
		self.bias = random.uniform(-1.0, 1.0)
		self.w1 = random.uniform(-1.0, 1.0)
		self.w2 = random.uniform(-1.0, 1.0)
		self.tasaAprendizaje = 0.0
		self.epocas = 0.0
		self.previous_decisions = []  # Lista para almacenar las lineas de decision previas

	def predict(self, x, y):
		return 1 if (self.w1 * x + self.w2 * y + self.bias) >= 0 else -1

	def train(self):
		for point in self.pointsBlue:
			x, y = point.x(), point.y()
			prediction = self.predict(x, y)
			error = 1 - prediction
			self.w1 += self.tasaAprendizaje * error * x
			self.w2 += self.tasaAprendizaje * error * y
			self.bias += self.tasaAprendizaje * error

		for point in self.pointsRed:
			x, y = point.x(), point.y()
			prediction = self.predict(x, y)
			error = -1 - prediction
			self.w1 += self.tasaAprendizaje * error * x
			self.w2 += self.tasaAprendizaje * error * y
			self.bias += self.tasaAprendizaje * error

		# Guardar la linea de decision despues de cada paso de entrenamiento
		if self.w2 != 0:
			x_vals = np.array([0, 10])  # Limites en el grafico (0, 10)
			y_vals = (-self.w1 * x_vals - self.bias) / self.w2
			self.previous_decisions.append((x_vals, y_vals))  # Guardar la linea
		
perceptron = Perceptron()

class PlotCanvas(FigureCanvas):
	def __init__(self):
		self.fig, self.ax = plt.subplots()
		super().__init__(self.fig)
		self.setFixedSize(500, 500)  # Esto controla el tamaño de la ventana
		self.WIDTH = 10  # Definir el ancho del grafico en unidades
		self.HEIGHT = 10  # Definir la altura del grafico en unidades
		self.ax.set_xlim(0, self.WIDTH)  # Establecer limite x en 10
		self.ax.set_ylim(0, self.HEIGHT)  # Establecer limite y en 10

	def plot_points(self):
		global perceptron

		self.ax.clear()
		self.ax.set_xlim(0, self.WIDTH)  # Limite x en 10
		self.ax.set_ylim(0, self.HEIGHT)  # Limite y en 10

		# Dibujar todas las lineas de decision anteriores
		for i, (x_vals, y_vals) in enumerate(perceptron.previous_decisions):

			# La ultima linea de decision de color magenta
			if i == len(perceptron.previous_decisions) - 1:
				self.ax.plot(x_vals, self.HEIGHT - y_vals, color="magenta", linestyle="--", alpha=1.0)
			else:
				self.ax.plot(x_vals, self.HEIGHT - y_vals, color="green", linestyle="--", alpha=0.3)

		# Dibujar puntos rojos
		for point in perceptron.pointsRed:
			x, y = point.x(), point.y()
			self.ax.scatter(x, self.HEIGHT - y, color="red")

		# Dibujar puntos azules
		for point in perceptron.pointsBlue:
			x, y = point.x(), point.y()
			self.ax.scatter(x, self.HEIGHT - y, color="blue")

		self.draw()

	def mousePressEvent(self, event):
		global perceptron

		if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
			# Obtener las coordenadas del mouse en pixeles
			x_pixel, y_pixel = event.position().x(), event.position().y()

			# Convertir las coordenadas del clic en pixeles a coordenadas en unidades del grafico
			x_data = self.ax.transData.inverted().transform((x_pixel, y_pixel))[0]
			y_data = self.ax.transData.inverted().transform((x_pixel, y_pixel))[1]

			# Ajustar las coordenadas para que esten dentro de los limites del grafico
			x = np.clip(x_data, 0, self.WIDTH)
			y = np.clip(y_data, 0, self.HEIGHT)

			# Añadir el punto al perceptron
			if event.button() == Qt.MouseButton.LeftButton:
				perceptron.pointsBlue.append(QPointF(x, y))

			if event.button() == Qt.MouseButton.RightButton:
				perceptron.pointsRed.append(QPointF(x, y))

			self.plot_points()  # Actualizar el grafico

class MainWindow(QWidget):
	def __init__(self):
		global perceptron

		super().__init__()
		self.setWindowTitle("Practica 2")
		self.plot_canvas = PlotCanvas()

		# Boton reinicio
		self.buttonReset = QPushButton("Reiniciar")
		self.buttonReset.clicked.connect(self.reset)

		# Campos de entrada
		self.labelBias = QLabel(f"Bias: {perceptron.bias:.4f}")
		self.labelW1 = QLabel(f"W1: {perceptron.w1:.4f}")
		self.labelW2 = QLabel(f"W2: {perceptron.w2:.4f}")

		self.labelEpocas = QLabel("Epocas:")
		self.inputEpocas = QLineEdit()
		self.inputEpocas.setPlaceholderText("Ingresar no. epocas")

		self.labelTasa = QLabel("Tasa aprendizaje:")
		self.inputTasa = QLineEdit()
		self.inputTasa.setPlaceholderText("Ingresar η")

		self.buttonClassify = QPushButton("Entrenar")
		self.buttonClassify.clicked.connect(self.start_training)

		input_layout = QVBoxLayout()
		input_layout.addWidget(self.buttonReset)
		input_layout.addWidget(self.labelBias)
		input_layout.addWidget(self.labelW1)
		input_layout.addWidget(self.labelW2)
		input_layout.addWidget(self.labelEpocas)
		input_layout.addWidget(self.inputEpocas)
		input_layout.addWidget(self.labelTasa)
		input_layout.addWidget(self.inputTasa)
		input_layout.addWidget(self.buttonClassify)
		input_layout.addStretch()

		main_layout = QHBoxLayout(self)
		main_layout.addWidget(self.plot_canvas)
		main_layout.addLayout(input_layout)
		self.setLayout(main_layout)

		# Configuracion del temporizador para actualizacion continua
		self.timer = QTimer(self)
		self.timer.timeout.connect(self.train_step)
		self.timer.setInterval(200)  # Actualiza cada 500 ms

	def reset(self):
		global perceptron

		self.timer.stop()
		perceptron.bias = random.uniform(-1.0, 1.0)
		perceptron.w1 = random.uniform(-1.0, 1.0)
		perceptron.w2 = random.uniform(-1.0, 1.0)
		perceptron.pointsBlue = []
		perceptron.pointsRed = []
		perceptron.previous_decisions = []

		self.labelBias.setText(f"Bias: {perceptron.bias:.4f}")
		self.labelW1.setText(f"W1: {perceptron.w1:.4f}")
		self.labelW2.setText(f"W2: {perceptron.w2:.4f}")

		self.plot_canvas.plot_points()

	def start_training(self):
		global perceptron

		try:
			self.timer.stop()
			perceptron.epocas = int(self.inputEpocas.text())
			perceptron.tasaAprendizaje = float(self.inputTasa.text())
			self.timer.start()  # Comienza el entrenamiento en tiempo real
		except ValueError:
			QMessageBox.critical(self, "Error", "Ingrese valores numericos validos.")

	def train_step(self):
		global perceptron

		# Actualizar labels
		self.labelBias.setText(f"Bias: {perceptron.bias:.4f}")
		self.labelW1.setText(f"W1: {perceptron.w1:.4f}")
		self.labelW2.setText(f"W2: {perceptron.w2:.4f}")
		
		if perceptron.epocas > 0:
			perceptron.train()  # Entrena el perceptron
			perceptron.epocas -= 1
			self.plot_canvas.plot_points()  # Actualiza la grafica
		else:
			self.timer.stop()  # Detiene el entrenamiento cuando se acaban las epocas

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
