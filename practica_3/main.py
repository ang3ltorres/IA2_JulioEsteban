import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Definición de la función original
def funcion(x: float) -> float:
	return ((x - 4.3)**4) / 110.5 + (-0.6 * (x - 4.3)**3) / 12.8 - 0.2 * (x - 4.3)**2 + 6.8

# Generación de datos con ruido
def evaluacionRuido(n: float, noise: float) -> tuple:
	x_vals = np.arange(0, 10, n)
	y_vals = [funcion(x) for x in x_vals]
	for i in range(len(y_vals)):
		y_vals[i] += random.uniform(-noise, noise)
	return x_vals, y_vals

# Implementación del Adaline
class Adaline:
	def __init__(self, tasa_aprendizaje=0.1, epocas=1):
		self.tasa_aprendizaje = tasa_aprendizaje
		self.epocas = epocas
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.epocas):
			y_pred = self.predict(X)
			error = y - y_pred
			self.weights += self.tasa_aprendizaje * X.T.dot(error)
			self.bias += self.tasa_aprendizaje * error.sum()

	def predict(self, X):
		return X.dot(self.weights) + self.bias

# Clase para el gráfico
class PlotCanvas(FigureCanvas):
	def __init__(self):
		self.fig, self.ax = plt.subplots()
		super().__init__(self.fig)
		self.setFixedSize(500, 500)
		self.WIDTH = 10
		self.HEIGHT = 10
		self.ax.set_xlim(0, self.WIDTH)
		self.ax.set_ylim(0, self.HEIGHT)

	def plot(self, x_vals, y_vals, y_pred):
		self.ax.clear()
		self.ax.set_xlim(0, self.WIDTH)  # Limite x en 10
		self.ax.set_ylim(0, self.HEIGHT)  # Limite y en 10
		self.ax.scatter(x_vals, y_vals, color='b', label='Datos con ruido')
		self.ax.plot(x_vals, [funcion(x) for x in x_vals], color='g', label='Función original')
		self.ax.plot(x_vals, y_pred, color='r', label='Adaline entrenado')
		self.ax.set_title("Filtro Adaptativo con Adaline")
		self.ax.set_xlabel("x")
		self.ax.set_ylabel("f(x)")
		self.ax.grid(True)
		self.ax.legend()
		self.draw()

# Ventana principal con interfaz de usuario
class App(QWidget):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		layout = QVBoxLayout()

		self.paso_input = QLineEdit("0.2")
		self.ruido_input = QLineEdit("0.8")
		self.tasa_input = QLineEdit("0.001")
		self.epocas_input = QLineEdit("1000")

		layout.addWidget(QLabel("Tamaño paso:"))
		layout.addWidget(self.paso_input)
		layout.addWidget(QLabel("Ruido:"))
		layout.addWidget(self.ruido_input)
		layout.addWidget(QLabel("Tasa de aprendizaje:"))
		layout.addWidget(self.tasa_input)
		layout.addWidget(QLabel("Épocas:"))
		layout.addWidget(self.epocas_input)

		self.plot_canvas = PlotCanvas()
		layout.addWidget(self.plot_canvas)

		self.button = QPushButton("Generar Gráfico")
		self.button.clicked.connect(self.generar_grafico)
		layout.addWidget(self.button)

		self.setLayout(layout)
		self.setWindowTitle("Practica 3")

	def generar_grafico(self):
		try:
			n = float(self.paso_input.text())
			ruido = float(self.ruido_input.text())
			tasa_aprendizaje = float(self.tasa_input.text())
			epocas = int(self.epocas_input.text())

			x_vals, y_vals = evaluacionRuido(n, ruido)
			X = x_vals.reshape(-1, 1)
			y = np.array(y_vals)

			adaline = Adaline(tasa_aprendizaje, epocas)
			adaline.fit(X, y)
			y_pred = adaline.predict(X)

			self.plot_canvas.plot(x_vals, y_vals, y_pred)
		except ValueError:
			print("Error en los datos ingresados!!")

# Ejecutar la aplicación
if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = App()
	ex.show()
	sys.exit(app.exec())
