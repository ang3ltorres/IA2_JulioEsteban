# Autores
#  Angel Emmanuel Suarez Torres
#  Angel Damian Raul Garcia
# Notas:
# Ha sido necesario invertir la coordenada Y en distintas ocasiones

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QFrame, QLabel, QMessageBox
from PyQt6.QtGui import QPainter, QPen, QBrush, QMouseEvent
from PyQt6.QtCore import Qt, QPoint
from typing import List

WIDTH = 500
HEIGHT = 500

class Perceptron:
	def __init__(self):
		self.points: List[QPoint] = []  # Lista de puntos
		self.w1 = 0  # Peso 1
		self.w2 = 0  # Peso 2
		self.bias = 0  # Bias
		self.classified = False  # Indica si los puntos ya fueron clasificados

	def predict(self, x, y):
		y = HEIGHT - y  # Convertir la coordenada Y
		return 1 if (self.w1 * x + self.w2 * y + self.bias) >= 0 else -1

perceptron = Perceptron()


class PlotArea(QFrame):
	def __init__(self):
		super().__init__()
		self.setFixedSize(WIDTH, HEIGHT)
		self.setStyleSheet("border: 1px solid gray;")

	def paintEvent(self, event):
		global perceptron
		painter = QPainter(self)

		# Configurar pluma para ejes y referencias
		ref_pen = QPen(Qt.GlobalColor.gray, 1, Qt.PenStyle.DashLine)
		axis_pen = QPen(Qt.GlobalColor.black, 2)

		# Dibujar líneas de referencia verticales y horizontales
		painter.setPen(ref_pen)
		step = 100  # Espaciado de las líneas de referencia
		for i in range(0, WIDTH, step):
				painter.drawLine(i, 0, i, HEIGHT)  # Líneas verticales
				painter.drawLine(0, i, WIDTH, i)  # Líneas horizontales
				painter.drawText(i + 5, HEIGHT - 5, str(i))  # Coordenadas (x, y) para evitar corte
				painter.drawText(1, i - 5, str(HEIGHT - i))  # Coordenadas (x, y) para evitar corte

		# Dibujar ejes principales X e Y
		painter.setPen(axis_pen)
		painter.drawLine(0, HEIGHT, WIDTH, HEIGHT)  # Eje X
		painter.drawLine(0, 0, 0, HEIGHT)  # Eje Y

		# Dibuja los puntos
		for point in perceptron.points:
			x, y = point.x(), HEIGHT - point.y()

			if perceptron.classified:
				color = Qt.GlobalColor.green if perceptron.predict(x, y) == 1 else Qt.GlobalColor.red
			else:
				color = Qt.GlobalColor.black

			painter.setPen(Qt.PenStyle.NoPen)
			painter.setBrush(QBrush(color, Qt.BrushStyle.SolidPattern))
			painter.drawEllipse(point.x(), HEIGHT - point.y(), 8, 8)

		# Dibujar el hiperplano de decisión
		if perceptron.classified and perceptron.w2 != 0:
			m = -perceptron.w1 / perceptron.w2  # Pendiente
			c = -perceptron.bias / perceptron.w2  # Intersección

			# Puntos extremos para dibujar la linea
			x1, y1 = 0, int(m * 0 + c)
			x2, y2 = WIDTH, int(m * WIDTH + c)

			# Ajustar coordenadas para PyQt6
			y1 = HEIGHT - y1  
			y2 = HEIGHT - y2  

			# Dibujar
			pen = QPen(Qt.GlobalColor.blue, 2)
			painter.setPen(pen)
			painter.drawLine(x1, y1, x2, y2)

	# Capturar el click
	def mousePressEvent(self, event: QMouseEvent):
		global perceptron

		# Si es dentro del plot
		if self.rect().contains(event.pos()):

			# Agregarlo a la lista de puntos
			self.click_position = event.pos()
			perceptron.points.append(QPoint(event.pos().x(), HEIGHT - event.pos().y()))
			perceptron.classified = False
			self.update()

class MainWindow(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Practica 1")
		self.drawing_area = PlotArea()

		# Crear etiquetas y campos de entrada
		self.label1 = QLabel("Peso 1:", self)
		self.input1 = QLineEdit(self)
		self.input1.setPlaceholderText("Ingresar peso 1")

		self.label2 = QLabel("Peso 2:", self)
		self.input2 = QLineEdit(self)
		self.input2.setPlaceholderText("Ingresar peso 2")

		self.label3 = QLabel("Bias:", self)
		self.input3 = QLineEdit(self)
		self.input3.setPlaceholderText("Ingresar Bias")

		self.button = QPushButton("Clasificar", self)
		self.button.clicked.connect(self.classify)

		# Inputs + Labels
		input_layout = QVBoxLayout()
		input_layout.addWidget(self.label1)
		input_layout.addWidget(self.input1)
		input_layout.addWidget(self.label2)
		input_layout.addWidget(self.input2)
		input_layout.addWidget(self.label3)
		input_layout.addWidget(self.input3)
		input_layout.addWidget(self.button)
		input_layout.addStretch()

		# Layout
		main_layout = QHBoxLayout(self)
		main_layout.addWidget(self.drawing_area)
		main_layout.addLayout(input_layout)
		self.setLayout(main_layout)

	# Callback botón de clasificar
	def classify(self):
		global perceptron
		try:
			perceptron.w1 = float(self.input1.text())
			perceptron.w2 = float(self.input2.text())
			perceptron.bias = float(self.input3.text())
			perceptron.classified = True
			self.drawing_area.update()
		except ValueError:
			print("Por favor, ingrese valores numéricos para los pesos y bias.")

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
