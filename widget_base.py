import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

from_window = uic.loadUiType('')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)    #지금 만든 가상환경을 쓴다는거임.
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())

