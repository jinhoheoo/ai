import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

fomm_window = uic.loadUiType('./poster.ui')[0]

class Exam(QWidget, fomm_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_myunglang.clicked.connect(self.btn_click_slot)
        self.btn_hansan.clicked.connect(self.btn_click_slot)
        self.btn_nolang.clicked.connect(self.btn_click_slot)
        self.btn_seoul.clicked.connect(self.btn_click_slot)

    def btn_click_slot(self):
        btn = self.sender()
        self.lbl_myunglang.hide()
        self.lbl_hansan.hide()
        self.lbl_nolang.hide()
        self.lbl_seoul.hide()
        if btn.objectName() == 'btn_myunglang': self.lbl_myunglang.show()
        elif btn.objectName() == 'btn_hansan': self.lbl_hansan.show()
        elif btn.objectName() == 'btn_nolang': self.lbl_nolang.show()
        elif btn.objectName() == 'btn_seoul': self.lbl_seoul.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)  # QApplication 객체 생성
    #sys.argv 프로그램이 실행될 때 사용자가 추가 정보를 전달할 수 있게 해주는데, 이 정보를 애플리케이션에서 활용할 수 있어요.
    #QApplication은 PyQt에서 GUI 애플리케이션을 시작하기 위한 주요 클래스 중 하나입니다.
    # 이 클래스는 PyQt에서 제공하는 Qt 라이브러리의 기능을 초기화하고 애플리케이션의 이벤트 루프를 관리합니다.
    # GUI 애플리케이션을 만들 때 필수적으로 사용되며, 다양한 설정 및 기능을 제공합니다.
    mainWindow = Exam()  # Exam 클래스의 객체 생성
    mainWindow.show()  # UI를 화면에 표시
    sys.exit(app.exec_())  # 애플리케이션 실행 및 이벤트 루프 시작