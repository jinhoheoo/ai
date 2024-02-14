import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

#import sys: sys 모듈을 가져오며, 이 모듈은 Python 인터프리터에서 사용되는 몇 가지 변수에 액세스하는 데 사용됩니다.
#from PyQt5.QtWidgets import *: PyQt5.QtWidgets 모듈에서 모든 클래스를 가져옵니다.
# 이 모듈은 응용 프로그램의 GUI 구성 요소를 포함하고 있습니다.
#from PyQt5 import uic: PyQt5에서 'uic' 모듈을 가져오며, 이 모듈은 Qt Designer로 작성된 UI 파일을 로드하는 데 사용됩니다.

from_window = uic.loadUiType('./cat_and_dog.ui')[0]
#uic는 ui를 클래스로 바꿔줌
#cat_and_dog.ui' UI 파일을 로드하고 튜플을 반환합니다. [0]은 튜플에서 클래스를 가져오는 데 사용됩니다.

class Exam(QWidget, from_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path = ('','')
        self.model = load_model('./cat_and_dog_0.818.h5')
        self.btn_open.clicked.connect(self.btn_clicked_slot)

    def btn_clicked_slot(self):
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(self,'Open file', '../datasets/cat_dog',
                                                'Image File(*.jpg;*.png);;All Files(*.*)')
        #QFileDialog 이거 운영체제가 제공해주는거임. 첫번째 파일은 파일 찾기 이름이 open file이라고 지어준거임.
        # 두번째인자는 처음에 열리는 파일 경로임 세번째는 열수 있는 파일 확장자들 적어준거임 ;;으로 고를수 있게 해준거임.
        #그렇게해서 파일 열면 내가 연 파일의 경로를 리턴해줌.
        print(self.path)
        if self.path[0]== '':
            self.path = old_path

        pixmap = QPixmap(self.path[0])
        self.lbl_image.setPixmap(pixmap)
        img = Image.open(self.path[0])
        img = img.convert('RGB')
        img = img.resize((64, 64))
        data = np.asarray(img)
        data = data / 255
        data = data.reshape(1, 64, 64, 3)
        pred = self.model.predict(data)
        print(pred)

        if pred < 0.5:
            self.lbl_result.setText('고양이 입니다.')
        else :
            self.lbl_result.setText('강아지 입니다.')

#class Exam(QWidget, from_window):: 'Exam'이라는 클래스를 정의하며, QWidget(모든 PyQt UI 객체의 기본 클래스) 및
# 로드된 UI 파일에서 얻은 클래스에서 상속받습니다.
# def __init__(self):: Exam 클래스를 초기화합니다.
#super().__init__(): 기본 클래스(QWidget)의 생성자를 호출하여 Exam 객체를 초기화합니다.
#self.setupUi(self): 로드된 UI 파일에서 'setupUi' 메서드를 호출하여 사용자 인터페이스를 설정합니다.

if __name__ == '__main__':
    app = QApplication(sys.argv)  # QApplication 객체 생성
    #sys.argv 프로그램이 실행될 때 사용자가 추가 정보를 전달할 수 있게 해주는데, 이 정보를 애플리케이션에서 활용할 수 있어요.
    #QApplication은 PyQt에서 GUI 애플리케이션을 시작하기 위한 주요 클래스 중 하나입니다.
    # 이 클래스는 PyQt에서 제공하는 Qt 라이브러리의 기능을 초기화하고 애플리케이션의 이벤트 루프를 관리합니다.
    # GUI 애플리케이션을 만들 때 필수적으로 사용되며, 다양한 설정 및 기능을 제공합니다.
    mainWindow = Exam()  # Exam 클래스의 객체 생성
    mainWindow.show()  # UI를 화면에 표시
    sys.exit(app.exec_())  # 애플리케이션 실행 및 이벤트 루프 시작
    #app.exec_() 사용자가 프로그램과 상호작용할 때 발생하는 이벤트를 감지하고 처리하기 위해 필요해요. 버튼 클릭, 키보드 입력 등이 여기에 해당돼요.
    #애플리케이션이 이벤트 루프를 돌며 동작하다가 사용자가 종료하면 프로그램이 정상적으로 종료됩니다.



    # 아나콘다로 가상환경 만들어 줄수 있음 각각의 가상환경마다
    # 예를들어 판다스와 같은 것들의 버전 각각 다르게 설정가능함.
    # 아나콘다에서 test라는 가상환경을  create 해서 거기서 numpy패키지 다운로드하고
    # 파이썬으로 돌아와서 젤 아래쪽에 python 적힌데 들어가서 add new interpreter에서
    # 아나콘다 거기들어가서 내가 만든 test 적용한 후 터미널 껏다가 다시 켜서
    # conda install pandas 이런식으로 다운로드 하면서 환경설정하면 됨.
    # 그냥 파이썬에서는 pip로 해서 다운로드 했었음
    # 우리는 아나콘다의 designer.exe를 키면 qt designer이 켜짐 이걸로 gui 만들거임.