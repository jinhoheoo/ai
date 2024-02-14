#GUI 만들 때 main window로 만들면 윈도우에서 실행가능한 형태로 만들 수 있음.window로 만들면 위에 메뉴창(menubar) 그냥 기본설정처럼 있음.
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_window = uic.loadUiType('./Qnotepad.ui')[0]    #ui파일 저장할 때 exam17있는데다가 저장해야 불러와짐.
#윈도우나 위젯의 배치를 설정해줘야 내가 원하는대로 배치할 수 있음 즉 수직으로 정렬 수평으로 정렬 등등 배치 규칙이 있어야 여러 형태를 편하게 배치가능함.
#Text edit 보다 plain text edit가 더 다양한 기능이 있음.
#combo박스는 줄 선택가능한 보기창 만드는거 그 눌렀을 때 여러 항목나오는거임.
#프론트엔드가 Qt desinger에서 창만들고 메뉴만들고 이런거고
#백엔드가 여기에서 뭐 눌렀을 때 어떤식으로 동작하고 어떻게 연관되고 이런거임
#mainwindow에서 도구모음 추가해서 여기 파일에 icon과 같은 png파일 넣고 open의 icon에 png파일 넣고해서 동작편집기에서 사진 드래그해서 넣으면 바로가기 사진 만들어짐
class Exam(QMainWindow, form_window): #위젯일때는 QWidget 메인 윈도우 일때는 QMainWindow로 해야함.
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path = ('제목없음','') #첫번째요소에 파일의 경로가 들어가고 두번째요소에 확장자가 들어감. 경로가 제목임.
        self.edited_flag = False  # flag을 쓰면 시스템 수정할 때 편리함.
        self.setWindowTitle('*' + self.path[0].split('/')[-1] + '- QT Note Pad')

        # file menu 구현
        self.actionSave_as.triggered.connect(self.action_save_as_slot)      #클릭을 했을 때 컨넥트함.
        #trigger는 방아쇠로 메뉴를 땡겼다 save as 눌렀을 때 함수에 적은 동작 실행시키는거임.
        #  "Save As" 메뉴 옵션을 나타내는 객체(액션)입니다.
        # triggered 시그널은 사용자가 해당 액션을 실행했을 때 발생합니다.
        # connect 메서드를 사용하여 triggered 시그널이 발생할 때
        self.actionSave.triggered.connect(self.action_save_slot)
        self.actionExit.triggered.connect(self.action_exit_slot)
        self.plain_te.textChanged.connect(self.text_changed_slot)
        self.actionOpen.triggered.connect(self.action_open_slot)
        self.actionNew.triggered.connect(self.action_new_slot)

        #edit menu 구현
        #qt designer에서 ctrl z와 같은 키 넣는건 shortcut 항목 누르고 키보드로 키 넣으면 됨.
        self.actionUn_do.triggered.connect(self.plain_te.undo)
        self.actionCut.triggered.connect(self.plain_te.cut)
        self.actionCopy.triggered.connect(self.plain_te.copy)
        self.actionPaste.triggered.connect(self.plain_te.paste)
        self.actionDelete.triggered.connect(self.plain_te.cut) #delete 기능이 없어서 cut()을 사용함
        self.actionSelect_all.triggered.connect(self.plain_te.selectAll)

       # self.actionFont.triggered.connect(self.action_font_slot)

    def save_edited(self):
        if self.edited_flag:  # flag을 쓰면 시스템 수정할 때 편리함
            ans = QMessageBox.question(self, '저장하기', '저장할까요?', QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes,
                                       QMessageBox.Yes)
            # self: 부모 위젯으로, 메시지 박스가 속할 부모 위젯을 지정합니다.
            # '저장하기': 메시지 박스의 제목으로 사용될 문자열입니다
            # '저장할까요?': 사용자에게 물을 질문이나 표시될 메시지 문자열입니다.
            # QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes: 버튼 종류를 지정하는 플래그로, 이 코드에서는 No, Cancel, Yes 버튼을 사용하도록 설정되어 있습니다.
            # QMessageBox.Yes: 기본적으로 선택된 버튼으로, 사용자가 Enter 키를 누르거나 다른 방법으로 선택하지 않은 경우의 기본 동작을 나타냅니다.
            # ans 변수에는 사용자가 선택한 버튼에 대한 정보가 들어갑니다. QMessageBox.No를 선택한 경우: ans에는 QMessageBox.No에 해당하는 값이 들어갑니다.
            # yes경우 16384가 리턴됨.

            if ans == QMessageBox.Yes:  # 파일 저장
                if self.action_save_slot():
                    #  print('debug01') 문자 사이사이에 디버그를 넣어서 코드가 어디까지 죽었는 지 알아보기 위함.
                    return
            elif ans == QMessageBox.Cancel:
                return 1

    def action_save_as_slot(self):
        old_path = self.path    #현재 경로를 old_path로 넣고 아래에 새로운 경로를 받음.
        self.path = QFileDialog.getSaveFileName(self, 'Save file', '', 'Text Files(*.txt);;Python Files(*,py);;All Files(*,*)')
        # path는 경로를 리턴해줌  첫번째는 self: 부모 위젯으로, 대화 상자를 호출하는 위젯입니다. 두번째는 대화상자의 이름,
        # 세번째인'': 대화 상자를 열었을 때 초기 디렉토리로 사용할 경로입니다. 여기서는 비어 있으므로 현재 작업 디렉토리가 사용될 것입니다.
        # 네번째는 파일형식을 고를수 있게함 확장자를 고를 수 있음.
        #self.path[0]은 선택한 파일의 경로를 나타냅니다. 코드에서는 파일 경로를 self.path에 저장하고 있습니다. 그냥 취소하면 0에 아무것도 안들어가게됨.
        print(self.path)
        if self.path[0]:
            with open(self.path[0],'w')as f:        #f라고 이름짓고 파일 오픈하는거임 with를 하면 따로 close 안해도 됨
                f.write(self.plain_te.toPlainText()) #toPlainText()로 해당 위젯에 입력된 텍스트를 가져올 수 있습니다. 그래서 그걸가지고 저장하는거임.
                self.edited_flag = False
                self.plain_te.textChanged_slot.connect(self.text_changed_slot)
                self.setWindowTitle(self.path[0].split('/')[-1] + '- QT Note Pad') # 문자열이니까 더하기
                #self.setWindowTitle(...): 이 메소드는 PyQt의 QWidget 클래스의 메소드로서, 창의 제목을 설정합니다.
                #self.path[0]: self.path 튜플의 첫 번째 요소로서, 선택된 파일의 경로를 나타냅니다.
                #.split('/'): 파일 경로를 디렉토리와 파일 이름으로 나누기 위해 사용하는 문자열 메소드입니다. /를 기준으로 나눕니다.
                #[-1]: 리스트에서 마지막 요소를 선택합니다. 여기서는 파일 이름이 마지막 요소가 됩니다.
                #+ '- QT Note Pad': 파일 이름 뒤에 문자열 "- QT Note Pad"를 추가하여 창의 새로운 제목을 만듭니다.
            return 0        #잘 저장되면 문제없는 걸 의미하는 0을 반환함.
        else:
            self.path = old_path        #경로가 없다면(path에 아무것도없다면) old_path를 현재 path로 함.
            return 1

        # 파일을 열고 저장읋 하려다가 수정 할 게 생각나서 저장하려다가 취소를 눌렀어. 그때 종료되지 않고, 창이 그대로 떠 있게 하기 위해서 return을 사용함.

    def action_save_slot(self):
        if self.path[0] != '제목 없음':
            with open(self.path[0], 'w') as f:
                f.write(self.plain_te.toPlainText())
                self.edited_flag = False
                self.plain_te.textChanged_slot.connect(self.text_changed_slot)
                self.setWindowTitle(self.path[0].split('/')[-1] + '- QT Note Pad')

        else:
            return self.action_save_as_slot()  # '제목 없음' 일 경우 저장됨.

    def action_exit_slot(self):    #문서 수정이 있으면 save로 해서 저장하면되고 없으면 바로 끄면됨.
        if self.save_edited():
            return
        # print('debug02')
        if self.edited_flag:
            return
        self.close()

    def text_changed_slot(self):
        print('change')  # text를 입력할 때마다 출력됨.
        self.edited_flag = True
        self.setWindowTitle('*' + self.path[0].split('/')[-1] + '- QT Note Pad')
        self.plain_te.textChanged.disconnect(self.text_changed_slot)

    def action_open_slot(self):
        if self.save_edited():
            return
        if self.edited_flag:
            return
        old_path = self.path
        self.path = QFileDialog.getSaveFileName(self, 'OPEN file', '', 'Text Files(*.txt);;Python Files(*,py);;All Files(*,*)')
        # path는 경로를 리턴해줌  첫번째는 self: 부모 위젯으로, 대화 상자를 호출하는 위젯입니다. 두번째는 대화상자의 이름,
        # 세번째인'': 대화 상자를 열었을 때 초기 디렉토리로 사용할 경로입니다. 여기서는 비어 있으므로 현재 작업 디렉토리가 사용될 것입니다.
        # 네번째는 파일형식을 고를수 있게함 확장자를 고를 수 있음.
        # self.path[0]은 선택한 파일의 경로를 나타냅니다. 코드에서는 파일 경로를 self.path에 저장하고 있습니다. 그냥 취소하면 0에 아무것도 안들어가게됨.
        if self.path[0]:
            with open(self.path[0], 'r') as f:
                str_read = f.read()
            self.plain_te.setPlainText(str_read)
            self.edited_flag = False
            self.setWindowTitle(self.path[0].split('/')[-1] + '- QT Note Pad')
        else:
            self.path = old_path


    def action_new_slot(self): # 제목없음이랑 같은 상태입니다.
        if self.save_edited():
            return
        self.plain_te.setPlainText('')
        self.edited_flag =False
        self.path = ('제목 없음','')
        self.setWindowTitle(self.path[0].split('/')[-1] + '- QT Note Pad')

    def action_font_slot(self):
        font = QFontDialog.getFont()    #윈도우에서 제공하는 폰트들 불러와줌
        print(font)
        if font[1]:
            self.plain_te.setFont(font[0])
        #QFontDialog.getFont(): 폰트 선택 대화 상자를 열어 사용자에게 폰트를 선택하도록 합니다. 반환값은 튜플로, 첫 번째 요소는 선택한 폰트 객체(QFont)이고, 두 번째 요소는 사용자가 "OK" 버튼을 눌렀는지 여부를 나타내는 불리언 값입니다.
        #print(font): 선택한 폰트와 사용자가 "OK" 버튼을 눌렀는지 여부를 콘솔에 출력합니다. 이는 디버깅을 위한 용도로 사용될 수 있습니다.
        #if font[1]:: 사용자가 "OK" 버튼을 눌렀을 때만 아래의 코드 블록이 실행됩니다.
        #self.plain_te.setFont(font[0]): 선택한 폰트(font[0])를 QPlainTextEdit 위젯에 적용합니다.

    def action_about_slot(self):
        QMessageBox.about(self, 'Qt Note Pad', '만든이:abc label\n\r 버전 정보:1.0.0')


if __name__ == '__main__':
    app = QApplication(sys.argv)    #지금 만든 가상환경을 쓴다는거임.
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())

