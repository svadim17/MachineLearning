from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from settings import SettingsWidget
from lab1_widget import Lab1Widget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.resize(1024, 600)
        self.init_header_bar()
        self.settings = SettingsWidget()
        self.create_action()
        self.create_toolbar()
        self.init_lab1Widget()
        # self.init_lab2Widget()
        # self.init_lab3Widget()
        self.connect_signals()

    def init_header_bar(self):
        self.setWindowTitle('Application')

    def create_action(self):
        self.act_exit = QAction('&Exit')
        self.act_exit.triggered.connect(self.close)

        self.act_settings = QAction('Settings')
        self.act_settings.setText('Settings')
        self.act_settings.setCheckable(True)
        self.act_settings.triggered.connect(self.settings.show)

    def create_toolbar(self):
        self.toolbar = QToolBar('Main')

        self.btn_main = QToolButton()
        self.btn_main.setText('Main')
        self.btn_main.setToolTip('Main')
        self.btn_main.setPopupMode(QToolButton.InstantPopup)
        self.btn_main.addAction(self.act_exit)

        self.toolbar.addWidget(self.btn_main)
        self.toolbar.addAction(self.act_settings)

        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)

    def init_lab1Widget(self):
        self.lab1Widget = Lab1Widget()
        self.addDockWidget(Qt.RightDockWidgetArea, self.lab1Widget)

    #
    # def init_lab2Widget(self):
    #     self.lab3Widget = Lab3Widget()
    #     self.addDockWidget(Qt.RightDockWidgetArea, self.lab3Widget)
    #     self.tabifyDockWidget(self.lab2Widget, self.lab3Widget)
    #
    # def init_lab3Widget(self):
    #     self.lab4Widget = Lab4Widget()
    #     self.addDockWidget(Qt.RightDockWidgetArea, self.lab4Widget)
    #     self.tabifyDockWidget(self.lab3Widget, self.lab4Widget)

    def connect_signals(self):
        self.settings.view.chb_full_screen.toggled.connect(self.window_mode)
        self.lab1Widget.btn_start.clicked.connect(self.lab1Widget.processor)
        # self.lab3Widget.btn_start.clicked.connect(self.lab3Widget.processor)
        # self.lab4Widget.btn_start.clicked.connect(self.lab4Widget.processor)

    def window_mode(self, state: bool):
        if state:
            self.showFullScreen()
        else:
            self.showNormal()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
