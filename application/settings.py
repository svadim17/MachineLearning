import yaml
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QApplication, \
    QGridLayout, QSlider, QPushButton, QDoubleSpinBox, QSpacerItem, QSizePolicy, QAbstractSpinBox, QCheckBox, \
    QSpinBox, QTableWidget, QTableWidgetItem, QCalendarWidget, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTabWidget
from PyQt5.QtGui import QIcon, QColor
from qt_material import QtStyleTools, list_themes
from loguru import logger


class SettingsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setLayout(QVBoxLayout())
        self.setWindowTitle('Settings')
        self.setMinimumSize(820, 500)
        self.config = {}

        self.tabWidget = QTabWidget()
        self.layout().addWidget(self.tabWidget)
        self.view = ViewSettingsWidget()
        self.tabWidget.addTab(self.view, 'View')

        self.btn_dump_config = QPushButton('Dump config')
        self.layout().addWidget(self.btn_dump_config)
        self.btn_dump_config.clicked.connect(self.dump_config)

        self.set_config()

    def set_config(self):
        try:
            self.read_config()
            self.view.set_config(self.config['view'])
        except:
            logger.exception('Can\'t load config')

    def read_config(self):
        with open('config.yaml', encoding='utf-8') as f:
            self.config = dict(yaml.load(f, Loader=yaml.SafeLoader))

    def dump_config(self):
        with open('config.yaml', 'w') as f:
            self.collect_config()
            yaml.dump(self.config, f, sort_keys=False)

    def collect_config(self):
        self.config.update(self.view.collect_config())


class ViewSettingsWidget(QWidget, QtStyleTools):
    def __init__(self):
        super().__init__()
        self.verticalLayout = QVBoxLayout()
        self.setLayout(self.verticalLayout)
        self.gridLayout = QGridLayout()
        self.verticalLayout.addLayout(self.gridLayout)

        self.l_themes = QLabel('Application Theme')
        self.cb_themes = QComboBox()
        self.cb_themes.addItems(list_themes())
        self.cb_themes.currentTextChanged.connect(lambda: self.apply_stylesheet(QApplication.instance(),
                                                                                self.cb_themes.currentText()))

        self.chb_full_screen = QCheckBox('Full Screen')

        self.widgets_to_layout()

    def collect_config(self):
        view_config = {'view': {'theme': self.cb_themes.currentText()}}
        return view_config

    def set_config(self, config: dict):
        try:
            self.cb_themes.setCurrentText(config['theme'])
        except:
            logger.exception('Can\'t load view conf')

    def widgets_to_layout(self):
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.l_themes, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.cb_themes, 3, 1, 1, 1)
        self.gridLayout.addItem(spacerItem, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.chb_full_screen, 5, 1, 1, 1)
        self.gridLayout.addItem(spacerItem, 6, 0, 1, 1)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = SettingsWidget()
    window.show()
    sys.exit(app.exec())