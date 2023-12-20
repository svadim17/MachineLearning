def create_thread_neural_network(self):
    thread_neural_network = QThread()

    def on_started():
        lab2_widget = Lab2Widget()
        lab2_widget.setParent(None)  # Убираем родителя
        lab2_widget.show()

    thread_neural_network.started.connect(on_started)
    thread_neural_network.finished.connect(thread_neural_network.deleteLater)

    thread_neural_network.start()
