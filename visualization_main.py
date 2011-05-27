import PyQt4.Qt as Qt
import sys

def main():
    app = Qt.QApplication(sys.argv)

    import visualization
    nao_gui = visualization.NaoGui()
    nao_gui.show()

    app.exec_()

if __name__ == "__main__":
    main()
