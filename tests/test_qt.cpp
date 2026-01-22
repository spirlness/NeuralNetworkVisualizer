#include <QApplication>
#include <QLabel>
#include <QWidget>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QWidget window;
    QLabel* label = new QLabel("Hello Qt!", &window);
    label->setAlignment(Qt::AlignCenter);
    window.resize(400, 300);
    window.show();

    return app.exec();
}
