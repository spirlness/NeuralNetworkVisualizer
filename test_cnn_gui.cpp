#include <QApplication>
#include "cnn_mainwindow.h"
#include <iostream>

int main(int argc, char *argv[]) {
    try {
        QApplication app(argc, argv);
        
        std::cout << "Creating CNN MainWindow..." << std::endl;
        CNNMainWindow window;
        
        std::cout << "Showing window..." << std::endl;
        window.show();
        
        std::cout << "Starting event loop..." << std::endl;
        return app.exec();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 1;
    }
}
