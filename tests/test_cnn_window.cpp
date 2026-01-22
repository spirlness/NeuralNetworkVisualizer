#include <QApplication>
#include <QMessageBox>
#include <iostream>
#include "cnn_mainwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    try {
        std::cout << "Step 1: QApplication created" << std::endl;
        
        std::cout << "Step 2: Creating CNNMainWindow..." << std::endl;
        CNNMainWindow* window = new CNNMainWindow();
        
        std::cout << "Step 3: Window created successfully" << std::endl;
        std::cout << "Step 4: Showing window..." << std::endl;
        
        window->show();
        
        std::cout << "Step 5: Window shown, starting event loop..." << std::endl;
        
        QMessageBox::information(nullptr, "Test", "CNN Window should be visible now!");
        
        int result = app.exec();
        
        delete window;
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        QMessageBox::critical(nullptr, "Error", QString("Exception: ") + e.what());
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN EXCEPTION" << std::endl;
        QMessageBox::critical(nullptr, "Error", "Unknown exception occurred!");
        return 1;
    }
}
