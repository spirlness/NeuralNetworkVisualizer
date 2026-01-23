#include <QApplication>
#include <QCommandLineParser>
#include <QMessageBox>
#include <QPushButton>
#include <QDebug>
#include "mainwindow.h"
#include "cnn_mainwindow.h"
#include "attention_mainwindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    app.setApplicationName("Neural Network Training Visualizer");
    app.setApplicationVersion("2.0");
    app.setOrganizationName("NNViz");

    // Parse command-line arguments
    QCommandLineParser parser;
    parser.setApplicationDescription("Neural Network Training Visualizer");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("mode", "Network mode: mlp or cnn", "[mlp|cnn]");
    parser.process(app);

    const QStringList args = parser.positionalArguments();
    
    // Determine mode from CLI or GUI
    enum class Mode { None, MLP, CNN, Attention };
    Mode selectedMode = Mode::None;

    if (!args.isEmpty()) {
        QString mode = args.first().toLower();
        if (mode == "mlp") {
            selectedMode = Mode::MLP;
        } else if (mode == "cnn") {
            selectedMode = Mode::CNN;
        } else if (mode == "attention") {
            selectedMode = Mode::Attention;
        } else {
            qWarning() << "Invalid mode:" << mode << "- Valid options: mlp, cnn, attention";
        }
    }

    // If no valid CLI argument, show GUI selection
    if (selectedMode == Mode::None) {
        QMessageBox msgBox;
        msgBox.setWindowTitle("Select Network Type");
        msgBox.setText("Choose the neural network type to visualize:");
        msgBox.setIcon(QMessageBox::Question);

        QPushButton* mlpButton = msgBox.addButton("MLP (Fully Connected)", QMessageBox::ActionRole);
        QPushButton* cnnButton = msgBox.addButton("CNN (Convolutional)", QMessageBox::ActionRole);
        QPushButton* attnButton = msgBox.addButton("Attention (Transformer)", QMessageBox::ActionRole);
        msgBox.addButton(QMessageBox::Cancel);

        msgBox.exec();

        if (msgBox.clickedButton() == mlpButton) {
            selectedMode = Mode::MLP;
        } else if (msgBox.clickedButton() == cnnButton) {
            selectedMode = Mode::CNN;
        } else if (msgBox.clickedButton() == attnButton) {
            selectedMode = Mode::Attention;
        } else {
            // User cancelled, default to MLP
            selectedMode = Mode::MLP;
        }
    }

    // Launch the appropriate window
    if (selectedMode == Mode::MLP) {
        MainWindow window;
        window.show();
        return app.exec();
    } else if (selectedMode == Mode::CNN) {
        CNNMainWindow window;
        window.show();
        return app.exec();
    } else if (selectedMode == Mode::Attention) {
        AttentionMainWindow window;
        window.show();
        return app.exec();
    }

    return 0;
}
