#include "cnn_mainwindow.h"
#include <QSplitter>
#include <QDateTime>
#include <QMessageBox>
#include <cmath>
#include <random>
#include <algorithm>

CNNMainWindow::CNNMainWindow(QWidget* parent)
    : QMainWindow(parent) {
    setWindowTitle("CNN Neural Network Visualizer");
    setMinimumSize(1400, 900);

    setupUI();
}

CNNMainWindow::~CNNMainWindow() {
    if (trainingThread_) {
        trainingThread_->stopTraining();
        trainingThread_->wait();
    }
}

void CNNMainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    // 左侧控制面板
    QWidget* controlPanel = new QWidget();
    controlPanel->setFixedWidth(320);
    QVBoxLayout* controlLayout = new QVBoxLayout(controlPanel);

    // CNN网络配置
    QGroupBox* cnnConfigGroup = new QGroupBox("CNN Configuration");
    QVBoxLayout* cnnConfigLayout = new QVBoxLayout(cnnConfigGroup);

    QHBoxLayout* inputLayout = new QHBoxLayout();
    inputLayout->addWidget(new QLabel("Input Size:"));
    inputSizeSpinBox_ = new QSpinBox();
    inputSizeSpinBox_->setRange(8, 64);
    inputSizeSpinBox_->setValue(16);
    inputLayout->addWidget(inputSizeSpinBox_);
    inputLayout->addWidget(new QLabel("x"));
    inputLayout->addWidget(new QLabel("16"));
    cnnConfigLayout->addLayout(inputLayout);

    QHBoxLayout* classesLayout = new QHBoxLayout();
    classesLayout->addWidget(new QLabel("Classes:"));
    numClassesSpinBox_ = new QSpinBox();
    numClassesSpinBox_->setRange(2, 10);
    numClassesSpinBox_->setValue(3);
    classesLayout->addWidget(numClassesSpinBox_);
    cnnConfigLayout->addLayout(classesLayout);

    QHBoxLayout* conv1Layout = new QHBoxLayout();
    conv1Layout->addWidget(new QLabel("Conv1 Filters:"));
    conv1FiltersSpinBox_ = new QSpinBox();
    conv1FiltersSpinBox_->setRange(4, 64);
    conv1FiltersSpinBox_->setValue(8);
    conv1Layout->addWidget(conv1FiltersSpinBox_);
    cnnConfigLayout->addLayout(conv1Layout);

    QHBoxLayout* conv2Layout = new QHBoxLayout();
    conv2Layout->addWidget(new QLabel("Conv2 Filters:"));
    conv2FiltersSpinBox_ = new QSpinBox();
    conv2FiltersSpinBox_->setRange(8, 128);
    conv2FiltersSpinBox_->setValue(16);
    conv2Layout->addWidget(conv2FiltersSpinBox_);
    cnnConfigLayout->addLayout(conv2Layout);

    QHBoxLayout* kernelLayout = new QHBoxLayout();
    kernelLayout->addWidget(new QLabel("Kernel Size:"));
    kernelSizeSpinBox_ = new QSpinBox();
    kernelSizeSpinBox_->setRange(3, 7);
    kernelSizeSpinBox_->setValue(3);
    kernelLayout->addWidget(kernelSizeSpinBox_);
    cnnConfigLayout->addLayout(kernelLayout);

    QHBoxLayout* hiddenLayout = new QHBoxLayout();
    hiddenLayout->addWidget(new QLabel("Hidden Neurons:"));
    hiddenNeuronsSpinBox_ = new QSpinBox();
    hiddenNeuronsSpinBox_->setRange(16, 256);
    hiddenNeuronsSpinBox_->setValue(64);
    hiddenLayout->addWidget(hiddenNeuronsSpinBox_);
    cnnConfigLayout->addLayout(hiddenLayout);

    buildButton_ = new QPushButton("Build Network");
    buildButton_->setStyleSheet("background-color: #2196F3; color: white; padding: 8px;");
    connect(buildButton_, &QPushButton::clicked, this, &CNNMainWindow::onBuildNetwork);
    cnnConfigLayout->addWidget(buildButton_);

    controlLayout->addWidget(cnnConfigGroup);

    // 训练数据配置
    QGroupBox* dataGroup = new QGroupBox("Training Data");
    QVBoxLayout* dataLayout = new QVBoxLayout(dataGroup);

    QHBoxLayout* samplesLayout = new QHBoxLayout();
    samplesLayout->addWidget(new QLabel("Samples/Class:"));
    samplesSpinBox_ = new QSpinBox();
    samplesSpinBox_->setRange(10, 500);
    samplesSpinBox_->setValue(50);
    samplesLayout->addWidget(samplesSpinBox_);
    dataLayout->addLayout(samplesLayout);

    controlLayout->addWidget(dataGroup);

    // 训练参数
    QGroupBox* trainGroup = new QGroupBox("Training Parameters");
    QVBoxLayout* trainLayout = new QVBoxLayout(trainGroup);

    QHBoxLayout* epochsLayout = new QHBoxLayout();
    epochsLayout->addWidget(new QLabel("Epochs:"));
    epochsSpinBox_ = new QSpinBox();
    epochsSpinBox_->setRange(10, 1000);
    epochsSpinBox_->setValue(100);
    epochsSpinBox_->setSingleStep(10);
    epochsLayout->addWidget(epochsSpinBox_);
    trainLayout->addLayout(epochsLayout);

    QHBoxLayout* lrLayout = new QHBoxLayout();
    lrLayout->addWidget(new QLabel("Learning Rate:"));
    learningRateSpinBox_ = new QDoubleSpinBox();
    learningRateSpinBox_->setRange(0.0001, 1.0);
    learningRateSpinBox_->setValue(0.01);
    learningRateSpinBox_->setSingleStep(0.001);
    learningRateSpinBox_->setDecimals(4);
    lrLayout->addWidget(learningRateSpinBox_);
    trainLayout->addLayout(lrLayout);

    controlLayout->addWidget(trainGroup);

    // 控制按钮
    QGroupBox* controlGroup = new QGroupBox("Controls");
    QVBoxLayout* buttonLayout = new QVBoxLayout(controlGroup);

    startButton_ = new QPushButton("Start Training");
    startButton_->setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;");
    startButton_->setEnabled(false);
    connect(startButton_, &QPushButton::clicked, this, &CNNMainWindow::onStartTraining);
    buttonLayout->addWidget(startButton_);

    pauseButton_ = new QPushButton("Pause");
    pauseButton_->setEnabled(false);
    connect(pauseButton_, &QPushButton::clicked, this, &CNNMainWindow::onPauseResumeTraining);
    buttonLayout->addWidget(pauseButton_);

    stopButton_ = new QPushButton("Stop");
    stopButton_->setEnabled(false);
    stopButton_->setStyleSheet("background-color: #f44336; color: white;");
    connect(stopButton_, &QPushButton::clicked, this, &CNNMainWindow::onStopTraining);
    buttonLayout->addWidget(stopButton_);

    resetButton_ = new QPushButton("Reset Network");
    connect(resetButton_, &QPushButton::clicked, this, &CNNMainWindow::onResetNetwork);
    buttonLayout->addWidget(resetButton_);

    controlLayout->addWidget(controlGroup);

    // 进度条
    progressBar_ = new QProgressBar();
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    controlLayout->addWidget(progressBar_);

    // 状态标签
    statusLabel_ = new QLabel("Ready - Build network to start");
    statusLabel_->setStyleSheet("color: #888; padding: 5px;");
    controlLayout->addWidget(statusLabel_);

    // 日志
    QGroupBox* logGroup = new QGroupBox("Log");
    QVBoxLayout* logLayout = new QVBoxLayout(logGroup);
    logTextEdit_ = new QTextEdit();
    logTextEdit_->setReadOnly(true);
    logTextEdit_->setMaximumHeight(120);
    logTextEdit_->setStyleSheet("background-color: #2d2d3d; color: #aaa; font-family: monospace;");
    logLayout->addWidget(logTextEdit_);
    controlLayout->addWidget(logGroup);

    controlLayout->addStretch();
    mainLayout->addWidget(controlPanel);

    // 右侧可视化区域
    QSplitter* visualSplitter = new QSplitter(Qt::Vertical);

    // CNN架构可视化
    QGroupBox* cnnViewGroup = new QGroupBox("CNN Architecture");
    QVBoxLayout* cnnViewLayout = new QVBoxLayout(cnnViewGroup);
    cnnView_ = new CNNView();
    connect(cnnView_, &CNNView::layerClicked, this, &CNNMainWindow::onLayerClicked);
    cnnViewLayout->addWidget(cnnView_);
    visualSplitter->addWidget(cnnViewGroup);

    // 下方分割：特征图和损失曲线
    QSplitter* bottomSplitter = new QSplitter(Qt::Horizontal);

    // 特征图可视化
    QGroupBox* featureGroup = new QGroupBox("Feature Maps");
    QVBoxLayout* featureLayout = new QVBoxLayout(featureGroup);
    QScrollArea* featureScroll = new QScrollArea();
    featureMapView_ = new FeatureMapView();
    featureScroll->setWidget(featureMapView_);
    featureScroll->setWidgetResizable(true);
    featureLayout->addWidget(featureScroll);
    bottomSplitter->addWidget(featureGroup);

    // 损失曲线
    QGroupBox* lossGroup = new QGroupBox("Training Loss");
    QVBoxLayout* lossLayout = new QVBoxLayout(lossGroup);
    lossChart_ = new LossChart();
    lossLayout->addWidget(lossChart_);
    bottomSplitter->addWidget(lossGroup);

    bottomSplitter->setSizes({500, 400});
    visualSplitter->addWidget(bottomSplitter);

    visualSplitter->setSizes({400, 350});
    mainLayout->addWidget(visualSplitter, 1);

    setStyleSheet(R"(
        QMainWindow {
            background-color: #252536;
        }
        QGroupBox {
            color: white;
            font-weight: bold;
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QLabel {
            color: #ddd;
        }
        QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #3d3d4d;
            color: white;
            border: 1px solid #555;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton {
            background-color: #3d3d5d;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #4d4d6d;
        }
        QPushButton:disabled {
            background-color: #2d2d3d;
            color: #666;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 5px;
            text-align: center;
            color: white;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 5px;
        }
    )");

    trainingThread_ = std::make_unique<CNNTrainingThread>(this);
    connect(trainingThread_.get(), &CNNTrainingThread::epochCompleted, 
            this, &CNNMainWindow::onEpochCompleted);
    connect(trainingThread_.get(), &CNNTrainingThread::trainingCompleted, 
            this, &CNNMainWindow::onTrainingCompleted);
    connect(trainingThread_.get(), &CNNTrainingThread::weightsUpdated, 
            this, &CNNMainWindow::onWeightsUpdated);

    log("CNN Visualizer initialized");
}

void CNNMainWindow::onBuildNetwork() {
    try {
        createNetwork();
        generateTrainingData();

        cnnView_->setNetwork(cnnNetwork_.get());
        cnnView_->updateView();

        startButton_->setEnabled(true);
        updateStatus("Network built - Ready to train");

        auto descriptions = cnnNetwork_->getLayerDescriptions();
        log("Network architecture:");
        for (const auto& desc : descriptions) {
            log(QString("  ") + QString::fromStdString(desc));
        }
        log(QString("Total parameters: %1").arg(cnnNetwork_->totalParameters()));
    } catch (const std::exception& e) {
        startButton_->setEnabled(false);
        updateStatus("Build failed");
        log(QString("Build failed: %1").arg(e.what()));
        QMessageBox::critical(this, "Build CNN Network Failed", e.what());
    }
}

void CNNMainWindow::createNetwork() {
    cnnNetwork_ = std::make_unique<CNNNetwork>();

    int inputSize = inputSizeSpinBox_->value();
    int numClasses = numClassesSpinBox_->value();
    int conv1Filters = conv1FiltersSpinBox_->value();
    int conv2Filters = conv2FiltersSpinBox_->value();
    int kernelSize = kernelSizeSpinBox_->value();
    int hiddenNeurons = hiddenNeuronsSpinBox_->value();

    cnnNetwork_->setInputSize(1, inputSize, inputSize);

    // Conv1 + Pool1
    cnnNetwork_->addConvLayer(conv1Filters, kernelSize, 1, 1, CNNActivationType::ReLU);
    cnnNetwork_->addPoolingLayer(2, 2, PoolingType::Max);

    // Conv2 + Pool2
    cnnNetwork_->addConvLayer(conv2Filters, kernelSize, 1, 1, CNNActivationType::ReLU);
    cnnNetwork_->addPoolingLayer(2, 2, PoolingType::Max);

    // Flatten + Dense
    cnnNetwork_->addFlattenLayer();
    cnnNetwork_->addDenseLayer(hiddenNeurons, ActivationType::ReLU);
    cnnNetwork_->addDenseLayer(numClasses, ActivationType::Sigmoid);

    cnnNetwork_->build();

    log(QString("Created CNN: %1x%1 input, %2 classes")
        .arg(inputSize).arg(numClasses));
}

void CNNMainWindow::generateTrainingData() {
    trainImages_.clear();
    trainLabels_.clear();

    int inputSize = inputSizeSpinBox_->value();
    int numClasses = numClassesSpinBox_->value();
    int samplesPerClass = samplesSpinBox_->value();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> posDist(0.2, 0.8);
    std::uniform_real_distribution<> sizeDist(0.15, 0.35);
    std::normal_distribution<> noiseDist(0.0, 0.05);

    for (int c = 0; c < numClasses; ++c) {
        for (int s = 0; s < samplesPerClass; ++s) {
            Tensor image(1, inputSize, inputSize, 0.0);

            double centerX = posDist(gen);
            double centerY = posDist(gen);
            double size = sizeDist(gen);

            for (int y = 0; y < inputSize; ++y) {
                for (int x = 0; x < inputSize; ++x) {
                    double nx = static_cast<double>(x) / inputSize;
                    double ny = static_cast<double>(y) / inputSize;
                    double dx = nx - centerX;
                    double dy = ny - centerY;

                    double value = 0.0;

                    switch (c % 3) {
                        case 0: // 圆形
                            if (std::sqrt(dx*dx + dy*dy) < size) {
                                value = 1.0;
                            }
                            break;
                        case 1: // 方形
                            if (std::abs(dx) < size && std::abs(dy) < size) {
                                value = 1.0;
                            }
                            break;
                        case 2: // 十字形
                            if ((std::abs(dx) < size/3 && std::abs(dy) < size) ||
                                (std::abs(dy) < size/3 && std::abs(dx) < size)) {
                                value = 1.0;
                            }
                            break;
                    }

                    value += noiseDist(gen);
                    value = std::clamp(value, 0.0, 1.0);
                    image(0, y, x) = value;
                }
            }

            trainImages_.push_back(image);

            std::vector<double> label(numClasses, 0.0);
            label[c] = 1.0;
            trainLabels_.push_back(label);
        }
    }

    log(QString("Generated %1 training samples (%2 classes)")
        .arg(trainImages_.size()).arg(numClasses));
}

void CNNMainWindow::onStartTraining() {
    if (!cnnNetwork_) {
        return;
    }

    lossChart_->clear();
    totalEpochs_ = epochsSpinBox_->value();
    double learningRate = learningRateSpinBox_->value();

    trainingThread_->setNetwork(cnnNetwork_.get());
    trainingThread_->setTrainingData(trainImages_, trainLabels_);
    trainingThread_->setParameters(totalEpochs_, learningRate);

    startButton_->setEnabled(false);
    stopButton_->setEnabled(true);
    pauseButton_->setEnabled(true);
    buildButton_->setEnabled(false);
    resetButton_->setEnabled(false);

    trainingThread_->start();

    updateStatus("Training...");
    log(QString("Training started: %1 epochs, LR=%2")
        .arg(totalEpochs_).arg(learningRate));
}

void CNNMainWindow::onEpochCompleted(int epoch, double loss) {
    lossChart_->addDataPoint(epoch, loss);

    int progress = static_cast<int>(100.0 * epoch / totalEpochs_);
    progressBar_->setValue(progress);

    if (epoch % 10 == 0) {
        log(QString("Epoch %1/%2 - Loss: %3")
            .arg(epoch).arg(totalEpochs_).arg(loss, 0, 'f', 6));
    }
}

void CNNMainWindow::onTrainingCompleted() {
    startButton_->setEnabled(true);
    stopButton_->setEnabled(false);
    pauseButton_->setEnabled(false);
    pauseButton_->setText("Pause");
    buildButton_->setEnabled(true);
    resetButton_->setEnabled(true);

    updateStatus("Training completed");
    log("Training completed!");
}

void CNNMainWindow::onWeightsUpdated() {
    cnnView_->updateView();
}

void CNNMainWindow::onStopTraining() {
    trainingThread_->stopTraining();

    startButton_->setEnabled(true);
    stopButton_->setEnabled(false);
    pauseButton_->setEnabled(false);
    pauseButton_->setText("Pause");
    buildButton_->setEnabled(true);
    resetButton_->setEnabled(true);

    updateStatus("Training stopped");
    log("Training stopped");
}

void CNNMainWindow::onPauseResumeTraining() {
    if (trainingThread_->isPaused()) {
        trainingThread_->resumeTraining();
        pauseButton_->setText("Pause");
        updateStatus("Training...");
        log("Training resumed");
    } else {
        trainingThread_->pauseTraining();
        pauseButton_->setText("Resume");
        updateStatus("Paused");
        log("Training paused");
    }
}

void CNNMainWindow::onResetNetwork() {
    onStopTraining();
    cnnNetwork_.reset();
    cnnView_->setNetwork(nullptr);
    featureMapView_->clear();
    lossChart_->clear();
    progressBar_->setValue(0);

    startButton_->setEnabled(false);
    updateStatus("Network reset - Build new network");
    log("Network reset");
}

void CNNMainWindow::onLayerClicked(int layerIndex) {
    if (!cnnNetwork_) return;

    std::lock_guard<std::mutex> lock(cnnNetwork_->getMutex());
    const auto& cnnLayers = cnnNetwork_->getCNNLayers();

    int cnnLayerIndex = layerIndex - 1;

    if (cnnLayerIndex >= 0 && cnnLayerIndex < static_cast<int>(cnnLayers.size())) {
        const auto& layer = cnnLayers[cnnLayerIndex];
        const Tensor& output = layer->getOutput();

        if (!output.empty()) {
            QString layerName = QString::fromStdString(layer->name()) +
                               QString(" (Layer %1)").arg(cnnLayerIndex);
            featureMapView_->setFeatureMap(output, layerName);
            log(QString("Showing feature maps for %1").arg(layerName));
        }
    }
}

void CNNMainWindow::updateStatus(const QString& message) {
    statusLabel_->setText(message);
}

void CNNMainWindow::log(const QString& message) {
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    logTextEdit_->append(QString("[%1] %2").arg(timestamp, message));
}
