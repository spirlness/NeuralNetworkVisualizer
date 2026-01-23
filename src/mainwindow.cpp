#include "mainwindow.h"
#include <QSplitter>
#include <QScrollArea>
#include <QDateTime>
#include <cmath>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      networkView_(nullptr),
      lossChart_(nullptr),
      trainingThread_(nullptr),
      currentEpoch_(0),
      totalEpochs_(100) {
    setWindowTitle("Neural Network Training Visualizer");
    setMinimumSize(1200, 800);

    setupUI();
    createNetwork();
    loadDataset(0);
}

MainWindow::~MainWindow() {
    if (trainingThread_ && trainingThread_->isRunning()) {
        trainingThread_->stopTraining();
        trainingThread_->wait();
    }
}

void MainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    // 左侧控制面板
    QWidget* controlPanel = new QWidget();
    controlPanel->setFixedWidth(300);
    QVBoxLayout* controlLayout = new QVBoxLayout(controlPanel);

    // 数据集选择
    QGroupBox* dataGroup = new QGroupBox("Dataset");
    QVBoxLayout* dataLayout = new QVBoxLayout(dataGroup);
    datasetComboBox_ = new QComboBox();
    datasetComboBox_->addItem("XOR Problem");
    datasetComboBox_->addItem("AND Gate");
    datasetComboBox_->addItem("OR Gate");
    datasetComboBox_->addItem("Circle Classification");
    datasetComboBox_->setToolTip("Select a dataset to train the network on.");
    connect(datasetComboBox_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onDatasetChanged, Qt::QueuedConnection);
    dataLayout->addWidget(datasetComboBox_);
    controlLayout->addWidget(dataGroup);

    // 网络结构设置
    QGroupBox* networkGroup = new QGroupBox("Network Structure");
    QVBoxLayout* networkLayout = new QVBoxLayout(networkGroup);

    QHBoxLayout* hiddenLayerLayout = new QHBoxLayout();
    hiddenLayerLayout->addWidget(new QLabel("Hidden Layers:"));
    hiddenLayerSpinBox_ = new QSpinBox();
    hiddenLayerSpinBox_->setRange(1, 5);
    hiddenLayerSpinBox_->setValue(2);
    hiddenLayerSpinBox_->setToolTip("Number of hidden layers in the network.");
    hiddenLayerLayout->addWidget(hiddenLayerSpinBox_);
    networkLayout->addLayout(hiddenLayerLayout);

    QHBoxLayout* neuronsLayout = new QHBoxLayout();
    neuronsLayout->addWidget(new QLabel("Neurons/Layer:"));
    hiddenNeuronsSpinBox_ = new QSpinBox();
    hiddenNeuronsSpinBox_->setRange(2, 32);
    hiddenNeuronsSpinBox_->setValue(8);
    hiddenNeuronsSpinBox_->setToolTip("Number of neurons in each hidden layer.");
    neuronsLayout->addWidget(hiddenNeuronsSpinBox_);
    networkLayout->addLayout(neuronsLayout);

    QHBoxLayout* activationLayout = new QHBoxLayout();
    activationLayout->addWidget(new QLabel("Activation:"));
    activationComboBox_ = new QComboBox();
    activationComboBox_->addItem("Sigmoid");
    activationComboBox_->addItem("ReLU");
    activationComboBox_->addItem("Tanh");
    activationComboBox_->setToolTip("Activation function for hidden layers.");
    activationLayout->addWidget(activationComboBox_);
    networkLayout->addLayout(activationLayout);

    controlLayout->addWidget(networkGroup);

    // 训练参数
    QGroupBox* trainGroup = new QGroupBox("Training Parameters");
    QVBoxLayout* trainLayout = new QVBoxLayout(trainGroup);

    QHBoxLayout* epochsLayout = new QHBoxLayout();
    epochsLayout->addWidget(new QLabel("Epochs:"));
    epochsSpinBox_ = new QSpinBox();
    epochsSpinBox_->setRange(10, 10000);
    epochsSpinBox_->setValue(1000);
    epochsSpinBox_->setSingleStep(100);
    epochsSpinBox_->setToolTip("Maximum number of training epochs.");
    epochsLayout->addWidget(epochsSpinBox_);
    trainLayout->addLayout(epochsLayout);

    QHBoxLayout* lrLayout = new QHBoxLayout();
    lrLayout->addWidget(new QLabel("Learning Rate:"));
    learningRateSpinBox_ = new QDoubleSpinBox();
    learningRateSpinBox_->setRange(0.001, 10.0);
    learningRateSpinBox_->setValue(0.5);
    learningRateSpinBox_->setSingleStep(0.1);
    learningRateSpinBox_->setDecimals(3);
    learningRateSpinBox_->setToolTip("Learning rate controls how much to change the model in response to the estimated error each time the model weights are updated.");
    lrLayout->addWidget(learningRateSpinBox_);
    trainLayout->addLayout(lrLayout);

    controlLayout->addWidget(trainGroup);

    // 控制按钮
    QGroupBox* controlGroup = new QGroupBox("Controls");
    QVBoxLayout* buttonLayout = new QVBoxLayout(controlGroup);

    startButton_ = new QPushButton("Start Training");
    startButton_->setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;");
    startButton_->setToolTip("Start training the neural network with current settings.");
    connect(startButton_, &QPushButton::clicked, this, &MainWindow::onStartTraining);
    buttonLayout->addWidget(startButton_);

    pauseButton_ = new QPushButton("Pause");
    pauseButton_->setEnabled(false);
    pauseButton_->setToolTip("Pause or resume the training process.");
    connect(pauseButton_, &QPushButton::clicked, this, &MainWindow::onPauseResumeTraining);
    buttonLayout->addWidget(pauseButton_);

    stopButton_ = new QPushButton("Stop");
    stopButton_->setEnabled(false);
    stopButton_->setStyleSheet("background-color: #f44336; color: white;");
    stopButton_->setToolTip("Stop the training process.");
    connect(stopButton_, &QPushButton::clicked, this, &MainWindow::onStopTraining);
    buttonLayout->addWidget(stopButton_);

    resetButton_ = new QPushButton("Reset Network");
    resetButton_->setToolTip("Reset the network weights and biases.");
    connect(resetButton_, &QPushButton::clicked, this, &MainWindow::onResetNetwork);
    buttonLayout->addWidget(resetButton_);

    controlLayout->addWidget(controlGroup);

    // 进度条
    progressBar_ = new QProgressBar();
    progressBar_->setRange(0, 100);
    progressBar_->setValue(0);
    controlLayout->addWidget(progressBar_);

    // 状态标签
    statusLabel_ = new QLabel("Ready");
    statusLabel_->setStyleSheet("color: #888; padding: 5px;");
    controlLayout->addWidget(statusLabel_);

    // 日志
    QGroupBox* logGroup = new QGroupBox("Log");
    QVBoxLayout* logLayout = new QVBoxLayout(logGroup);
    logTextEdit_ = new QTextEdit();
    logTextEdit_->setReadOnly(true);
    logTextEdit_->setMaximumHeight(150);
    logTextEdit_->setStyleSheet("background-color: #2d2d3d; color: #aaa; font-family: monospace;");
    logLayout->addWidget(logTextEdit_);
    controlLayout->addWidget(logGroup);

    controlLayout->addStretch();

    mainLayout->addWidget(controlPanel);

    // 右侧可视化区域
    QSplitter* visualSplitter = new QSplitter(Qt::Vertical);

    // 网络结构可视化
    QGroupBox* networkViewGroup = new QGroupBox("Network Structure");
    QVBoxLayout* networkViewLayout = new QVBoxLayout(networkViewGroup);
    networkView_ = new NetworkView();
    networkViewLayout->addWidget(networkView_);
    visualSplitter->addWidget(networkViewGroup);

    // 损失曲线
    QGroupBox* lossGroup = new QGroupBox("Training Loss");
    QVBoxLayout* lossLayout = new QVBoxLayout(lossGroup);
    lossChart_ = new LossChart();
    lossLayout->addWidget(lossChart_);
    visualSplitter->addWidget(lossGroup);

    visualSplitter->setSizes({500, 300});

    mainLayout->addWidget(visualSplitter, 1);

    // 设置样式
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

    log("Application initialized");
}

void MainWindow::createNetwork() {
    network_ = std::make_unique<NeuralNetwork>();

    int inputSize = 2; // 根据数据集
    int outputSize = 1;

    network_->setInputSize(inputSize);

    ActivationType activation;
    switch (activationComboBox_->currentIndex()) {
        case 1:
            activation = ActivationType::ReLU;
            break;
        case 2:
            activation = ActivationType::Tanh;
            break;
        default:
            activation = ActivationType::Sigmoid;
            break;
    }

    int hiddenLayers = hiddenLayerSpinBox_->value();
    int neuronsPerLayer = hiddenNeuronsSpinBox_->value();

    for (int i = 0; i < hiddenLayers; ++i) {
        network_->addLayer(neuronsPerLayer, activation);
    }

    network_->addLayer(outputSize, ActivationType::Sigmoid);
    network_->build();

    networkView_->setNetwork(network_.get());

    log(QString("Network created: %1 input, %2 hidden layers x %3 neurons, %4 output")
            .arg(inputSize).arg(hiddenLayers).arg(neuronsPerLayer).arg(outputSize));
}

void MainWindow::loadDataset(int datasetIndex) {
    trainInputs_.clear();
    trainTargets_.clear();

    switch (datasetIndex) {
        case 0:
            trainInputs_ = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            trainTargets_ = {{0}, {1}, {1}, {0}};
            log("Loaded XOR dataset");
            break;

        case 1:
            trainInputs_ = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            trainTargets_ = {{0}, {0}, {0}, {1}};
            log("Loaded AND dataset");
            break;

        case 2:
            trainInputs_ = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            trainTargets_ = {{0}, {1}, {1}, {1}};
            log("Loaded OR dataset");
            break;

        case 3:
            for (int i = 0; i < 100; ++i) {
                double x = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
                double y = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
                double dist = std::sqrt(x * x + y * y);
                trainInputs_.push_back({(x + 1.0) / 2.0, (y + 1.0) / 2.0});
                trainTargets_.push_back({dist < 0.5 ? 1.0 : 0.0});
            }
            log("Loaded Circle classification dataset (100 samples)");
            break;
    }
}

void MainWindow::onDatasetChanged(int index) {
    loadDataset(index);
    onResetNetwork();
}

void MainWindow::onStartTraining() {
    if (!network_) {
        createNetwork();
    }

    lossChart_->clear();
    currentEpoch_ = 0;
    totalEpochs_ = epochsSpinBox_->value();

    trainingThread_ = std::make_unique<TrainingThread>();
    trainingThread_->setNetwork(network_.get());
    trainingThread_->setTrainingData(trainInputs_, trainTargets_);
    trainingThread_->setParameters(totalEpochs_, learningRateSpinBox_->value());

    connect(trainingThread_.get(), &TrainingThread::epochCompleted,
            this, &MainWindow::onEpochCompleted, Qt::QueuedConnection);
    connect(trainingThread_.get(), &TrainingThread::trainingCompleted,
            this, &MainWindow::onTrainingCompleted, Qt::QueuedConnection);
    connect(trainingThread_.get(), &TrainingThread::weightsUpdated,
            this, &MainWindow::onWeightsUpdated, Qt::QueuedConnection);

    trainingThread_->start();

    startButton_->setEnabled(false);
    stopButton_->setEnabled(true);
    pauseButton_->setEnabled(true);
    resetButton_->setEnabled(false);

    updateStatus("Training...");
    log(QString("Training started: %1 epochs, LR=%2")
            .arg(totalEpochs_).arg(learningRateSpinBox_->value()));
}

void MainWindow::onStopTraining() {
    if (trainingThread_) {
        trainingThread_->stopTraining();
    }
}

void MainWindow::onPauseResumeTraining() {
    if (!trainingThread_) return;

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

void MainWindow::onResetNetwork() {
    if (trainingThread_ && trainingThread_->isRunning()) {
        trainingThread_->stopTraining();
        trainingThread_->wait();
    }

    createNetwork();
    lossChart_->clear();
    progressBar_->setValue(0);

    startButton_->setEnabled(true);
    stopButton_->setEnabled(false);
    pauseButton_->setEnabled(false);
    pauseButton_->setText("Pause");
    resetButton_->setEnabled(true);

    updateStatus("Network reset");
    log("Network reset");
}

void MainWindow::onEpochCompleted(int epoch, double loss) {
    currentEpoch_ = epoch;
    lossChart_->addDataPoint(epoch, loss);

    int progress = static_cast<int>(100.0 * epoch / totalEpochs_);
    progressBar_->setValue(progress);

    if (epoch % 100 == 0 || epoch == totalEpochs_) {
        log(QString("Epoch %1/%2 - Loss: %3")
                .arg(epoch).arg(totalEpochs_).arg(loss, 0, 'f', 6));
    }
}

void MainWindow::onTrainingCompleted() {
    startButton_->setEnabled(true);
    stopButton_->setEnabled(false);
    pauseButton_->setEnabled(false);
    pauseButton_->setText("Pause");
    resetButton_->setEnabled(true);
    progressBar_->setValue(100);

    updateStatus("Training completed");
    log("Training completed!");

    // 测试结果
    log("--- Test Results ---");
    for (size_t i = 0; i < trainInputs_.size() && i < 10; ++i) {
        auto output = network_->forward(trainInputs_[i]);
        QString inputStr;
        for (double v : trainInputs_[i]) {
            inputStr += QString::number(v, 'f', 2) + " ";
        }
        log(QString("Input: [%1] -> Output: %2 (Expected: %3)")
                .arg(inputStr.trimmed())
                .arg(output[0], 0, 'f', 4)
                .arg(trainTargets_[i][0], 0, 'f', 1));
    }
}

void MainWindow::onWeightsUpdated() {
    networkView_->updateView();
}

void MainWindow::updateStatus(const QString& message) {
    statusLabel_->setText(message);
}

void MainWindow::log(const QString& message) {
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    logTextEdit_->append(QString("[%1] %2").arg(timestamp, message));
}
