#include "attention_mainwindow.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QSplitter>
#include <QDateTime>

AttentionMainWindow::AttentionMainWindow(QWidget* parent)
    : QMainWindow(parent),
      network_(nullptr),
      trainingThread_(nullptr),
      totalEpochs_(1000) {
    setWindowTitle("Attention Mechanism / Transformer Visualizer");
    setMinimumSize(1200, 800);

    setupUI();
    createNetwork();
}

AttentionMainWindow::~AttentionMainWindow() {
    if (trainingThread_ && trainingThread_->isRunning()) {
        trainingThread_->stopTraining();
        trainingThread_->wait();
    }
}

void AttentionMainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);

    // Left Control Panel
    QWidget* controlPanel = new QWidget();
    controlPanel->setFixedWidth(300);
    QVBoxLayout* controlLayout = new QVBoxLayout(controlPanel);

    // Params
    QGroupBox* paramGroup = new QGroupBox("Parameters");
    QVBoxLayout* paramLayout = new QVBoxLayout(paramGroup);

    // Seq Len
    QHBoxLayout* seqLayout = new QHBoxLayout();
    seqLayout->addWidget(new QLabel("Seq Len:"));
    seqLenSpinBox_ = new QSpinBox();
    seqLenSpinBox_->setRange(3, 20);
    seqLenSpinBox_->setValue(5);
    seqLayout->addWidget(seqLenSpinBox_);
    paramLayout->addLayout(seqLayout);

    // D_model
    QHBoxLayout* dimLayout = new QHBoxLayout();
    dimLayout->addWidget(new QLabel("D_model:"));
    dModelSpinBox_ = new QSpinBox();
    dModelSpinBox_->setRange(4, 64);
    dModelSpinBox_->setValue(16);
    dModelSpinBox_->setSingleStep(4);
    dimLayout->addWidget(dModelSpinBox_);
    paramLayout->addLayout(dimLayout);

    // Layers
    QHBoxLayout* layerLayout = new QHBoxLayout();
    layerLayout->addWidget(new QLabel("Layers:"));
    layersSpinBox_ = new QSpinBox();
    layersSpinBox_->setRange(1, 6);
    layersSpinBox_->setValue(1);
    layerLayout->addWidget(layersSpinBox_);
    paramLayout->addLayout(layerLayout);

    controlLayout->addWidget(paramGroup);

    // Training Params
    QGroupBox* trainGroup = new QGroupBox("Training");
    QVBoxLayout* trainLayout = new QVBoxLayout(trainGroup);

    QHBoxLayout* epochLayout = new QHBoxLayout();
    epochLayout->addWidget(new QLabel("Epochs:"));
    epochsSpinBox_ = new QSpinBox();
    epochsSpinBox_->setRange(10, 10000);
    epochsSpinBox_->setValue(1000);
    epochLayout->addWidget(epochsSpinBox_);
    trainLayout->addLayout(epochLayout);

    QHBoxLayout* lrLayout = new QHBoxLayout();
    lrLayout->addWidget(new QLabel("LR:"));
    learningRateSpinBox_ = new QDoubleSpinBox();
    learningRateSpinBox_->setRange(0.0001, 1.0);
    learningRateSpinBox_->setValue(0.01);
    learningRateSpinBox_->setDecimals(4);
    learningRateSpinBox_->setSingleStep(0.001);
    lrLayout->addWidget(learningRateSpinBox_);
    trainLayout->addLayout(lrLayout);

    controlLayout->addWidget(trainGroup);

    // Controls
    QGroupBox* btnGroup = new QGroupBox("Controls");
    QVBoxLayout* btnLayout = new QVBoxLayout(btnGroup);

    startButton_ = new QPushButton("Start Training");
    connect(startButton_, &QPushButton::clicked, this, &AttentionMainWindow::onStartTraining);
    btnLayout->addWidget(startButton_);

    pauseButton_ = new QPushButton("Pause");
    pauseButton_->setEnabled(false);
    connect(pauseButton_, &QPushButton::clicked, this, &AttentionMainWindow::onPauseResumeTraining);
    btnLayout->addWidget(pauseButton_);

    stopButton_ = new QPushButton("Stop");
    stopButton_->setEnabled(false);
    connect(stopButton_, &QPushButton::clicked, this, &AttentionMainWindow::onStopTraining);
    btnLayout->addWidget(stopButton_);

    resetButton_ = new QPushButton("Reset");
    connect(resetButton_, &QPushButton::clicked, this, &AttentionMainWindow::onResetNetwork);
    btnLayout->addWidget(resetButton_);

    controlLayout->addWidget(btnGroup);

    progressBar_ = new QProgressBar();
    controlLayout->addWidget(progressBar_);

    statusLabel_ = new QLabel("Ready");
    controlLayout->addWidget(statusLabel_);

    logTextEdit_ = new QTextEdit();
    logTextEdit_->setReadOnly(true);
    controlLayout->addWidget(logTextEdit_);

    mainLayout->addWidget(controlPanel);

    // Right Panel
    QSplitter* splitter = new QSplitter(Qt::Vertical);

    // Visualization
    QGroupBox* vizGroup = new QGroupBox("Visualization");
    QVBoxLayout* vizLayout = new QVBoxLayout(vizGroup);
    attentionView_ = new AttentionView();
    vizLayout->addWidget(attentionView_);
    splitter->addWidget(vizGroup);

    // Loss Chart
    QGroupBox* lossGroup = new QGroupBox("Loss");
    QVBoxLayout* lossLayout = new QVBoxLayout(lossGroup);
    lossChart_ = new LossChart();
    lossLayout->addWidget(lossChart_);
    splitter->addWidget(lossGroup);

    splitter->setSizes({600, 200});

    mainLayout->addWidget(splitter, 1);

    // Styles
    setStyleSheet(R"(
        QMainWindow { background-color: #252536; }
        QLabel { color: #ddd; }
        QGroupBox { color: white; font-weight: bold; border: 1px solid #444; margin-top: 10px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QSpinBox, QDoubleSpinBox { background-color: #3d3d4d; color: white; padding: 5px; }
        QPushButton { background-color: #3d3d5d; color: white; border: none; padding: 8px; border-radius: 4px; }
        QPushButton:hover { background-color: #4d4d6d; }
        QPushButton:disabled { background-color: #2d2d3d; color: #666; }
        QTextEdit { background-color: #2d2d3d; color: #aaa; }
    )");
}

void AttentionMainWindow::createNetwork() {
    int seqLen = seqLenSpinBox_->value();
    int d_model = dModelSpinBox_->value();
    int layers = layersSpinBox_->value();
    int d_k = d_model; // Use same for simplicity
    int d_ff = d_model * 2; // Simple multiplier

    network_ = std::make_unique<AttentionNetwork>(seqLen, d_model, d_k, d_ff, layers);
    attentionView_->setNetwork(network_.get());

    log(QString("Network created: Seq=%1, D=%2, Layers=%3").arg(seqLen).arg(d_model).arg(layers));
}

void AttentionMainWindow::onStartTraining() {
    int currentSeqLen = seqLenSpinBox_->value();
    int currentDModel = dModelSpinBox_->value();
    int currentLayers = layersSpinBox_->value();

    if (network_) {
        // Check if architecture parameters changed
        bool paramsChanged = false;
        if (network_->getDModel() != (size_t)currentDModel) paramsChanged = true;
        if (network_->getNumLayers() != (size_t)currentLayers) paramsChanged = true;
        // SeqLen is now dynamic, but if it changed significantly we might want to reset?
        // Actually, if seqLen changed, we definitely want the training thread to use the new one.
        // The network handles dynamic seqLen, but we might want to update the network's concept of it.
        if (network_->getSeqLen() != (size_t)currentSeqLen) {
             // For visualization consistency, we accept the change. 
             // Network will resize on first forward pass.
        }

        if (paramsChanged) {
            log("Configuration changed. Recreating network...");
            createNetwork();
        }
    } else {
        createNetwork();
    }

    lossChart_->clear();
    progressBar_->setValue(0);
    totalEpochs_ = epochsSpinBox_->value();

    trainingThread_ = std::make_unique<AttentionTrainingThread>();
    trainingThread_->setNetwork(network_.get());
    trainingThread_->setParameters(totalEpochs_, learningRateSpinBox_->value(), seqLenSpinBox_->value());

    connect(trainingThread_.get(), &AttentionTrainingThread::epochCompleted, this, &AttentionMainWindow::onEpochCompleted);
    connect(trainingThread_.get(), &AttentionTrainingThread::trainingCompleted, this, &AttentionMainWindow::onTrainingCompleted);
    connect(trainingThread_.get(), &AttentionTrainingThread::weightsUpdated, this, &AttentionMainWindow::onWeightsUpdated);

    trainingThread_->startTraining();

    startButton_->setEnabled(false);
    stopButton_->setEnabled(true);
    pauseButton_->setEnabled(true);
    resetButton_->setEnabled(false);

    updateStatus("Training...");
}

void AttentionMainWindow::onStopTraining() {
    if (trainingThread_) trainingThread_->stopTraining();
}

void AttentionMainWindow::onPauseResumeTraining() {
    if (!trainingThread_) return;
    if (trainingThread_->isPaused()) {
        trainingThread_->resumeTraining();
        pauseButton_->setText("Pause");
        updateStatus("Training...");
    } else {
        trainingThread_->pauseTraining();
        pauseButton_->setText("Resume");
        updateStatus("Paused");
    }
}

void AttentionMainWindow::onResetNetwork() {
    onStopTraining();
    createNetwork();
    lossChart_->clear();
    progressBar_->setValue(0);
    updateStatus("Network Reset");
}

void AttentionMainWindow::onEpochCompleted(int epoch, double loss) {
    lossChart_->addDataPoint(epoch, loss);
    progressBar_->setValue((epoch * 100) / totalEpochs_);
    if (epoch % 100 == 0) log(QString("Epoch %1: Loss %2").arg(epoch).arg(loss));
}

void AttentionMainWindow::onTrainingCompleted() {
    startButton_->setEnabled(true);
    stopButton_->setEnabled(false);
    pauseButton_->setEnabled(false);
    resetButton_->setEnabled(true);
    updateStatus("Training Completed");
    log("Training Completed");
}

void AttentionMainWindow::onWeightsUpdated() {
    attentionView_->updateView();
}

void AttentionMainWindow::updateStatus(const QString& message) {
    statusLabel_->setText(message);
}

void AttentionMainWindow::log(const QString& message) {
    logTextEdit_->append(QString("[%1] %2").arg(QDateTime::currentDateTime().toString("HH:mm:ss"), message));
}
