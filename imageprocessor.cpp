#include "imageprocessor.h"
#include "ui_imageprocessor.h"
#include "imageprocess.h"
#include <QFileDialog>
#include <qdebug.h>
#include <memory>
#include <QMessageBox>

ImageProcessor::ImageProcessor(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ImageProcessor)
{
    ui->setupUi(this);

    ui->imageLabel->setBackgroundRole(QPalette::Base);
    ui->imageLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->imageLabel->setScaledContents(true);
    ui->imageScrollArea->setBackgroundRole(QPalette::Dark);
    ui->imageScrollArea->setWidgetResizable(true);
    ui->processScrollAreaContents->resize(ui->processScrollAreaContents->size());

    m_origIm = std::make_unique<QImage>();
}

ImageProcessor::~ImageProcessor()
{
    delete ui;
}


void ImageProcessor::on_actionOpen_Image_triggered()
{
    QString filename = QFileDialog::getOpenFileName(this,tr("Open Image"),
                       "",
                       tr("Image Files (*.png *.jpg *.bmp)"));
    if(QString::compare(filename, "") != 0){
        m_origIm->load(filename);
        resetUI();
        ui->processScrollArea->setEnabled(true);
        ui->bitPlaneZero->setEnabled(true);
        if(!m_origIm->isGrayscale()){
            ui->convertToGray->setEnabled(true);
            m_origIm = std::make_unique<QImage>(m_origIm->convertToFormat(QImage::Format_RGB32));
        } else
            m_origIm = std::make_unique<QImage>(m_origIm->convertToFormat(QImage::Format_Grayscale8));
        ui->mainLabel->setText(tr("Process Image:"));
    }
}

void ImageProcessor::display(const std::unique_ptr<QImage> &origIm, const std::unique_ptr<QImage> &procIm)
{
    QImage dispIm = QImage(origIm->width() + procIm->width(),
                               std::max(origIm->height(), procIm->height()), origIm->format());

    for(int i = 0; i < origIm->width(); i++){
        for(int j = 0; j < origIm->height(); j++){
            dispIm.setPixel(i,j, origIm->pixel(i,j));
        }
    }
    for(int i = 0; i < procIm->width(); i++){
        for(int j = 0; j < procIm->height(); j++){
            dispIm.setPixel(i+origIm->width(),j,procIm->pixel(i,j));
        }
    }
    ui->imageLabel->setPixmap(QPixmap::fromImage(dispIm));
    ui->imageLabel->adjustSize();
}

void ImageProcessor::on_resizeFactor_valueChanged(int arg1)
{
    if(arg1 > 100){
        ui->upsampleCombo->setEnabled(true);
    }
    else{
        ui->upsampleCombo->setEnabled(false);
    }
}

void ImageProcessor::on_applyResize_clicked()
{
    if(ui->resizeFactor->value() > 100){
        double xFactor = ui->resizeFactor->value()/100.0;
        double yFactor = ui->resizeFactor->value()/100.0;
        if(ui->upsampleCombo->currentText() == "Nearest Neighbor") {
            m_procIm = std::make_unique<QImage>(ImageProcess::nearestNeighbor(*(m_origIm.get()), xFactor, yFactor));
        }
        else if(ui->upsampleCombo->currentText() == "Linear X-Direction") {
            m_procIm = std::make_unique<QImage>(ImageProcess::linear(*(m_origIm.get()), xFactor, yFactor, true));
        }
        else if(ui->upsampleCombo->currentText() == "Linear Y-Direction") {
            m_procIm = std::make_unique<QImage>(ImageProcess::linear(*(m_origIm.get()), xFactor, yFactor, false));
        }
        else { // bilinear
            m_procIm = std::make_unique<QImage>(ImageProcess::bilinear(*(m_origIm.get()), xFactor, yFactor));
        }
    } else if(ui->resizeFactor->value() < 100) {
        m_procIm = std::make_unique<QImage>(ImageProcess::nearestNeighbor(*(m_origIm.get()), ui->resizeFactor->value()/100.0,
                                             ui->resizeFactor->value()/100.0));
    } else { return; }

    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::on_image_process()
{
    m_isImageProcessed = true;
    ui->saveProcImage->setEnabled(true);
    ui->useProcImage->setEnabled(true);
}

void ImageProcessor::on_saveProcImage_clicked()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save Image"),
                       "/home/mathius/Documents/CS555/DigitalImageProcessing/images",
                                                    //"",
                       tr("Image Files (*.png *.jpg *.bmp)"));
    if(!filename.endsWith(tr(".jpg"))) filename = filename.append(".jpg");
    m_procIm->save(filename);
}

void ImageProcessor::on_useProcImage_clicked()
{
    m_origIm = std::move(m_procIm);
    resetUI();
}

void ImageProcessor::on_convertToGray_clicked()
{
    m_procIm = std::make_unique<QImage>(m_origIm->convertToFormat(QImage::Format_Grayscale8));
    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::resetUI()
{
    ui->imageLabel->setPixmap(QPixmap::fromImage(*(m_origIm.get())));
    ui->imageLabel->adjustSize();
    m_isImageProcessed = false;
    if(m_origIm->isGrayscale()){
        ui->convertToGray->setEnabled(false);
    }
    ui->saveProcImage->setEnabled(false);
    ui->useProcImage->setEnabled(false);
    ui->resizeFactor->setValue(100);
}

void ImageProcessor::on_applyColorBin_clicked()
{
    if(ui->bitness->value() < 8){
        m_procIm = std::make_unique<QImage>(ImageProcess::colorBin(*(m_origIm.get()), ui->bitness->value()));
        display(m_origIm, m_procIm);
        on_image_process();
    }
}


void ImageProcessor::on_applySpacialFilter_clicked()
{
    // ==== Smoothing =========================================================
    if(QString::compare(ui->spacialFilterComboBox->currentText(),
                        tr("Smoothing")) == 0){
        int dim = ui->spacialSizeSpin->value();
        std::vector<std::vector<double> > kernel = ImageProcess::createKernelSmoothing(dim);
        m_procIm = std::make_unique<QImage>(ImageProcess::convolve(*(m_origIm.get()), kernel));
    }
    // ==== Sharpening Laplacian ==============================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Sharpening Laplacian")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        std::vector<std::vector<double> > kernel = ImageProcess::createKernelLaplacian(dim);
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveLoG(*(m_origIm.get()), kernel));
    }
    // ==== High-Boost ========================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("High-Boost")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        int k = ui->hbConstSpin->value();
        if(k < 0){
            QMessageBox messageBox;
            messageBox.setText("Error: High-Boost filter constant must be greater than or equal to 0.");
            messageBox.setFixedSize(500,200);
            messageBox.exec();
        } else
            m_procIm = std::make_unique<QImage>(ImageProcess::highboost(*(m_origIm.get()), dim, k));
    }
    // ==== Histogram Eq ======================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Histogram Equalization")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::localHistEqualization(*(m_origIm.get()), dim));
    }
    // ==== Median ============================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Median")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                         dim, 0.0, ImageProcess::medianConv));
    }
    // ==== Max ===============================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Max")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                         dim, 0.0, ImageProcess::maxConv));
    }
    // ==== Min ===============================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Min")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                         dim, 0.0, ImageProcess::minConv));
    }
    // ==== Arithmetic Mean ===================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Arithmetic Mean")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                         dim, 0.0, ImageProcess::ameanConv));
    }
    // ==== Geometric Mean ====================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Geometric Mean")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                         dim, 0.0, ImageProcess::gmeanConv));
    }
    // ==== Harmonic Mean =====================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Harmonic Mean")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                         dim, 0.0, ImageProcess::hmeanConv));
    }
    // ==== Contraharmonic Mean ===============================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Contraharmonic Mean")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        double Q = ui->hbConstSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                         dim, Q, ImageProcess::chmeanConv));
    }
    // ==== Midpoint ==========================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Midpoint")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                         dim, 0.0, ImageProcess::midpointConv));
    }
    // ==== Alpha-Trimmed Mean =====================================================
    else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Alpha-Trimmed Mean")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        int d = ui->atSpinBox->value();
        if(d > dim*dim){
            QMessageBox messageBox;
            messageBox.setText("Error: Alpha-trimmed constant must be smaller than the number of pixels in the kernel.");
            messageBox.setFixedSize(500,200);
            messageBox.exec();
        } else
            m_procIm = std::make_unique<QImage>(ImageProcess::convolveHSV(*(m_origIm.get()), NULL,
                                                                dim, d, ImageProcess::atmeanConv));
    }
    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::on_applyBitPlane_clicked()
{
    m_procIm = std::make_unique<QImage>(
                ImageProcess::removeBitPlane(*(m_origIm.get()), ui->bitPlane->value(), ui->bitPlaneZero->isChecked()));
    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::on_spacialSizeSpin_valueChanged(const QString &arg1)
{
    ui->spacialSizeLabel->setText(tr("Kernel Size: ") + arg1 + " x");
}

void ImageProcessor::on_histEqualButton_clicked()
{
    m_procIm = std::make_unique<QImage>(ImageProcess::globalHistEqualization(*(m_origIm.get())));
    display(m_origIm,m_procIm);
    on_image_process();
}

void ImageProcessor::on_createHazeDepth_clicked()
{
    m_procIm = std::make_unique<QImage>(ImageProcess::getHazeDepth(*(m_origIm.get())));
    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::on_applyDehaze_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this,tr("Open Depth Image"),
                       "",
                       tr("Image Files (*.png *.jpg *.bmp)"));
    if(QString::compare(filename, "") != 0){
        QImage tempIm(filename);
        m_procIm = std::make_unique<QImage>(ImageProcess::dehaze(*(m_origIm.get()), tempIm, ui->betaSpin->value()));
    }
    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::on_createDatasetButton_clicked()
{
    QString train_set_folder = QFileDialog::getExistingDirectory(this,tr("Open Training Set Directory"), "");
    if(QString::compare(train_set_folder, "") != 0){
        ImageProcess::createDehazeTrainSet(train_set_folder, ui->betaSpin->value());
    }
}

void ImageProcessor::on_trainDehazeButton_clicked()
{
    QString train_set_folder = QFileDialog::getExistingDirectory(this,tr("Open Training Set Directory"), "");
    if(QString::compare(train_set_folder, "") != 0){
        ImageProcess::trainDehaze(train_set_folder, 517);
    }
}
