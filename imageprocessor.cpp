#include "imageprocessor.h"
#include "ui_imageprocessor.h"
#include "imageprocess.h"
#include <QFileDialog>
#include <qdebug.h>

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

//    QString filename = "/home/mathius/Documents/CS555/DigitalImageProcessing/images/climb.jpg";
//    m_origIm.load(filename);

//    m_origIm.convertToFormat(QImage::Format_Grayscale8);
//    ui->imageLabel->setPixmap(QPixmap::fromImage(m_origIm));
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
        m_origIm.load(filename);
        resetUI();
        ui->processScrollArea->setEnabled(true);
        ui->bitPlaneZero->setEnabled(true);
        if(!m_origIm.isGrayscale()){
            ui->convertToGray->setEnabled(true);
            m_origIm = m_origIm.convertToFormat(QImage::Format_RGB32);
        } else
            m_origIm = m_origIm.convertToFormat(QImage::Format_Grayscale8);
        ui->mainLabel->setText(tr("Process Image:"));
    }
}

void ImageProcessor::display(QImage origIm, QImage procIm)
{
    QImage dispIm(origIm.width() + procIm.width(), std::max(origIm.height(), procIm.height()), origIm.format());

    for(int i = 0; i < origIm.width(); i++){
        for(int j = 0; j < origIm.height(); j++){
            dispIm.setPixel(i,j, origIm.pixel(i,j));
        }
    }
    for(int i = 0; i < procIm.width(); i++){
        for(int j = 0; j < procIm.height(); j++){
            dispIm.setPixel(i+origIm.width(),j,procIm.pixel(i,j));
        }
    }
    ui->imageLabel->setPixmap(QPixmap::fromImage(dispIm));
    ui->imageLabel->adjustSize();
}

QImage ImageProcessor::upsample(QImage im, QString method, double xFactor, double yFactor)
{
    if(QString::compare(method, tr("Nearest Neighbor")) == 0){
        return ImageProcess::nearestNeighbor(im, xFactor, yFactor);
    }
    else if(QString::compare(method, tr("Linear X-Direction")) == 0){
        return ImageProcess::linear(im, xFactor, yFactor, true);
    }
    else if(QString::compare(method, tr("Linear Y-Direction")) == 0){
        return ImageProcess::linear(im, xFactor, yFactor, false);
    }
    else{
        return ImageProcess::bilinear(im, xFactor, yFactor);
    }
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
        m_procIm = upsample(m_origIm, ui->upsampleCombo->currentText(),
                            ui->resizeFactor->value()/100.0,
                            ui->resizeFactor->value()/100.0);
        display(m_origIm, m_procIm);
        on_image_process();
    }
    if(ui->resizeFactor->value() < 100){
        m_procIm = ImageProcess::nearestNeighbor(m_origIm, ui->resizeFactor->value()/100.0,
                                             ui->resizeFactor->value()/100.0);
        display(m_origIm, m_procIm);
        on_image_process();
    }
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
                       //"/home/mathius/Documents/CS555/DigitalImageProcessing/images",
                                                    "",
                       tr("Image Files (*.png *.jpg *.bmp)"));
    if(!filename.endsWith(tr(".jpg"))) filename = filename.append(".jpg");
    m_procIm.save(filename);
}

void ImageProcessor::on_useProcImage_clicked()
{
    m_origIm = m_procIm;
    resetUI();
}

void ImageProcessor::on_convertToGray_clicked()
{
    m_procIm = m_origIm.convertToFormat(QImage::Format_Grayscale8);
    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::resetUI()
{
    ui->imageLabel->setPixmap(QPixmap::fromImage(m_origIm));
    ui->imageLabel->adjustSize();
    m_isImageProcessed = false;
    if(m_origIm.isGrayscale()){
        ui->convertToGray->setEnabled(false);
    }
    ui->saveProcImage->setEnabled(false);
    ui->useProcImage->setEnabled(false);
    ui->resizeFactor->setValue(100);
}

void ImageProcessor::on_applyColorBin_clicked()
{
    if(ui->bitness->value() < 8){
        m_procIm = ImageProcess::colorBin(m_origIm, ui->bitness->value());
        display(m_origIm, m_procIm);
        on_image_process();
    }
}


void ImageProcessor::on_applySpacialFilter_clicked()
{
    if(QString::compare(ui->spacialFilterComboBox->currentText(),
                        tr("Smoothing")) == 0){
        int dim = ui->spacialSizeSpin->value();
        std::vector<std::vector<double> > kernel = ImageProcess::createKernelSmoothing(dim);
        m_procIm = ImageProcess::convolve(m_origIm, kernel);
    } else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Median")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = ImageProcess::convolveMedian(m_origIm, dim);
    } else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Sharpening Laplacian")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        std::vector<std::vector<double> > kernel = ImageProcess::createKernelLaplacian(dim);
        m_procIm = ImageProcess::convolveLoG(m_origIm, kernel);
    } else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("High-Boost")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        int k = ui->hbConstSpin->value();
        m_procIm = ImageProcess::highboost(m_origIm, dim, k);
    } else if(QString::compare(ui->spacialFilterComboBox->currentText(),
                               tr("Histogram Equalization")) == 0) {
        int dim = ui->spacialSizeSpin->value();
        m_procIm = ImageProcess::localHistEqualization(m_origIm, dim);
    }
    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::on_applyBitPlane_clicked()
{
    m_procIm = ImageProcess::removeBitPlane(m_origIm, ui->bitPlane->value(), ui->bitPlaneZero->isChecked());
    display(m_origIm, m_procIm);
    on_image_process();
}

void ImageProcessor::on_spacialSizeSpin_valueChanged(const QString &arg1)
{
    ui->spacialSizeLabel->setText(tr("Kernel Size: ") + arg1 + " x");
}

void ImageProcessor::on_histEqualButton_clicked()
{
    delete m_procIm;
    m_procIm = ImageProcess::globalHistEqualization(m_origIm);
    display(m_origIm,m_procIm);
    on_image_process();
}
