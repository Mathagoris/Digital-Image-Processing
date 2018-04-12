#include "imageprocessor.h"
#include "ui_imageprocessor.h"
#include <QFileDialog>
#include <algorithm>
#include <math.h>
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
    m_origIm.load(filename);
//    qDebug() << "red:" << QColor(m_origIm.pixel(0,0)).red();
//    qDebug() << "green:" << QColor(m_origIm.pixel(0,0)).green();
//    qDebug() << "blue:" << QColor(m_origIm.pixel(0,0)).blue();
//    qDebug() << "format:" << m_origIm.isGrayscale();
    //m_origIm = m_origIm.convertToFormat(QImage::Format_Grayscale8);
//    qDebug() << "red:" << QColor(m_origIm.pixel(0,0)).red();
//    qDebug() << "green:" << QColor(m_origIm.pixel(0,0)).green();
//    qDebug() << "blue:" << QColor(m_origIm.pixel(0,0)).blue();
//    qDebug() << "format:" << m_origIm.isGrayscale();
    resetUI();
    ui->processScrollArea->setEnabled(true);
    if(!m_origIm.isGrayscale()) ui->convertToGray->setEnabled(true);
    ui->mainLabel->setText(tr("Process Image:"));
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
        return nearestNeighbor(im, xFactor, yFactor);
    }
    else if(QString::compare(method, tr("Linear X-Direction")) == 0){
        return linear(im, xFactor, yFactor, true);
    }
    else if(QString::compare(method, tr("Linear Y-Direction")) == 0){
        return linear(im, xFactor, yFactor, false);
    }
    else{
        return bilinear(im, xFactor, yFactor);
    }
}

QImage ImageProcessor::nearestNeighbor(QImage im, double xFactor, double yFactor)
{
    int newWidth = std::max(int(floor(im.width()*xFactor)),1);
    int newHeight = std::max(int(floor(im.height()*yFactor)),1);
    QImage procIm(newWidth, newHeight, im.format());
    for(int i = 0; i < newWidth; ++i){
        for(int j = 0; j < newHeight; ++j){
            procIm.setPixel(i,j, im.pixel(std::min(int(round((i+.5)/xFactor-.5)),im.width()-1),
                                          std::min(int(round((j+.5)/yFactor-.5)),im.height()-1)));
        }
    }
    return procIm;
}

QImage ImageProcessor::linear(QImage im, double xFactor, double yFactor, bool inXDir)
{
    int newWidth = std::max(int(floor(im.width()*xFactor)),1);
    int newHeight = std::max(int(floor(im.height()*yFactor)),1);
    QImage procIm(newWidth, newHeight, im.format());
    for(int i = 0; i < newWidth; ++i){
        for(int j = 0; j < newHeight; ++j){
            if(inXDir){
                double xLoc = (i+.5)/xFactor-.5;
                int yLoc = std::min(int(round((j+.5)/yFactor-.5)),im.height()-1);
                double lowX = std::min(floor(xLoc),im.width()-1.0);
                double highX = std::min(ceil(xLoc),im.width()-1.0);
                if(xLoc < 0){
                    procIm.setPixel(i,j,im.pixel(0,yLoc));
                } else if(xLoc > im.width()-1.0) {
                    procIm.setPixel(i,j,im.pixel(im.width()-1,yLoc));
                } else if(lowX == highX){
                    procIm.setPixel(i,j,im.pixel(lowX,yLoc));
                } else {
                    // Should work for RGB and Grayscale
                    // Format: RGB - (R,G,B), Grayscale - (grey, grey, grey)
                    QColor newColor;
                    newColor.setRed(int(QColor(im.pixel(lowX,yLoc)).red() + (xLoc-lowX) *
                                   ((QColor(im.pixel(highX,yLoc)).red() - QColor(im.pixel(lowX,yLoc)).red())/
                                    (highX-lowX))));
                    newColor.setGreen(int(QColor(im.pixel(lowX,yLoc)).green() + (xLoc-lowX) *
                                     ((QColor(im.pixel(highX,yLoc)).green() - QColor(im.pixel(lowX,yLoc)).green())/
                                      (highX-lowX))));
                    newColor.setBlue(int(QColor(im.pixel(lowX,yLoc)).blue() + (xLoc-lowX) *
                                   ((QColor(im.pixel(highX,yLoc)).blue() - QColor(im.pixel(lowX,yLoc)).blue())/
                                    (highX-lowX))));
                    procIm.setPixel(i,j,newColor.rgb());
                }
            } else {
                int xLoc = std::min(int(round((i+.5)/xFactor-.5)),im.width()-1);
                double yLoc = (j+.5)/yFactor-.5;
                double lowY = std::min(floor(yLoc),im.height()-1.0);
                double highY = std::min(ceil(yLoc),im.height()-1.0);
                if(yLoc < 0){
                    procIm.setPixel(i,j,im.pixel(xLoc,0));
                } else if(yLoc > im.height()-1.0) {
                    procIm.setPixel(i,j,im.pixel(xLoc,im.height()-1));
                } else if(lowY == highY){
                    procIm.setPixel(i,j,im.pixel(xLoc,lowY));
                } else {
                    // Should work for RGB and Grayscale
                    // Format: RGB - (R,G,B), Grayscale - (grey, grey, grey)
                    QColor newColor;
                    newColor.setRed(int(QColor(im.pixel(xLoc,lowY)).red() + (yLoc-lowY) *
                                   ((QColor(im.pixel(xLoc,highY)).red() - QColor(im.pixel(xLoc,lowY)).red())/
                                    (highY-lowY))));
                    newColor.setGreen(int(QColor(im.pixel(xLoc,lowY)).green() + (yLoc-lowY) *
                                     ((QColor(im.pixel(xLoc,highY)).green() - QColor(im.pixel(xLoc,lowY)).green())/
                                      (highY-lowY))));
                    newColor.setBlue(int(QColor(im.pixel(xLoc,lowY)).blue() + (yLoc-lowY) *
                                   ((QColor(im.pixel(xLoc,highY)).blue() - QColor(im.pixel(xLoc,lowY)).blue())/
                                    (highY-lowY))));
                    procIm.setPixel(i,j,newColor.rgb());
                }
            }
        }
    }
    return procIm;
}

QImage ImageProcessor::bilinear(QImage im, double xFactor, double yFactor)
{
    int newWidth = std::max(int(floor(im.width()*xFactor)),1);
    int newHeight = std::max(int(floor(im.height()*yFactor)),1);
    QImage procIm(newWidth, newHeight, im.format());
    for(int i = 0; i < newWidth; ++i){
        for(int j = 0; j < newHeight; ++j){
            double xLoc = (i+.5)/xFactor-.5;
            double yLoc = (j+.5)/yFactor-.5;
            if(xLoc < 0){
                xLoc = 0.0;
            } else if(xLoc > im.width()-1.0) {
                xLoc = im.width()-1.0;
            }
            if(yLoc < 0){
                yLoc = 0.0;
            } else if(yLoc > im.height()-1.0) {
                yLoc = im.height()-1.0;
            }
            double lowX = std::min(floor(xLoc),im.width()-1.0);
            double highX = std::min(ceil(xLoc),im.width()-1.0);
            double lowY = std::min(floor(yLoc),im.height()-1.0);
            double highY = std::min(ceil(yLoc),im.height()-1.0);
            if(lowX == highX){
                if(lowX == 0) highX += 1;
                else lowX -= 1;
            }
            if(lowY == highY){
                if(lowY == 0) highY += 1;
                else lowY -= 1;
            }
            double distFromLeft = xLoc - lowX;
            double distFromBot = yLoc - lowY;
            // Should work for RGB and Grayscale
            // Format: RGB - (R,G,B), Grayscale - (grey, grey, grey)
            QColor newColor;
            newColor.setRed(int((1-distFromLeft)*(1-distFromBot)*QColor(im.pixel(lowX,lowY)).red() +
                                 distFromLeft*(1-distFromBot)*QColor(im.pixel(highX,lowY)).red() +
                                 distFromLeft*distFromBot*QColor(im.pixel(highX,highY)).red() +
                                 (1-distFromLeft)*distFromBot*QColor(im.pixel(lowX,highY)).red()));
            newColor.setGreen(int((1-distFromLeft)*(1-distFromBot)*QColor(im.pixel(lowX,lowY)).green() +
                                  distFromLeft*(1-distFromBot)*QColor(im.pixel(highX,lowY)).green() +
                                  distFromLeft*distFromBot*QColor(im.pixel(highX,highY)).green() +
                                  (1-distFromLeft)*distFromBot*QColor(im.pixel(lowX,highY)).green()));
            newColor.setBlue(int((1-distFromLeft)*(1-distFromBot)*QColor(im.pixel(lowX,lowY)).blue() +
                                 distFromLeft*(1-distFromBot)*QColor(im.pixel(highX,lowY)).blue() +
                                 distFromLeft*distFromBot*QColor(im.pixel(highX,highY)).blue() +
                                 (1-distFromLeft)*distFromBot*QColor(im.pixel(lowX,highY)).blue()));
            procIm.setPixel(i,j,newColor.rgb());
        }
    }
    return procIm;
}

QImage ImageProcessor::colorBin(QImage im, int bitness)
{
    QImage procIm(im.width(), im.height(), im.format());
    for(int i = 0; i < procIm.width(); ++i){
        for(int j = 0; j < procIm.height(); ++j){
            QColor newColor;
            newColor.setRed(colorBin(QColor(im.pixel(i,j)).red(), bitness));
            newColor.setGreen(colorBin(QColor(im.pixel(i,j)).green(), bitness));
            newColor.setBlue(colorBin(QColor(im.pixel(i,j)).blue(), bitness));
            procIm.setPixel(i,j, newColor.rgb());
        }
    }
    return procIm;
}

int ImageProcessor::colorBin(int value, int bitness)
{
    int numBins = pow(2,bitness);
    int interval = 256/numBins;
    int bin = value/interval;
    if(bin == 0) return 0;
    else if(bin == numBins-1) return 255;
    else return (bin + .5) * interval;

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
        m_procIm = nearestNeighbor(m_origIm, ui->resizeFactor->value()/100.0,
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
        m_procIm = colorBin(m_origIm, ui->bitness->value());
        display(m_origIm, m_procIm);
        on_image_process();
    }
}
