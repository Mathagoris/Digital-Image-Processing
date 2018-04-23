#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QMainWindow>

namespace Ui {
class ImageProcessor;
}

class ImageProcessor : public QMainWindow
{
    Q_OBJECT

public:
    explicit ImageProcessor(QWidget *parent = 0);
    ~ImageProcessor();

private slots:
    void on_actionOpen_Image_triggered();

    void on_resizeFactor_valueChanged(int arg1);

    void on_applyResize_clicked();

    void on_saveProcImage_clicked();

    void on_useProcImage_clicked();

    void on_convertToGray_clicked();

    void on_applyColorBin_clicked();

    void on_applySpacialFilter_clicked();

    void on_applyBitPlane_clicked();

private:
    void display(QImage origIm, QImage procIm);
    void resetUI();
    void on_image_process();

    //image resize
    QImage upsample(QImage im, QString method, double xFactor, double yFactor);
    QImage nearestNeighbor(QImage im, double xFactor, double yFactor);
    QImage linear(QImage im, double xFactor, double yFactor, bool inXDir = true);
    QImage bilinear(QImage im, double xFactor, double yFactor);

    //color resolution
    QImage colorBin(QImage im, int bitness);
    int colorBin(int value, int bitness);

    //kernel filtering
    std::vector<std::vector<double> > createKernelSmoothing(int dim);
    std::vector<std::vector<double> > createKernelLaplacian(int dim);
    double laplacianOfGaussian(int x, int y, int dim);
    QImage convolve(QImage im, std::vector<std::vector<double> > kernel);
    QImage convolveMedian(QImage im, int dim);
    QImage convolveLoG(QImage im, std::vector<std::vector<double> > kernel);
    int getMedian(std::vector<int> &vals);
    QImage padImage(QImage im, int padding);
    QImage removePadding(QImage padded, int padding);
    QImage highboost(QImage im, int dim, double k);

    //bitplane
    QImage removeBitPlane(QImage im, int plane);

    //misc
    QImage sub(QImage im1, QImage im2);

    QImage m_origIm;
    QImage m_procIm;

    bool m_isImageProcessed = false;

    Ui::ImageProcessor *ui;
};

#endif // IMAGEPROCESSOR_H
