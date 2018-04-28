#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <QMainWindow>
namespace ImageProcess
{
    //image resize
    QImage nearestNeighbor(QImage im, double xFactor, double yFactor);
    QImage linear(QImage im, double xFactor, double yFactor, bool inXDir = true);
    QImage bilinear(QImage im, double xFactor, double yFactor);

    //color resolution
    QImage colorBin(QImage im, int bitness);
    int colorBin(int value, int bitness);

    //spatial filtering
    std::vector<std::vector<double> > createKernelSmoothing(int dim);
    double laplacianOfGaussian(int x, int y, int dim);
    std::vector<std::vector<double> > createKernelLaplacian(int dim);
    QImage convolve(QImage im, std::vector<std::vector<double> > kernel);
    QImage convolveMedian(QImage im, int dim);
    QImage convolveLoG(QImage im, std::vector<std::vector<double> > kernel);
    QImage highboost(QImage im, int dim, double k);

    //bit plane
    QImage removeBitPlane(QImage im, int plane, bool zeroOut);

    //histogram equalization
    QImage globalHistEqualization(QImage im);
    QImage localHistEqualization(QImage im, int dim);

    //misc
    QImage padImage(QImage im, int padding);
    QImage removePadding(QImage padded, int padding);
    QImage sub(QImage im1, QImage im2);
    int getMedian(std::vector<int> &vals);
}

#endif // IMAGEPROCESS_H
