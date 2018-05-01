#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <QMainWindow>
#include <memory>

namespace ImageProcess
{
    //image resize
    std::unique_ptr<QImage> nearestNeighbor(const std::unique_ptr<QImage> &im, double xFactor, double yFactor);
    std::unique_ptr<QImage> linear(const std::unique_ptr<QImage> &im, double xFactor, double yFactor, bool inXDir = true);
    std::unique_ptr<QImage> bilinear(const std::unique_ptr<QImage> &im, double xFactor, double yFactor);

    //color resolution
    std::unique_ptr<QImage> colorBin(const std::unique_ptr<QImage> &im, int bitness);
    int colorBin(int value, int bitness);

    //spatial filtering
    std::vector<std::vector<double> > createKernelSmoothing(int dim);
    double laplacianOfGaussian(int x, int y, int dim);
    std::vector<std::vector<double> > createKernelLaplacian(int dim);
    std::unique_ptr<QImage> convolve(const std::unique_ptr<QImage> &im, std::vector<std::vector<double> > kernel);
    std::unique_ptr<QImage> convolveMedian(const std::unique_ptr<QImage> &im, int dim);
    std::unique_ptr<QImage> convolveLoG(const std::unique_ptr<QImage> &im, std::vector<std::vector<double> > kernel);
    std::unique_ptr<QImage> highboost(const std::unique_ptr<QImage> &im, int dim, double k);

    //bit plane
    std::unique_ptr<QImage> removeBitPlane(const std::unique_ptr<QImage> &im, int plane, bool zeroOut);

    //histogram equalization
    std::unique_ptr<QImage> globalHistEqualization(const std::unique_ptr<QImage> &im);
    std::unique_ptr<QImage> localHistEqualization(const std::unique_ptr<QImage> &im, int dim);
    std::map<int,int> getNewHistEqValues(const std::unique_ptr<QImage> &im);

    //misc
    std::unique_ptr<QImage> padImage(const std::unique_ptr<QImage> &im, int padding);
    std::unique_ptr<QImage> removePadding(const std::unique_ptr<QImage> &padded, int padding);
    std::unique_ptr<QImage> sub(const std::unique_ptr<QImage> &im1, const std::unique_ptr<QImage> &im2);
    int getMedian(std::vector<int> &vals);
}

#endif // IMAGEPROCESS_H
