#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H

#include <QMainWindow>
#include <bitset>
#include <memory>

namespace ImageProcess
{
    //image resize
    QImage nearestNeighbor(const QImage &im, double xFactor, double yFactor);
    QImage linear(const QImage &im, double xFactor, double yFactor, bool inXDir = true);
    QImage bilinear(const QImage &im, double xFactor, double yFactor);

    //color resolution
    QImage colorBin(const QImage &im, int bitness);
    int colorBin(int value, int bitness);

    //spatial filtering
    std::vector<std::vector<double> > createKernelSmoothing(int dim);
    double laplacianOfGaussian(int x, int y, int dim);
    std::vector<std::vector<double> > createKernelLaplacian(int dim);
    QImage convolve(const QImage &im, std::vector<std::vector<double> > kernel);
    QImage convolveLoG(const QImage &im, std::vector<std::vector<double> > kernel);
    QImage highboost(const QImage &im, int dim, double k);

    //image restoration
    QImage convolveHSV(const QImage &im, const std::vector<int> *kernel,
                       const int dim, const double c,
           int (*conv)(std::vector<int> &vals, const std::vector<int> *kernel,
                       const int dim, const double c));
    int medianConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);
    int maxConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);
    int minConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);
    int ameanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);
    int gmeanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);
    int hmeanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);
    int chmeanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);
    int midpointConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);
    int atmeanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c);


    //bit plane
    QImage removeBitPlane(const QImage &im, int plane, bool zeroOut);

    //histogram equalization
    QImage globalHistEqualization(const QImage &im);
    QImage localHistEqualization(const QImage &im, int dim);
    std::map<int,int> getNewHistEqValues(const QImage &im);

    //dehaze
    QImage getHazeDepth(const QImage &hazyIm);
    QColor getAtmosphericLight(const QImage &hazyIm, const QImage &hazeDepth);
    QImage dehaze(const QImage &hazyIm, const QImage &hazeDepth, const double beta);
    void createDehazeTrainSet(const QString dataFolder, double beta);
    QImage trainDehaze(const QString dataFolder, const int numIters);

    //compression
    bool compressImage(const QImage &im, const QString filename, int method);
    QImage decompressImage(const QString filename, int method);
    std::vector<bitset<8> > rleEncode(const QImage & im, const unsigned char plane);
    std::vector<bitset<8> > rleGrayEncode(const QImage &im);
    std::vector<bitset<8> > rleBitPlaneEncode(const QImage &im);
    std::vector<bitset<8> > huffmanEncode(const QImage &im);
    QImage rleGrayDecode(const std::vector<bitset<8> > &bits);
    QImage rleBitPlaneDecode(const std::vector<bitset<8> > &bits);
    QImage huffmanDecode(const std::vector<bitset<8> > &bits);

    //misc
    QImage padImage(const QImage &im, int padding);
    QImage removePadding(const QImage &padded, int padding);
    QImage sub(const QImage &im1, const QImage &im2);
    QImage subHsv(const QImage &im1, const QImage &im2);
    QImage addHsv(const QImage &im1, const QImage &im2);
}

#endif // IMAGEPROCESS_H
