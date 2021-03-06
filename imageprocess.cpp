#include "imageprocess.h"
#include "hashcounter.h"
#include <algorithm>
#include <math.h>
#include <limits>
#include <memory>
#include <random>
#include <dirent.h>
#include <qdebug.h>
#include <bitset>

namespace {
    // bitplane masks
    const unsigned char bit0 = 0b00000001;
    const unsigned char bit1 = 0b00000010;
    const unsigned char bit2 = 0b00000100;
    const unsigned char bit3 = 0b00001000;
    const unsigned char bit4 = 0b00010000;
    const unsigned char bit5 = 0b00100000;
    const unsigned char bit6 = 0b01000000;
    const unsigned char bit7 = 0b10000000;
    const unsigned char bit_planes[] = { bit0, bit1, bit2, bit3, bit4, bit5, bit6, bit7 };

    // dehaze coefficients set to optimal values. Can    also be retrained
    double theta_0 = 0.121779;
    double theta_1 = 0.959710;
    double theta_2 = -0.780245;
    double sigma = 0.041337;

    // rle encoding consts
    const int ESC = 0;
    const int EOL = 0;
    const int EOF = 1;

    struct DepthPixel {
        int i;
        int j;
        double value;
        bool operator<(DepthPixel other) const
        {
            return value > other.value;
        }

    };

    struct FreqNode
    {
        int val;
        int freq;
        FreqNode *left, *right;
        FreqNode(int val, int freq)
        {
            left = right = NULL;
            this->val = val;
            this->freq = freq;
        }
        bool operator<(FreqNode* other)
        {
            return freq < other->freq;
        }
    };

    struct FreqCompare {

        bool operator()(FreqNode* l, FreqNode* r)

        {
            return (l->freq > r->freq);
        }
    };
}

std::vector<QString> read_directory(const QString &dir_name)
{
    std::vector<QString> v;
    DIR* dirp = opendir(dir_name.toStdString().c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        QString str(dp->d_name);
        if(QString::compare(str, ".") != 0 && QString::compare(str, "..") != 0)
            v.push_back(QString(dp->d_name));
    }
    closedir(dirp);
    return v;
}

// ==== IMAGE RESIZE ==============================================================================
//
//
// ================================================================================================

QImage ImageProcess::nearestNeighbor(const QImage &im, double xFactor, double yFactor)
{
    int newWidth = std::max(int(floor(im.width()*xFactor)),1);
    int newHeight = std::max(int(floor(im.height()*yFactor)),1);
    QImage procIm = QImage(newWidth, newHeight, im.format());
    for(int i = 0; i < newWidth; ++i){
        for(int j = 0; j < newHeight; ++j){
            procIm.setPixel(i,j, im.pixel(std::min(int(round((i+.5)/xFactor-.5)),im.width()-1),
                                          std::min(int(round((j+.5)/yFactor-.5)),im.height()-1)));
        }
    }
    return procIm;
}

QImage ImageProcess::linear(const QImage &im, double xFactor, double yFactor, bool inXDir)
{
    int newWidth = std::max(int(floor(im.width()*xFactor)),1);
    int newHeight = std::max(int(floor(im.height()*yFactor)),1);
    QImage procIm = QImage(newWidth, newHeight, im.format());
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

QImage ImageProcess::bilinear(const QImage &im, double xFactor, double yFactor)
{
    int newWidth = std::max(int(floor(im.width()*xFactor)),1);
    int newHeight = std::max(int(floor(im.height()*yFactor)),1);
    QImage procIm = QImage(newWidth, newHeight, im.format());
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

// ==== COLOR RESOLUTION ==========================================================================
//
//
// ================================================================================================

QImage ImageProcess::colorBin(const QImage &im, int bitness)
{
    QImage procIm = QImage(im.width(), im.height(), im.format());
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

int ImageProcess::colorBin(int value, int bitness)
{
    int numBins = pow(2,bitness);
    int interval = 256/numBins;
    int bin = value/interval;
    if(bin == 0) return 0;
    else if(bin == numBins-1) return 255;
    else return (bin + .5) * interval;

}

// ==== SPATIAL FILTERING =========================================================================
//
//
// ================================================================================================


// potentially change to gaussian blur... -___-
std::vector<std::vector<double> > ImageProcess::createKernelSmoothing(int dim)
{
    std::vector<std::vector<double> > kernel(dim, std::vector<double>(dim));
    for(int i = 0; i < dim; ++i){
        for(int j = 0; j < dim; ++j){
            kernel[i][j] = 1.0/(dim*dim);
        }
    }
    return kernel;
}

double ImageProcess::laplacianOfGaussian(int x, int y, int dim)
{
    double sigma = dim/2;
    sigma /= 2.0;
    const double PI = 3.141592653589793;
    double g = 0;
    for(double ySubPixel = y - 0.5; ySubPixel < y + 0.55; ySubPixel += 0.1){
        for(double xSubPixel = x - 0.5; xSubPixel < x + 0.55; xSubPixel += 0.1){
            double s = -((xSubPixel*xSubPixel)+(ySubPixel*ySubPixel))/
                (2*sigma*sigma);
            g = g + (1/(PI*pow(sigma,4)))*
                (1+s)*exp(s);
        }
    }
    g = -g/121;
    return g;
}

std::vector<std::vector<double> > ImageProcess::createKernelLaplacian(int dim)
{
    std::vector<std::vector<double> > kernel(dim, std::vector<double>(dim));
    int kernelMid = dim/2;
    for(int i = -kernelMid; i <= kernelMid; ++i){
        for(int j = -kernelMid; j <= kernelMid; ++j){
            double lap = laplacianOfGaussian(i,j,dim);
            kernel[i+kernelMid][j+kernelMid] = lap;
        }
    }
    return kernel;
}

QImage ImageProcess::convolve(const QImage &im, std::vector<std::vector<double> > kernel)
{
    int padding = kernel.size()/2;
    int kernelMid = kernel.size()/2;
    QImage padded = padImage(im, padding);
    QImage procIm = QImage(im.width(), im.height(), im.format());
    for(int i = 0; i < procIm.width(); ++i){
        for(int j = 0; j < procIm.height(); ++j){
            double redSum = 0;
            double greenSum = 0;
            double blueSum = 0;
            for(int x = -padding; x <= padding; ++x){
                for(int y = -padding; y <= padding; ++y){
                    redSum += QColor(padded.pixel(padding+i+x,padding+j+y)).red() * kernel[kernelMid+x][kernelMid+y];
                    greenSum += QColor(padded.pixel(padding+i+x,padding+j+y)).green() * kernel[kernelMid+x][kernelMid+y];
                    blueSum += QColor(padded.pixel(padding+i+x,padding+j+y)).blue() * kernel[kernelMid+x][kernelMid+y];
                }
            }
            procIm.setPixel(i,j,QColor(redSum, greenSum, blueSum).rgb());
        }
    }
    return procIm;
}

// TODO: Not scaling properly
QImage ImageProcess::convolveLoG(const QImage &im, std::vector<std::vector<double> > kernel)
{
    int padding = kernel.size()/2;
    int kernelMid = kernel.size()/2;
    QImage padded = padImage(im, padding);
    QImage procIm = QImage(im.width(), im.height(), im.format());
    std::vector<std::vector<std::vector<double> > >
            tempIm(im.width(),
                   std::vector<std::vector<double> >(im.height(),
                                                  std::vector<double>(3,0.0)));
    double rMax = std::numeric_limits<double>::min();
    double rMin = std::numeric_limits<double>::max();
    double gMax = std::numeric_limits<double>::min();
    double gMin = std::numeric_limits<double>::max();
    double bMax = std::numeric_limits<double>::min();
    double bMin = std::numeric_limits<double>::max();
    for(int i = 0; i < procIm.width(); ++i){
        for(int j = 0; j < procIm.height(); ++j){
            double redSum = 0;
            double greenSum = 0;
            double blueSum = 0;
            for(int x = -padding; x <= padding; ++x){
                for(int y = -padding; y <= padding; ++y){
                    redSum += QColor(padded.pixel(padding+i+x,padding+j+y)).red() * kernel[kernelMid+x][kernelMid+y];
                    greenSum += QColor(padded.pixel(padding+i+x,padding+j+y)).green() * kernel[kernelMid+x][kernelMid+y];
                    blueSum += QColor(padded.pixel(padding+i+x,padding+j+y)).blue() * kernel[kernelMid+x][kernelMid+y];
                }
            }
            tempIm[i][j][0] = redSum;
            tempIm[i][j][1] = greenSum;
            tempIm[i][j][2] = blueSum;
            if(redSum < rMin) rMin = redSum;
            if(redSum > rMax) rMax = redSum;
            if(greenSum < gMin) gMin = greenSum;
            if(greenSum > gMax) gMax = greenSum;
            if(blueSum < bMin) bMin = blueSum;
            if(blueSum > bMax) bMax = blueSum;
        }
    }
    // Normalize LoG and subtract from original image
    // TODO: Maybe normalize after subtract?
    double rNorm = 255.0/(rMax - rMin);
    double gNorm = 255.0/(gMax - gMin);
    double bNorm = 255.0/(bMax - bMin);
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            int newRed = std::max(int(QColor(im.pixel(i,j)).red() -
                    round((tempIm[i][j][0] - rMin)*rNorm)), 0);
            int newGreen = std::max(int(QColor(im.pixel(i,j)).green() -
                    round((tempIm[i][j][1] - gMin)*gNorm)), 0);
            int newBlue = std::max(int(QColor(im.pixel(i,j)).blue() -
                    round((tempIm[i][j][2] - bMin)*bNorm)), 0);
//            int newRed = round((tempIm[i][j][0] - rMin)*rNorm);
//            int newGreen = round((tempIm[i][j][1] - gMin)*gNorm);
//            int newBlue = round((tempIm[i][j][2] - bMin)*bNorm);
//            int newRed = round((tempIm[i][j][0]));// - rMin)*rNorm);
//            int newGreen = round((tempIm[i][j][1]));// - gMin)*gNorm);
//            int newBlue = round((tempIm[i][j][2]));// - bMin)*bNorm);
            procIm.setPixel(i,j,QColor(newRed, newGreen, newBlue).rgb());
        }
    }
    return procIm;
}

QImage ImageProcess::highboost(const QImage &im, int dim, double k)
{
    std::vector<std::vector<double> > kernel = createKernelSmoothing(dim);
    QImage smooth = convolve(im, kernel);
    QImage mask = sub(im, smooth);
    QImage procIm = QImage(im.width(), im.height(), im.format());
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            QColor newColor;
            newColor.setRed(std::min(int(QColor(im.pixel(i,j)).red() +
                                     k * QColor(mask.pixel(i,j)).red()), 255));
            newColor.setGreen(std::min(int(QColor(im.pixel(i,j)).green() +
                                     k * QColor(mask.pixel(i,j)).green()), 255));
            newColor.setBlue(std::min(int(QColor(im.pixel(i,j)).blue() +
                                     k * QColor(mask.pixel(i,j)).blue()), 255));
            procIm.setPixel(i,j,newColor.rgb());
        }
    }
    return procIm;
}

// ==== Image Restoration =========================================================================
//
//
// ================================================================================================

QImage ImageProcess::convolveHSV(const QImage &im, const std::vector<int> *kernel,
                                  const int dim, const double c,
                      int (*conv)(std::vector<int> &vals, const std::vector<int> *kernel,
                                  const int dim, const double c))
 {
     int padding = dim/2;
     QImage padded = padImage(im, padding);
     QImage procIm = QImage(im.width(), im.height(), im.format());
     for(int i = 0; i < im.width(); ++i){
         for(int j = 0; j < im.height(); ++j){
             std::vector<int> vals(dim*dim);
             for(int x = -padding; x <= padding; ++x){
                 for(int y = -padding; y <= padding; ++y){
                     // if its a color image we need to find the median of the
                     // hue not the median of RGB
                     if(im.format() == QImage::Format_RGB32){
                         vals[dim * (x + padding) + (y + padding)] =
                                 QColor(padded.pixel(padding+i+x,padding+j+y)).toHsv().hue();
                     } else {
                         vals[dim * (x + padding) + (y + padding)] =
                                 QColor(padded.pixel(padding+i+x,padding+j+y)).red();
                     }
                 }
             }
             int newVal = conv(vals, kernel, dim, c);
             QColor pix = QColor(padded.pixel(padding+i,padding+j));
             // if its a color image we need to find the median of the
             // hue not the median of RGB
             if(im.format() == QImage::Format_RGB32) {
                 pix = pix.toHsv();
                 pix.setHsv(newVal, pix.saturation(), pix.value());
                 pix = pix.toRgb();
             }
             else
                 pix.setRgb(newVal, newVal, newVal);
             procIm.setPixel(i,j,pix.rgb());
         }
     }
     return procIm;
 }

 int ImageProcess::medianConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     size_t n = vals.size() / 2;
     std::nth_element(vals.begin(), vals.begin()+n, vals.end());
     return vals[n];
 }

 int ImageProcess::maxConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     return *std::max_element(vals.begin(), vals.end());
 }

 int ImageProcess::minConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     return *std::min_element(vals.begin(), vals.end());
 }

 int ImageProcess::ameanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     int sum = 0;
     for(int i = 0; i < vals.size(); ++i){
         sum += vals[i];
     }
     return round(sum/double(dim*dim));
 }

 int ImageProcess::gmeanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     double logSum = 0.0;
     for(int i = 0; i < vals.size(); ++i){
         logSum += log(vals[i]);
     }
     return round(exp(logSum/(dim*dim)));
 }

 int ImageProcess::hmeanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     double sum = 0.0;
     for(int i = 0; i < vals.size(); ++i){
         sum += 1.0/vals[i];
     }
     return round((dim*dim)/sum);
 }

 int ImageProcess::chmeanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     double nume = 0.0;
     double denom = 0.0;
     for(int i = 0; i < vals.size(); ++i){
         nume += std::pow(vals[i], c+1);
         denom += std::pow(vals[i], c);
     }
     return round(nume/denom);
 }

 int ImageProcess::midpointConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     return round((*std::max_element(vals.begin(), vals.end()) + *std::min_element(vals.begin(), vals.end()))/2.0);
 }

 int ImageProcess::atmeanConv(std::vector<int> &vals, const std::vector<int> *kernel, const int dim, const double c)
 {
     int d = c;
     int sum = 0;
     std::sort(vals.begin(),vals.end());
     for(int i = d/2; i < vals.size()-(d/2); ++i){
         sum += vals[i];
     }
     return round(sum/double((dim*dim)-d));

 }

// ==== BIT PLANE =================================================================================
//
//
// ================================================================================================

QImage ImageProcess::removeBitPlane(const QImage &im, int plane, bool zeroOut)
{
    QImage procIm = QImage(im.width(), im.height(), im.format());
    int bit = bit_planes[plane];
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            QColor newColor;
            if(zeroOut){
                // 1s and 0s to 0s
                newColor.setRed(QColor(im.pixel(i,j)).red() & ~bit);
                newColor.setGreen(QColor(im.pixel(i,j)).green() & ~bit);
                newColor.setBlue(QColor(im.pixel(i,j)).blue() & ~bit);
            } else {
                // 0s and 1s to 1s
                newColor.setRed(QColor(im.pixel(i,j)).red() | bit);
                newColor.setGreen(QColor(im.pixel(i,j)).green() | bit);
                newColor.setBlue(QColor(im.pixel(i,j)).blue() | bit);
            }
            procIm.setPixel(i,j,newColor.rgb());
        }
    }
    return procIm;
}

// ==== HISTOGRAM EQUALIZATION ====================================================================
//
//
// ================================================================================================

QImage ImageProcess::globalHistEqualization(const QImage &im)
{
    std::map<int,int> newValues = getNewHistEqValues(im);
    QImage procIm = QImage(im.width(),im.height(),im.format());
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            QColor pix = QColor(im.pixel(i,j));
            if(im.format() == QImage::Format_RGB32){
                pix = pix.toHsv();
                pix.setHsv(pix.hue(), pix.saturation(), newValues[pix.value()]);
                pix.toRgb();
            } else {
                pix.setRgb(newValues[pix.red()],newValues[pix.red()],newValues[pix.red()]);
            }
            procIm.setPixel(i,j,pix.rgb());
        }
    }
    return procIm;
}

// TODO: maybe change all kernels to work as image
QImage ImageProcess::localHistEqualization(const QImage &im, int dim)
{
    int padding = dim/2;
    QImage padded = padImage(im, padding);
    QImage procIm = QImage(im.width(),im.height(),im.format());
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            QImage kernel = QImage(dim,dim,im.format());
            for(int x = -padding; x <= padding; ++x){
                for(int y = -padding; y <= padding; ++y){
                    kernel.setPixel(padding+x,padding+y, padded.pixel(i+x,j+y));
                }
            }
            std::map<int,int> newValues = getNewHistEqValues(kernel);
            QColor pix = QColor(im.pixel(i,j));
            if(im.format() == QImage::Format_RGB32){
                pix = pix.toHsv();
                pix.setHsv(pix.hue(), pix.saturation(), newValues[pix.value()]);
                pix.toRgb();
            } else {
                pix.setRgb(newValues[pix.red()],newValues[pix.red()],newValues[pix.red()]);
            }
            procIm.setPixel(i,j,pix.rgb());
        }
    }
    return procIm;
}

std::map<int,int> ImageProcess::getNewHistEqValues(const QImage &im)
{
    HashCounter<int> counter;
    double numPixel = im.width() * im.height();
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            if(im.format() == QImage::Format_RGB32)
                counter.increment(QColor(im.pixel(i,j)).toHsv().value());
            else
                counter.increment(QColor(im.pixel(i,j)).red());
        }
    }
    std::vector<double> prob;
    std::map<int, int> newValues;
    double cumulativeProb = 0.0;
    for(std::map<int,int>::iterator it = counter.begin(); it != counter.end(); ++it){
        prob.push_back(it->second/numPixel);
        cumulativeProb += prob[prob.size()-1];
        newValues[it->first] = round(cumulativeProb*255);
    }
    return newValues;
}

// ==== Dehaze ====================================================================================
//
//
// ================================================================================================

QImage ImageProcess::getHazeDepth(const QImage &hazyIm)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, sigma);

    QImage hazeDepth(hazyIm.width(), hazyIm.height(), QImage::Format_Grayscale8);
    for(int i = 0; i < hazyIm.width(); ++i){
        for(int j = 0; j < hazyIm.height(); ++j){
            double depth;
            double noise = distribution(generator);
            QColor newColor;
            QColor pixel = QColor(hazyIm.pixel(i,j));
            depth = std::min(std::max(theta_0 + theta_1 * pixel.valueF() + theta_2 * pixel.saturationF() + noise, 0.0), 1.0);
            newColor.setRgbF(depth,depth,depth);
            hazeDepth.setPixel(i,j,newColor.rgb());
        }
    }
    return hazeDepth;
}

QColor ImageProcess::getAtmosphericLight(const QImage &hazyIm, const QImage &hazeDepth)
{
    int top = round((hazyIm.width()*hazyIm.height()) * 0.001);
    std::vector<DepthPixel> intensities;
    for(int i = 0; i < hazyIm.width(); ++i){
        for(int j = 0; j < hazyIm.height(); ++j){
            intensities.push_back({i,j,QColor(hazeDepth.pixel(i,j)).toHsv().valueF()});
        }
    }
    std::sort(intensities.begin(), intensities.end());
    QColor A = QColor(hazyIm.pixel(intensities[0].i, intensities[0].j)).toHsv();
    for(int x = 1; x < top; ++x){
        if(A.valueF() < intensities[x].value)
            A = QColor(hazyIm.pixel(intensities[x].i, intensities[x].j)).toHsv();
    }
    return A.toRgb();
}

QImage ImageProcess::dehaze(const QImage &hazyIm, const QImage &hazeDepth, const double beta)
{
    QImage procIm = QImage(hazyIm.width(), hazyIm.height(), hazyIm.format());
    QColor A = getAtmosphericLight(hazyIm, hazeDepth);

    for(int i = 0; i < hazyIm.width(); ++i){
        for(int j = 0; j < hazyIm.height(); ++j){
            QColor newColor;
            QColor pixel = QColor(hazyIm.pixel(i,j));
            double r,g,b;
            double t = std::min(std::max(exp(-(beta*QColor(hazeDepth.pixel(i,j)).redF())),0.1),0.9);
            r = std::min(std::max(((pixel.redF()-A.redF())/t)+A.redF(),0.0),1.0);
            g = std::min(std::max(((pixel.greenF()-A.greenF())/t)+A.greenF(),0.0),1.0);
            b = std::min(std::max(((pixel.blueF()-A.blueF())/t)+A.blueF(),0.0),1.0);
            newColor.setRgbF(r,g,b);

            procIm.setPixel(i,j, newColor.rgb());
        }
    }
    return procIm;
}

void ImageProcess::createDehazeTrainSet(const QString dataFolder, double beta)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    std::uniform_real_distribution<double> atm_distribution(0.85,1);
    std::vector<QString> img_filenames = read_directory(dataFolder + "/Haze-Free");

    for (QString filename : img_filenames){
        QImage hazeFree(dataFolder + "/Haze-Free/" + filename);
        QImage hazyIm(hazeFree.width(), hazeFree.height(), hazeFree.format());
        QImage depthIm(hazeFree.width(), hazeFree.height(), QImage::Format_Grayscale8);
        QColor A;
        A.setRgbF(atm_distribution(generator),atm_distribution(generator),atm_distribution(generator));
        for(int i = 0; i < hazyIm.width(); ++i){
            for(int j = 0; j < hazyIm.height(); ++j){
                // Set Depth Image pixel
                double depth = distribution(generator);
                QColor depthColor;
                depthColor.setRgbF(depth,depth,depth);
                depthIm.setPixel(i,j,depthColor.rgb());

                // Set Hazy Image pixel
                QColor newColor;
                QColor pixel = QColor(hazeFree.pixel(i,j));
                double t = exp(-(beta*depth));
                double r,g,b;
                r = std::min(std::max(pixel.redF()*t + A.redF()*(1-t),0.0),1.0);
                g = std::min(std::max(pixel.greenF()*t + A.greenF()*(1-t),0.0),1.0);
                b = std::min(std::max(pixel.blueF()*t + A.blueF()*(1-t),0.0),1.0);

                newColor.setRgbF(r,g,b);
                hazyIm.setPixel(i,j,newColor.rgb());
            }
        }
        QString hazyPath = dataFolder + "/Hazy/";
        QString depthPath = dataFolder + "/Depth/";
        // "####.jpg"
        QString imgNum = filename.right(8);
        hazyIm.save(hazyPath + "hazy" + imgNum);
        depthIm.save(depthPath + "depth" + imgNum);
    }
}

QImage ImageProcess::trainDehaze(const QString dataFolder, const int numIters)
{
    theta_0 = 0.0;
    theta_1 = 1.0;
    theta_2 = -1.0;
    double sum = 0.0, wSum = 0.0, vSum= 0.0, sSum = 0.0;
    std::vector<QString> hazy_img_filenames = read_directory(dataFolder + "/Hazy");
    std::vector<QString> depth_img_filenames = read_directory(dataFolder + "/Depth");
    std::sort(hazy_img_filenames.begin(), hazy_img_filenames.end());
    std::sort(depth_img_filenames.begin(), depth_img_filenames.end());

    int n = hazy_img_filenames.size();

    for(int t = 0; t < numIters; ++t){
        for(int x = 0; x < n; ++x){
            QImage hazyIm(dataFolder + "/Hazy/" + hazy_img_filenames[x]);
            QImage depthIm(dataFolder + "/Depth/" + depth_img_filenames[x]);
            for(int i = 0; i < hazyIm.width(); ++i){
                for(int j = 0; j < hazyIm.height(); ++j){
                    QColor hazyPixel = QColor(hazyIm.pixel(i,j)).toHsv();
                    double temp = QColor(depthIm.pixel(i,j)).redF() - theta_0
                            - theta_1 * hazyPixel.valueF()
                            - theta_2 * hazyPixel.saturationF();
                    wSum += temp;
                    vSum += hazyPixel.valueF() * temp;
                    sSum += hazyPixel.saturationF() * temp;
                    sum += temp * temp;
                }
            }
        }
        sigma = pow(sum/n, 0.5);
        theta_0 += wSum; theta_1 += vSum; theta_2 += sSum;
        qDebug() << "Theta 0: " << theta_0;
        qDebug() << "Theta 1: " << theta_1;
        qDebug() << "Theta 2: " << theta_2;
        qDebug() << "Sigma: " << sigma;
    }
}

// ==== Compression ===============================================================================
//
//
// ================================================================================================

bool ImageProcess::compressImage(const QImage &im, const QString filename, int method)
{
    std::vector<std::bitset<8> > compressedImage;
    switch (method) {
    case 0:
        compressedImage = rleGrayEncode(im);
        break;
    case 1:
        compressedImage = rleBitPlaneEncode(im);
        break;
    case 2:
        compressedImage = huffmanEncode(im);
        break;
    default:
        break;
    }

}

QImage ImageProcess::decompressImage(const QString filename, int method)
{

}

std::vector<bitset<8> > ImageProcess::rleEncode(const QImage &im, const unsigned char plane)
{
    std::vector<std::bitset<8> > encoding;
    int run_count = 0, last_val = -1;
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            int pixel_val = QColor(im.pixel(i,j)).red();
            // sets pixel_val to 1 if the bit in bit plane is 1 0 if not
            // ignore if we are encoding gray image intensities
            if(plane != 0)
                pixel_val = pixel_val & plane > 0 ? 1 : 0;
            if(last_val == pixel_val){
                if(run_count > 254){
                    encoding.push_back(std::bitset<8>(run_count));
                    encoding.push_back(std::bitset<8>(last_val));
                    run_count = 1;
                } else {
                    run_count += 1;
                }
            } else {
                encoding.push_back(std::bitset<8>(run_count));
                encoding.push_back(std::bitset<8>(last_val));
                run_count = 1;
                last_val = pixel_val;
            }
        }
        encoding.push_back(std::bitset<8>(run_count));
        encoding.push_back(std::bitset<8>(last_val));
        encoding.push_back(std::bitset<8>(ESC));
        encoding.push_back(std::bitset<8>(EOL));
    }
    encoding.push_back(std::bitset<8>(ESC));
    encoding.push_back(std::bitset<8>(EOF));
    return encoding;
}

std::vector<bitset<8> > ImageProcess::rleGrayEncode(const QImage &im)
{
    return rleEncode(im, 0);
}

std::vector<bitset<8> > ImageProcess::rleBitPlaneEncode(const QImage &im)
{
    std::vector<bitset<16> > single_plane_encoding;
    std::vector<bitset<16> > full_encoding;
    for(unsigned char plane : bit_planes){
        single_plane_encoding = rleEncode(im, plane);
        single_plane_encoding.pop_back();
        single_plane_encoding.pop_back();
        for(int i = 0; i < single_plane_encoding.size(); ++i)
            full_encoding.push_back(single_plane_encoding[i]);
        single_plane_encoding.clear();
    }
    full_encoding.push_back(std::bitset<8>(ESC));
    full_encoding.push_back(std::bitset<8>(EOF));
    return full_encoding;
}

std::vector<bitset<8> > ImageProcess::huffmanEncode(const QImage &im)
{
    std::vector<FreqNode> probabilities = getHistogram(im);
    std::map<int, std::string> huffTbl = huffmanCodes(probabilities);

}

QImage ImageProcess::rleGrayDecode(const std::vector<bitset<8> > &bits)
{

}

QImage ImageProcess::rleBitPlaneDecode(const std::vector<bitset<8> > &bits)
{

}

QImage ImageProcess::huffmanDecode(const std::vector<bitset<8> > &bits)
{

}

std::map<int, std::string> ImageProcess::huffmanTable(std::vector<FreqNode> &freqs)
{
    std::make_heap(freqs.begin(), freqs.end());
    FreqNode *left, *right, *top;

    // Iterate while size of heap doesn't become 1
    while (freqs.size() != 1) {

        // Extract the two minimum
        // freq items from min heap
        left = freqs.front();
        std::pop_heap(freqs.begin(), freqs.end());
        freqs.pop_back();

        right = freqs.front();
        std::pop_heap(freqs.begin(), freqs.end());
        freqs.pop_back();

        // Create a new internal node with
        // frequency equal to the sum of the
        // two nodes frequencies. Make the
        // two extracted node as left and right children
        // of this new node. Add this node
        // to the min heap -1 is a special value
        // for internal nodes, not used
        top = new FreqNode(-1, left->freq + right->freq);

        top->left = left;
        top->right = right;

        freqs.push_back(top);
        std::push_heap(freqs.begin(), freqs.end());
    }
    std::map<int, std::string> huffTbl;
    // Print Huffman codes using
    // the Huffman tree built above
    makeCodes(minHeap.top(), "", huffTbl);
    return huffTbl;

}

namespace {
    void makeCodes(FreqNode* root, std::string str, std::map<int, std::string> table)
    {
        if (!root)
            return;

        if (root->val != -1)
            table[root->val] = str;

        makeCodes(root->left, str + "0", table);
        makeCodes(root->right, str + "1", table);
    }
}

// ==== MISC ======================================================================================
//
//
// ================================================================================================

QImage ImageProcess::padImage(const QImage &im, int padding)
{
    QImage padded = QImage(im.width()+(2*padding), im.height()+(2*padding), im.format());
    // Pad corners
    for(int i = 0; i < padding; ++i){
        for(int j = 0; j < padding; ++j){
            padded.setPixel(i,j, im.pixel(0,0));
            padded.setPixel(padded.width()-1-i, j, im.pixel(im.width()-1,0));
            padded.setPixel(i, padded.height()-1-j, im.pixel(0,im.height()-1));
            padded.setPixel(padded.width()-1-i, padded.height()-1-j,
                            im.pixel(im.width()-1,im.height()-1));
        }
    }
    // Pad top and bottom edge
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < padding; ++j){
            padded.setPixel(padding+i, j, im.pixel(i,0));
            padded.setPixel(padding+i, padded.height()-1-j, im.pixel(i,im.height()-1));
        }
    }
    // Pad left and right edge
    for(int i = 0; i < padding; ++i){
        for(int j = 0; j < im.height(); ++j){
            padded.setPixel(i,padding + j, im.pixel(0,j));
            padded.setPixel(padded.width()-1-i, padding + j, im.pixel(im.width()-1,j));
        }
    }
    // Fill middle
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            padded.setPixel(padding+i,padding+j, im.pixel(i,j));
        }
    }
    return padded;
}

QImage ImageProcess::removePadding(const QImage &padded, int padding)
{
    QImage unpadded = QImage(padded.width()-(2*padding),padded.height()-(2*padding),padded.format());
    for(int i = 0; i < unpadded.width(); ++i){
        for(int j = 0; j < unpadded.height(); ++j){
            unpadded.setPixel(i,j,padded.pixel(padding+i,padding+j));
        }
    }
    return unpadded;
}

QImage ImageProcess::sub(const QImage &im1,const QImage &im2)
{
    QImage procIm = QImage(im1.width(), im1.height(), im1.format());
    for(int i = 0; i < im1.width(); ++i){
        for(int j = 0; j < im1.height(); ++j){
            QColor newColor;
            newColor.setRed(std::max(QColor(im1.pixel(i,j)).red() -
                                     QColor(im2.pixel(i,j)).red(),0));
            newColor.setGreen(std::max(QColor(im1.pixel(i,j)).green() -
                                     QColor(im2.pixel(i,j)).green(),0));
            newColor.setBlue(std::max(QColor(im1.pixel(i,j)).blue() -
                                     QColor(im2.pixel(i,j)).blue(),0));
            procIm.setPixel(i,j,newColor.rgb());
        }
    }
    return procIm;
}


QImage ImageProcess::subHsv(const QImage &im1, const QImage &im2)
{
    QImage procIm = QImage(im1.width(), im1.height(), im1.format());
    for(int i = 0; i < im1.width(); ++i){
        for(int j = 0; j < im1.height(); ++j){
            QColor newColor;
            newColor = newColor.toHsv();
            int h,s,v;
            QColor color1 = QColor(im1.pixel(i,j)).toHsv();
            QColor color2 = QColor(im2.pixel(i,j)).toHsv();
            h = (color1.hue() - color2.hue()) % 360;
            s = std::max(color1.saturation() - color2.saturation(), 0);
            v = std::max(color1.value() - color2.value(), 0);
            newColor.setHsv(h,s,v);
            procIm.setPixel(i,j,newColor.toRgb().rgb());
        }
    }
    return procIm;
}

QImage ImageProcess::addHsv(const QImage &im1, const QImage &im2)
{
    QImage procIm = QImage(im1.width(), im1.height(), im1.format());
    for(int i = 0; i < im1.width(); ++i){
        for(int j = 0; j < im1.height(); ++j){
            QColor newColor;
            newColor = newColor.toHsv();
            int h,s,v;
            QColor color1 = QColor(im1.pixel(i,j)).toHsv();
            QColor color2 = QColor(im2.pixel(i,j)).toHsv();
            h = (color1.hue() + color2.hue()) % 360;
            s = std::max(color1.saturation() - color2.saturation(), 0);
            v = std::max(color1.value() - color2.value(), 0);
            newColor.setHsv(h,s,v);
            procIm.setPixel(i,j,newColor.toRgb().rgb());
        }
    }
    return procIm;
}

std::vector<FreqNode> ImageProcess::getHistogram(const QImage &im)
{
    HashCounter<int> counter;
    double numPixel = im.width() * im.height();
    for(int i = 0; i < im.width(); ++i){
        for(int j = 0; j < im.height(); ++j){
            if(im.format() == QImage::Format_RGB32)
                counter.increment(QColor(im.pixel(i,j)).toHsv().value());
            else
                counter.increment(QColor(im.pixel(i,j)).red());
        }
    }
    std::vector <Prob> probs;
    for(std::map<int,int>::iterator it = counter.begin(); it != counter.end(); ++it){
        probs.push_back({it->first, double(it->second)/numPixel});
    }
    return probs;
}


