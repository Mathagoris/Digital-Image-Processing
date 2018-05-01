#include "imageprocess.h"
#include "hashcounter.h"
#include <algorithm>
#include <math.h>
#include <limits>
#include <memory>

const unsigned char bit0 = 0b00000001;
const unsigned char bit1 = 0b00000010;
const unsigned char bit2 = 0b00000100;
const unsigned char bit3 = 0b00001000;
const unsigned char bit4 = 0b00010000;
const unsigned char bit5 = 0b00100000;
const unsigned char bit6 = 0b01000000;
const unsigned char bit7 = 0b10000000;


// ==== IMAGE RESIZE ==============================================================================
//
//
// ================================================================================================

std::unique_ptr<QImage> ImageProcess::nearestNeighbor(const std::unique_ptr<QImage> &im, double xFactor, double yFactor)
{
    int newWidth = std::max(int(floor(im->width()*xFactor)),1);
    int newHeight = std::max(int(floor(im->height()*yFactor)),1);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(newWidth, newHeight, im->format());
    for(int i = 0; i < newWidth; ++i){
        for(int j = 0; j < newHeight; ++j){
            procIm->setPixel(i,j, im->pixel(std::min(int(round((i+.5)/xFactor-.5)),im->width()-1),
                                          std::min(int(round((j+.5)/yFactor-.5)),im->height()-1)));
        }
    }
    return procIm;
}

std::unique_ptr<QImage> ImageProcess::linear(const std::unique_ptr<QImage> &im, double xFactor, double yFactor, bool inXDir)
{
    int newWidth = std::max(int(floor(im->width()*xFactor)),1);
    int newHeight = std::max(int(floor(im->height()*yFactor)),1);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(newWidth, newHeight, im->format());
    for(int i = 0; i < newWidth; ++i){
        for(int j = 0; j < newHeight; ++j){
            if(inXDir){
                double xLoc = (i+.5)/xFactor-.5;
                int yLoc = std::min(int(round((j+.5)/yFactor-.5)),im->height()-1);
                double lowX = std::min(floor(xLoc),im->width()-1.0);
                double highX = std::min(ceil(xLoc),im->width()-1.0);
                if(xLoc < 0){
                    procIm->setPixel(i,j,im->pixel(0,yLoc));
                } else if(xLoc > im->width()-1.0) {
                    procIm->setPixel(i,j,im->pixel(im->width()-1,yLoc));
                } else if(lowX == highX){
                    procIm->setPixel(i,j,im->pixel(lowX,yLoc));
                } else {
                    // Should work for RGB and Grayscale
                    // Format: RGB - (R,G,B), Grayscale - (grey, grey, grey)
                    QColor newColor;
                    newColor.setRed(int(QColor(im->pixel(lowX,yLoc)).red() + (xLoc-lowX) *
                                   ((QColor(im->pixel(highX,yLoc)).red() - QColor(im->pixel(lowX,yLoc)).red())/
                                    (highX-lowX))));
                    newColor.setGreen(int(QColor(im->pixel(lowX,yLoc)).green() + (xLoc-lowX) *
                                     ((QColor(im->pixel(highX,yLoc)).green() - QColor(im->pixel(lowX,yLoc)).green())/
                                      (highX-lowX))));
                    newColor.setBlue(int(QColor(im->pixel(lowX,yLoc)).blue() + (xLoc-lowX) *
                                   ((QColor(im->pixel(highX,yLoc)).blue() - QColor(im->pixel(lowX,yLoc)).blue())/
                                    (highX-lowX))));
                    procIm->setPixel(i,j,newColor.rgb());
                }
            } else {
                int xLoc = std::min(int(round((i+.5)/xFactor-.5)),im->width()-1);
                double yLoc = (j+.5)/yFactor-.5;
                double lowY = std::min(floor(yLoc),im->height()-1.0);
                double highY = std::min(ceil(yLoc),im->height()-1.0);
                if(yLoc < 0){
                    procIm->setPixel(i,j,im->pixel(xLoc,0));
                } else if(yLoc > im->height()-1.0) {
                    procIm->setPixel(i,j,im->pixel(xLoc,im->height()-1));
                } else if(lowY == highY){
                    procIm->setPixel(i,j,im->pixel(xLoc,lowY));
                } else {
                    // Should work for RGB and Grayscale
                    // Format: RGB - (R,G,B), Grayscale - (grey, grey, grey)
                    QColor newColor;
                    newColor.setRed(int(QColor(im->pixel(xLoc,lowY)).red() + (yLoc-lowY) *
                                   ((QColor(im->pixel(xLoc,highY)).red() - QColor(im->pixel(xLoc,lowY)).red())/
                                    (highY-lowY))));
                    newColor.setGreen(int(QColor(im->pixel(xLoc,lowY)).green() + (yLoc-lowY) *
                                     ((QColor(im->pixel(xLoc,highY)).green() - QColor(im->pixel(xLoc,lowY)).green())/
                                      (highY-lowY))));
                    newColor.setBlue(int(QColor(im->pixel(xLoc,lowY)).blue() + (yLoc-lowY) *
                                   ((QColor(im->pixel(xLoc,highY)).blue() - QColor(im->pixel(xLoc,lowY)).blue())/
                                    (highY-lowY))));
                    procIm->setPixel(i,j,newColor.rgb());
                }
            }
        }
    }
    return procIm;
}

std::unique_ptr<QImage> ImageProcess::bilinear(const std::unique_ptr<QImage> &im, double xFactor, double yFactor)
{
    int newWidth = std::max(int(floor(im->width()*xFactor)),1);
    int newHeight = std::max(int(floor(im->height()*yFactor)),1);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(newWidth, newHeight, im->format());
    for(int i = 0; i < newWidth; ++i){
        for(int j = 0; j < newHeight; ++j){
            double xLoc = (i+.5)/xFactor-.5;
            double yLoc = (j+.5)/yFactor-.5;
            if(xLoc < 0){
                xLoc = 0.0;
            } else if(xLoc > im->width()-1.0) {
                xLoc = im->width()-1.0;
            }
            if(yLoc < 0){
                yLoc = 0.0;
            } else if(yLoc > im->height()-1.0) {
                yLoc = im->height()-1.0;
            }
            double lowX = std::min(floor(xLoc),im->width()-1.0);
            double highX = std::min(ceil(xLoc),im->width()-1.0);
            double lowY = std::min(floor(yLoc),im->height()-1.0);
            double highY = std::min(ceil(yLoc),im->height()-1.0);
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
            newColor.setRed(int((1-distFromLeft)*(1-distFromBot)*QColor(im->pixel(lowX,lowY)).red() +
                                 distFromLeft*(1-distFromBot)*QColor(im->pixel(highX,lowY)).red() +
                                 distFromLeft*distFromBot*QColor(im->pixel(highX,highY)).red() +
                                 (1-distFromLeft)*distFromBot*QColor(im->pixel(lowX,highY)).red()));
            newColor.setGreen(int((1-distFromLeft)*(1-distFromBot)*QColor(im->pixel(lowX,lowY)).green() +
                                  distFromLeft*(1-distFromBot)*QColor(im->pixel(highX,lowY)).green() +
                                  distFromLeft*distFromBot*QColor(im->pixel(highX,highY)).green() +
                                  (1-distFromLeft)*distFromBot*QColor(im->pixel(lowX,highY)).green()));
            newColor.setBlue(int((1-distFromLeft)*(1-distFromBot)*QColor(im->pixel(lowX,lowY)).blue() +
                                 distFromLeft*(1-distFromBot)*QColor(im->pixel(highX,lowY)).blue() +
                                 distFromLeft*distFromBot*QColor(im->pixel(highX,highY)).blue() +
                                 (1-distFromLeft)*distFromBot*QColor(im->pixel(lowX,highY)).blue()));
            procIm->setPixel(i,j,newColor.rgb());
        }
    }
    return procIm;
}

// ==== COLOR RESOLUTION ==========================================================================
//
//
// ================================================================================================

std::unique_ptr<QImage> ImageProcess::colorBin(const std::unique_ptr<QImage> &im, int bitness)
{
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im->width(), im->height(), im->format());
    for(int i = 0; i < procIm->width(); ++i){
        for(int j = 0; j < procIm->height(); ++j){
            QColor newColor;
            newColor.setRed(colorBin(QColor(im->pixel(i,j)).red(), bitness));
            newColor.setGreen(colorBin(QColor(im->pixel(i,j)).green(), bitness));
            newColor.setBlue(colorBin(QColor(im->pixel(i,j)).blue(), bitness));
            procIm->setPixel(i,j, newColor.rgb());
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

std::unique_ptr<QImage> ImageProcess::convolve(const std::unique_ptr<QImage> &im, std::vector<std::vector<double> > kernel)
{
    int padding = kernel.size()/2;
    int kernelMid = kernel.size()/2;
    std::unique_ptr<QImage> padded = padImage(im, padding);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im->width(), im->height(), im->format());
    for(int i = 0; i < procIm->width(); ++i){
        for(int j = 0; j < procIm->height(); ++j){
            double redSum = 0;
            double greenSum = 0;
            double blueSum = 0;
            for(int x = -padding; x <= padding; ++x){
                for(int y = -padding; y <= padding; ++y){
                    redSum += QColor(padded->pixel(padding+i+x,padding+j+y)).red() * kernel[kernelMid+x][kernelMid+y];
                    greenSum += QColor(padded->pixel(padding+i+x,padding+j+y)).green() * kernel[kernelMid+x][kernelMid+y];
                    blueSum += QColor(padded->pixel(padding+i+x,padding+j+y)).blue() * kernel[kernelMid+x][kernelMid+y];
                }
            }
            procIm->setPixel(i,j,QColor(redSum, greenSum, blueSum).rgb());
        }
    }
    return procIm;
}

// TODO: Median filter for color image is very subtle
std::unique_ptr<QImage> ImageProcess::convolveMedian(const std::unique_ptr<QImage> &im, int dim)
{
    int padding = dim/2;
    std::unique_ptr<QImage> padded = padImage(im, padding);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im->width(), im->height(), im->format());
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < im->height(); ++j){
            std::vector<int> vals(dim*dim);
            for(int x = -padding; x <= padding; ++x){
                for(int y = -padding; y <= padding; ++y){
                    // if its a color image we need to find the median of the
                    // hue not the median of RGB
                    if(im->format() == QImage::Format_RGB32){
                        vals[dim * (x + padding) + (y + padding)] =
                                QColor(padded->pixel(padding+i+x,padding+j+y)).toHsv().hue();
                    } else {
                        vals[dim * (x + padding) + (y+ padding)] =
                                QColor(padded->pixel(padding+i+x,padding+j+y)).red();
                    }
                }
            }
            int median = getMedian(vals);
            QColor pix = QColor(padded->pixel(padding+i,padding+j));
            // if its a color image we need to find the median of the
            // hue not the median of RGB
            if(im->format() == QImage::Format_RGB32) {
                pix = pix.toHsv();
                pix.setHsv(median, pix.saturation(), pix.value());
                pix = pix.toRgb();
            }
            else
                pix.setRgb(median, median, median);
            procIm->setPixel(i,j,pix.rgb());
        }
    }
    return procIm;
}

// TODO: Not scaling properly
std::unique_ptr<QImage> ImageProcess::convolveLoG(const std::unique_ptr<QImage> &im, std::vector<std::vector<double> > kernel)
{
    int padding = kernel.size()/2;
    int kernelMid = kernel.size()/2;
    std::unique_ptr<QImage> padded = padImage(im, padding);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im->width(), im->height(), im->format());
    std::vector<std::vector<std::vector<double> > >
            tempIm(im->width(),
                   std::vector<std::vector<double> >(im->height(),
                                                  std::vector<double>(3,0.0)));
    double rMax = std::numeric_limits<double>::min();
    double rMin = std::numeric_limits<double>::max();
    double gMax = std::numeric_limits<double>::min();
    double gMin = std::numeric_limits<double>::max();
    double bMax = std::numeric_limits<double>::min();
    double bMin = std::numeric_limits<double>::max();
    for(int i = 0; i < procIm->width(); ++i){
        for(int j = 0; j < procIm->height(); ++j){
            double redSum = 0;
            double greenSum = 0;
            double blueSum = 0;
            for(int x = -padding; x <= padding; ++x){
                for(int y = -padding; y <= padding; ++y){
                    redSum += QColor(padded->pixel(padding+i+x,padding+j+y)).red() * kernel[kernelMid+x][kernelMid+y];
                    greenSum += QColor(padded->pixel(padding+i+x,padding+j+y)).green() * kernel[kernelMid+x][kernelMid+y];
                    blueSum += QColor(padded->pixel(padding+i+x,padding+j+y)).blue() * kernel[kernelMid+x][kernelMid+y];
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
    double rNorm = 255.0/(rMax - rMin);
    double gNorm = 255.0/(gMax - gMin);
    double bNorm = 255.0/(bMax - bMin);
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < im->height(); ++j){
            int newRed = std::max(int(QColor(im->pixel(i,j)).red() -
                    round((tempIm[i][j][0] - rMin)*rNorm)), 0);
            int newGreen = std::max(int(QColor(im->pixel(i,j)).green() -
                    round((tempIm[i][j][1] - gMin)*gNorm)), 0);
            int newBlue = std::max(int(QColor(im->pixel(i,j)).blue() -
                    round((tempIm[i][j][2] - bMin)*bNorm)), 0);
//            int newRed = round((tempIm[i][j][0] - rMin)*rNorm);
//            int newGreen = round((tempIm[i][j][1] - gMin)*gNorm);
//            int newBlue = round((tempIm[i][j][2] - bMin)*bNorm);
            procIm->setPixel(i,j,QColor(newRed, newGreen, newBlue).rgb());
        }
    }
    return procIm;
}

std::unique_ptr<QImage> ImageProcess::highboost(const std::unique_ptr<QImage> &im, int dim, double k)
{
    std::vector<std::vector<double> > kernel = createKernelSmoothing(dim);
    std::unique_ptr<QImage> smooth = convolve(im, kernel);
    std::unique_ptr<QImage> mask = sub(im, smooth);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im->width(), im->height(), im->format());
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < im->height(); ++j){
            QColor newColor;
            newColor.setRed(std::min(int(QColor(im->pixel(i,j)).red() +
                                     k * QColor(mask->pixel(i,j)).red()), 255));
            newColor.setGreen(std::min(int(QColor(im->pixel(i,j)).green() +
                                     k * QColor(mask->pixel(i,j)).green()), 255));
            newColor.setBlue(std::min(int(QColor(im->pixel(i,j)).blue() +
                                     k * QColor(mask->pixel(i,j)).blue()), 255));
            procIm->setPixel(i,j,newColor.rgb());
        }
    }
    return procIm;
}

// ==== BIT PLANE =================================================================================
//
//
// ================================================================================================

std::unique_ptr<QImage> ImageProcess::removeBitPlane(const std::unique_ptr<QImage> &im, int plane, bool zeroOut)
{
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im->width(), im->height(), im->format());
    int bit;
    switch (plane) {
    case 0:
        bit = bit0;
        break;
    case 1:
        bit = bit1;
        break;
    case 2:
        bit = bit2;
        break;
    case 3:
        bit = bit3;
        break;
    case 4:
        bit = bit4;
        break;
    case 5:
        bit = bit5;
        break;
    case 6:
        bit = bit6;
        break;
    default:
        bit = bit7;
        break;
    }
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < im->height(); ++j){
            QColor newColor;
            if(zeroOut){
                // 1s and 0s to 0s
                newColor.setRed(QColor(im->pixel(i,j)).red() & ~bit);
                newColor.setGreen(QColor(im->pixel(i,j)).green() & ~bit);
                newColor.setBlue(QColor(im->pixel(i,j)).blue() & ~bit);
            } else {
                // 0s and 1s to 1s
                newColor.setRed(QColor(im->pixel(i,j)).red() | bit);
                newColor.setGreen(QColor(im->pixel(i,j)).green() | bit);
                newColor.setBlue(QColor(im->pixel(i,j)).blue() | bit);
            }
            procIm->setPixel(i,j,newColor.rgb());
        }
    }
    return procIm;
}

// ==== HISTOGRAM EQUALIZATION ====================================================================
//
//
// ================================================================================================

std::unique_ptr<QImage> ImageProcess::globalHistEqualization(const std::unique_ptr<QImage> &im)
{
    std::map<int,int> newValues = getNewHistEqValues(im);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im->width(),im->height(),im->format());
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < im->height(); ++j){
            QColor pix = QColor(im->pixel(i,j));
            if(im->format() == QImage::Format_RGB32){
                pix = pix.toHsv();
                pix.setHsv(pix.hue(), pix.saturation(), newValues[pix.value()]);
                pix.toRgb();
            } else {
                pix.setRgb(newValues[pix.red()],newValues[pix.red()],newValues[pix.red()]);
            }
            procIm->setPixel(i,j,pix.rgb());
        }
    }
    return procIm;
}

// TODO: maybe change all kernels to work as image
std::unique_ptr<QImage> ImageProcess::localHistEqualization(const std::unique_ptr<QImage> &im, int dim)
{
    int padding = dim/2;
    std::unique_ptr<QImage> padded = padImage(im, padding);
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im->width(),im->height(),im->format());
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < im->height(); ++j){
            std::unique_ptr<QImage> kernel = std::make_unique<QImage>(dim,dim,im->format());
            for(int x = -padding; x <= padding; ++x){
                for(int y = -padding; y <= padding; ++y){
                    kernel->setPixel(padding+x,padding+y, padded->pixel(i+x,j+y));
                }
            }
            std::map<int,int> newValues = getNewHistEqValues(kernel);
            QColor pix = QColor(im->pixel(i,j));
            if(im->format() == QImage::Format_RGB32){
                pix = pix.toHsv();
                pix.setHsv(pix.hue(), pix.saturation(), newValues[pix.value()]);
                pix.toRgb();
            } else {
                pix.setRgb(newValues[pix.red()],newValues[pix.red()],newValues[pix.red()]);
            }
            procIm->setPixel(i,j,pix.rgb());
        }
    }
    return procIm;
}

std::map<int,int> ImageProcess::getNewHistEqValues(const std::unique_ptr<QImage> &im)
{
    HashCounter<int> counter;
    double numPixel = im->width() * im->height();
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < im->height(); ++j){
            if(im->format() == QImage::Format_RGB32)
                counter.increment(QColor(im->pixel(i,j)).toHsv().value());
            else
                counter.increment(QColor(im->pixel(i,j)).red());
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

// ==== MISC ======================================================================================
//
//
// ================================================================================================

std::unique_ptr<QImage> ImageProcess::padImage(const std::unique_ptr<QImage> &im, int padding)
{
    std::unique_ptr<QImage> padded = std::make_unique<QImage>(im->width()+(2*padding),
                                                              im->height()+(2*padding),
                                                              im->format());
    // Pad corners
    for(int i = 0; i < padding; ++i){
        for(int j = 0; j < padding; ++j){
            padded->setPixel(i,j, im->pixel(0,0));
            padded->setPixel(padded->width()-1-i, j, im->pixel(im->width()-1,0));
            padded->setPixel(i, padded->height()-1-j, im->pixel(0,im->height()-1));
            padded->setPixel(padded->width()-1-i, padded->height()-1-j,
                            im->pixel(im->width()-1,im->height()-1));
        }
    }
    // Pad top and bottom edge
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < padding; ++j){
            padded->setPixel(padding+i, j, im->pixel(i,0));
            padded->setPixel(padding+i, padded->height()-1-j, im->pixel(i,im->height()-1));
        }
    }
    // Pad left and right edge
    for(int i = 0; i < padding; ++i){
        for(int j = 0; j < im->height(); ++j){
            padded->setPixel(i,padding + j, im->pixel(0,j));
            padded->setPixel(padded->width()-1-i, padding + j, im->pixel(im->width()-1,j));
        }
    }
    // Fill middle
    for(int i = 0; i < im->width(); ++i){
        for(int j = 0; j < im->height(); ++j){
            padded->setPixel(padding+i,padding+j, im->pixel(i,j));
        }
    }
    return padded;
}

std::unique_ptr<QImage> ImageProcess::removePadding(const std::unique_ptr<QImage> &padded, int padding)
{
    std::unique_ptr<QImage> unpadded = std::make_unique<QImage>(padded->width()-(2*padding),
                                                                padded->height()-(2*padding),
                                                                padded->format());
    for(int i = 0; i < unpadded->width(); ++i){
        for(int j = 0; j < unpadded->height(); ++j){
            unpadded->setPixel(i,j,padded->pixel(padding+i,padding+j));
        }
    }
    return unpadded;
}

std::unique_ptr<QImage> ImageProcess::sub(const std::unique_ptr<QImage> &im1,const std::unique_ptr<QImage> &im2)
{
    std::unique_ptr<QImage> procIm = std::make_unique<QImage>(im1->width(), im1->height(), im1->format());
    for(int i = 0; i < im1->width(); ++i){
        for(int j = 0; j < im1->height(); ++j){
            QColor newColor;
            newColor.setRed(std::max(QColor(im1->pixel(i,j)).red() -
                                     QColor(im2->pixel(i,j)).red(),0));
            newColor.setGreen(std::max(QColor(im1->pixel(i,j)).green() -
                                     QColor(im2->pixel(i,j)).green(),0));
            newColor.setBlue(std::max(QColor(im1->pixel(i,j)).blue() -
                                     QColor(im2->pixel(i,j)).blue(),0));
            procIm->setPixel(i,j,newColor.rgb());
        }
    }
    return procIm;
}

int ImageProcess::getMedian(std::vector<int> &vals)
{
    size_t n = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin()+n, vals.end());
    return vals[n];
}
