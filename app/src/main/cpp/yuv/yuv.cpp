//
// Created by andforce on 8/30/18.
//

#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <android/bitmap.h>

#include "mat_map_cvt.h"

using namespace cv;
using namespace std;


Mat range(Mat &img) {
    int m = img.rows;
    int n = img.cols;
    Mat temp = img(Range(1, m - 1), Range(1, n - 1));
    return temp;
}


void saveMat2File(Mat &src, string file) {
    string path = "/storage/emulated/0/Download/" + file;
    //imwrite(path, src);
}

/**
 * mask 需要是CV_8UC1
 * @param mask
 * @return
 */
Mat createAlphaFromMask(Mat &mask) {
    Mat alpha = Mat::zeros(mask.rows, mask.cols, CV_8UC1);
    //Mat gray = Mat::zeros(mask.rows, mask.cols, CV_8UC1);

    //cvtColor(mask, gray, COLOR_RGB2GRAY);

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            alpha.at<uchar>(i, j) = static_cast<uchar>(255 - mask.at<uchar>(i, j));
        }
    }

    return alpha;
}

int addAlpha(Mat &src, Mat &dst, Mat &alpha) {
    if (src.channels() == 4) {
        return -1;
    } else if (src.channels() == 1) {
        cvtColor(src, src, COLOR_GRAY2RGB);
    }

    dst = Mat(src.rows, src.cols, CV_8UC4);

    vector<Mat> srcChannels;
    vector<Mat> dstChannels;
    //分离通道
    split(src, srcChannels);

    dstChannels.push_back(srcChannels[0]);
    dstChannels.push_back(srcChannels[1]);
    dstChannels.push_back(srcChannels[2]);
    //添加透明度通道
    dstChannels.push_back(alpha);
    //合并通道
    merge(dstChannels, dst);

    return 0;
}

Mat removeChannel(Mat &src, int which){
    vector<Mat> channels;
    split(src, channels);

    for (int i = 0; i < channels[which].rows; i++) {
        for (int j = 0; j < channels[which].cols; j++) {
            channels[which].at<uchar>(i, j) = 0;
        }
    }

    Mat dst;
    merge(channels, dst);
    return dst;
}


/**
 * Android Bitmap ARGB
 * OpenCV CV_8UC4 默认是RGBA
 * flood识别的是BGR
 * 所以要把RGBA-->BGR
 */

Mat srcBGR;
Mat maskGray;
int FILLMODE = 1;
int g_nNewMaskVal = 255;



Mat srcMat;
Mat dstMat;
Mat maskMat;

int g_nConnectivity = 4;
bool g_bUseMask = false;


//----------------------
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
 * 以下链接给了我极大的帮助
 * http://blog.csdn.net/chenxiao_ji/article/details/52875199
 * http://www.jianshu.com/p/62e52c4ab5c4
 * 本人也仅仅是将以上作者的版本改成C++版本而已
 * 喝水不忘挖井人，如果有人使用了该库，请不要忘记他们
 * author github:https://github.com/zy445566
 * welcome to star
 */

void shiftDFT(cv::Mat mag) {

    mag = mag(cv::Rect(0, 0, mag.cols & (-2), mag.rows & (-2)));

    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    cv::Mat q0 = cv::Mat(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1 = cv::Mat(mag,  cv::Rect(cx, 0, cx, cy));
    cv::Mat q2 =  cv::Mat(mag,  cv::Rect(0, cy, cx, cy));
    cv::Mat q3 =  cv::Mat(mag,  cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp =  cv::Mat();
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

cv::Mat getBlueChannel(const cv::Mat &image)
{
    //cv::Mat nextImg = image;
    std::vector<cv::Mat> channel;
    split(image,channel);
    return channel[0];
}

cv::Mat getDftMat(const cv::Mat &padded) {
    std::vector<cv::Mat> planes;
    planes.push_back(padded);
    Mat mat = cv::Mat::zeros(padded.size(), CV_32F);
    planes.push_back(mat);
    cv::Mat comImg;
    merge(planes, comImg);
    cv::dft(comImg, comImg);
    return comImg;
}

void addTextByMat(cv::Mat comImg, const cv::String &watermarkText, const cv::Point &point,double fontSize) {

    cv::putText(comImg, watermarkText, point, cv::FONT_HERSHEY_DUPLEX, fontSize, cv::Scalar::all(0),2);
    cv::flip(comImg, comImg, -1);
    cv::putText(comImg, watermarkText, point, cv::FONT_HERSHEY_DUPLEX, fontSize, cv::Scalar::all(0), 2);
    cv::flip(comImg, comImg, -1);
}

cv::Mat transFormMatWithText(const cv::Mat &srcImg, const cv::String &watermarkText,double fontSize) {
    cv::Mat padded=getBlueChannel(srcImg);
    padded.convertTo(padded, CV_32F);
    cv::Mat comImg = getDftMat(padded);
    // add text 
    cv::Point center(padded.cols/2, padded.rows/2);
    addTextByMat(comImg,watermarkText,center,fontSize);
//    cv::Point outer(45, 45);
//    addTextByMat(comImg,watermarkText,outer,fontSize);

    //back image
    cv::Mat invDFT;
    idft(comImg, invDFT, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT, 0);
    cv::Mat restoredImage;
    invDFT.convertTo(restoredImage, CV_8U);
    std::vector<cv::Mat> backPlanes;
    split(srcImg, backPlanes);
    backPlanes.erase(backPlanes.begin());
    backPlanes.insert(backPlanes.begin(), restoredImage);
    cv::Mat backImage;
    cv::merge(backPlanes,backImage);
    return backImage;
}


cv::Mat getTextFormMat(const cv::Mat &backImage) {
    cv::Mat padded = getBlueChannel(backImage);
    padded.convertTo(padded, CV_32F);
    cv::Mat comImg = getDftMat(padded);

    std::vector<cv::Mat> backPlanes;
    // split the comples image in two backPlanes  
    cv::split(comImg, backPlanes);

    cv::Mat mag;
    // compute the magnitude  
    cv::magnitude(backPlanes[0], backPlanes[1], mag);

    // move to a logarithmic scale  
    cv::add(cv::Mat::ones(mag.size(), CV_32F), mag, mag);
    cv::log(mag, mag);
    shiftDFT(mag);
    mag.convertTo(mag, CV_8UC1);
    normalize(mag, mag, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return mag;
}


void transformImageWithText(const String &filename, const String &watermarkText,const String &outfilename) {
    cv::Mat srcImg = cv::imread(filename);
    if (srcImg.empty()) { LOGD("read image failed"); }
    //cv::Mat comImg = transFormMatWithText(srcImg, watermarkText,2.0);

    cv::Mat padded = getBlueChannel(srcImg);
    padded.convertTo(padded, CV_32F);
    cv::Mat comImg = getDftMat(padded);
    // add text
//    cv::Point center(padded.cols / 2, padded.rows / 2);
//    addTextByMat(comImg, watermarkText, center, 20.0);
    cv::Point outer(45, 45);
    addTextByMat(comImg, watermarkText, outer, 40.0);
    //back image
    cv::Mat invDFT;
    idft(comImg, invDFT, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT, 0);
    cv::Mat restoredImage;
    invDFT.convertTo(restoredImage, CV_8U);
    std::vector<cv::Mat> backPlanes;
    split(srcImg, backPlanes);
    backPlanes.erase(backPlanes.begin());
    backPlanes.insert(backPlanes.begin(), restoredImage);
    cv::Mat backImage;
    cv::merge(backPlanes, backImage);

    bool res = cv::imwrite(outfilename, backImage);
}

void getTextFormImage(const String &filename,const String &backfilename) {
    cv::Mat comImg = cv::imread(filename);
    cv::Mat backImage = getTextFormMat(comImg);
    bool res = cv::imwrite(backfilename,backImage);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_andforce_opencv_android_BitmapColorUtils_convertBitmap2YUV420SP(JNIEnv *env, jclass clazz,
                                                                         jobject src_bitmap) {
    Mat srcRGBA;
    BitmapToMat(env, src_bitmap, srcRGBA, CV_8UC4);

    Mat dstRGB;
    cvtColor(srcRGBA, dstRGB, COLOR_RGBA2RGB);

    Mat mergedRGB;
    vector<Mat> channels;
    split(dstRGB, channels);
    for (int i = 0; i < channels[0].rows; i++) {
        for (int j = 0; j < channels[0].cols; j++) {
            channels[0].at<uchar>(i, j) = 0;
        }
    }
    merge(channels, mergedRGB);

    imwrite("/sdcard/convertBitmap2YUV420SP.jpg", mergedRGB);

//    auto *in_buffer = new uint8_t[dstRGB.rows * dstRGB.cols * 3];
//    auto *rgb = new unsigned char[dstRGB.rows * dstRGB.cols];
//    if (dstRGB.isContinuous()) {
//        rgb = dstRGB.data;
//    }
}