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

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_andforce_opencv_android_FloodFillUtils_floodFillBitmapWithMask(JNIEnv *env, jclass type,
                                                             jobject bitmap, jobject maskBitmap,
                                                             jobject resultBitmap, jint x, jint y,
                                                             jint low, jint up) {

    LOGD("start()");
    // 把Bitmap转成Mat
    Mat srcRGBA;
    BitmapToMat(env, bitmap, srcRGBA, CV_8UC4);
    Mat maskRGBA;
//    BitmapToMat(env, maskBitmap, maskRGBA, CV_8UC4);

    //转换成BGR
    cvtColor(srcRGBA, srcBGR, COLOR_RGBA2BGR);
//    cvtColor(maskRGBA,maskGray,COLOR_RGBA2BGR);


    int lowDifference = FILLMODE == 0 ? 0 : low;
    int UpDifference = FILLMODE == 0 ? 0 : up;

    int b = (unsigned) 0;
    int g = (unsigned) 0;
    int r = (unsigned) 255;


    Rect fillRect;
    Scalar newVal = Scalar(b, g, r);
    Point mSeedPoint = Point(x, y);
//    int area = floodFill(srcBGR, mSeedPoint, newVal, &fillRect,
//                     Scalar(lowDifference, lowDifference, lowDifference),
//                     Scalar(UpDifference, UpDifference, UpDifference), flags);

    maskGray.create(srcBGR.rows + 2, srcBGR.cols + 2, CV_8UC1);
    maskGray = Scalar::all(0);

    //threshold(maskGray, maskGray, 1, 128, THRESH_BINARY);
    int flags = 4 | FLOODFILL_MASK_ONLY | FLOODFILL_FIXED_RANGE | (g_nNewMaskVal << 8);
    //g_nConnectivity + (g_nNewMaskVal << 8) + (FILLMODE == 1 ? FLOODFILL_FIXED_RANGE : 0);


    //    InputOutputArray:输入和输出图像。
    //    mask:            输入的掩码图像。
    //    seedPoint：      算法开始处理的开始位置。
    //    newVal：         图像中所有被算法选中的点，都用这个数值来填充。
    //    rect:            最小包围矩阵。
    //    loDiff：         最大的低亮度之间的差异。
    //    upDiff：         最大的高亮度之间的差异。
    //    flag：           选择算法连接方式。
    int area = floodFill(srcBGR, maskGray, mSeedPoint, newVal, &fillRect,
                         Scalar(lowDifference, lowDifference, lowDifference),
                         Scalar(UpDifference, UpDifference, UpDifference), flags);


    Mat sizeCorrect;
    sizeCorrect = range(maskGray);
    Mat alpha = createAlphaFromMask(sizeCorrect);
    Mat resultMat;
    addAlpha(srcBGR, resultMat, alpha);
    saveMat2File(resultMat, "mergedResultRGBA.png");

//    Mat resultMatRGBA;
//    cvtColor(resultMat,resultMatRGBA,COLOR_BGRA2RGBA);
//
//    saveMat2File(resultMatRGBA, "resultMatRGBA.png");

    //resultMatRGBA = removeChannel(resultMatRGBA, 3);
    //MatToBitmap2(env, resultMatRGBA, resultBitmap, static_cast<jboolean>(false), CV_8UC4);


    // 把mask转成Bitmap返回到Java层
    cvtColor(maskGray, maskRGBA, COLOR_GRAY2RGBA);

    sizeCorrect = range(maskRGBA);

    MatToBitmap2(env, sizeCorrect, maskBitmap, static_cast<jboolean>(false), CV_8UC4);

    // 返回int[]
    int size = resultMat.rows * resultMat.cols;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, (jint *)resultMat.data);
    //env->ReleaseIntArrayElements(buf_, buf, 0);
    return result;
}


Mat srcMat;
Mat dstMat;
Mat maskMat;

int g_nConnectivity = 4;
bool g_bUseMask = false;

extern "C"
JNIEXPORT void JNICALL
Java_com_andforce_opencv_android_FloodFillUtils_floodFill(JNIEnv *env, jobject instance, jobject bitmap,
                                               jint x, jint y, jint low, jint up) {

    BitmapToMat(env, bitmap, srcMat, CV_8UC4);

    Mat bgra;
    Mat bgr;

    //转换成BGRA
    cvtColor(srcMat, bgra, COLOR_RGBA2BGRA);
    //转换成BGR
    cvtColor(srcMat, bgr, COLOR_RGBA2BGR);

    srcMat = bgr;

    string path = "/storage/emulated/0/Download/src1.jpg";
    //imwrite(path, srcMat);

//    MatToBitmap(env, srcMat, bitmap, CV_8UC3);
//
//    if (true){
//        return;
//    }

    srcMat.copyTo(dstMat);


    string dstPath = "/storage/emulated/0/Download/dst.jpg";
//    imwrite(dstPath, dstMat);

    maskMat.create(srcMat.rows + 2, srcMat.cols + 2, CV_8UC1);

    Point mSeedPoint = Point(x, y);

    LOGD("start find-----------------%d, %d", x, y);

    int lowDifference = FILLMODE == 0 ? 0 : low;
    int UpDifference = FILLMODE == 0 ? 0 : up;

//    int b = (unsigned) theRNG() & 255;
//    int g = (unsigned) theRNG() & 255;
//    int r = (unsigned) theRNG() & 255;

    int b = (unsigned) 0;
    int g = (unsigned) 0;
    int r = (unsigned) 255;


    Rect ccomp;
    Scalar newVal = Scalar(b, g, r);
    Mat dst = dstMat;//目标图的赋值

    //int flags = g_nConnectivity + (g_nNewMaskVal << 8) + (FILLMODE == 1 ? FLOODFILL_FIXED_RANGE : 0);

    int flags = 4 | FLOODFILL_FIXED_RANGE | (g_nNewMaskVal << 8);

    int area;
    if (g_bUseMask) {


        threshold(maskMat, maskMat, 1, 128, THRESH_BINARY);
        area = floodFill(dst, maskMat, mSeedPoint, newVal, &ccomp,
                         Scalar(lowDifference, lowDifference, lowDifference),
                         Scalar(UpDifference, UpDifference, UpDifference), flags);

    } else {
        LOGD("start find-----------------floodFill flags: %d", flags);

        area = floodFill(dst, mSeedPoint, newVal, &ccomp,
                         Scalar(lowDifference, lowDifference, lowDifference),
                         Scalar(UpDifference, UpDifference, UpDifference), flags);

        string path = "/storage/emulated/0/Download/555.jpg";
//        imwrite(path, dst);
    }

    LOGD("有多少个点被重画-----------------%d", area);

    Mat show;
    cvtColor(dst, dst, COLOR_BGR2RGBA);

    MatToBitmap2(env, dst, bitmap, static_cast<jboolean>(true), CV_8UC4);

//    Mat mat_image_src ;
//    BitmapToMat(env,bitmap,mat_image_src);//图片转化成mat
//    Mat mat_image_dst;
//    blur(mat_image_src, mat_image_dst, Size2i(10,10));
//    //第四步：转成java数组->更新
//    MatToBitmap(env,mat_image_dst,bitmap);//mat转成化图片

}


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
Java_com_andforce_opencv4_FloodFillUtils_blindWaterMark(JNIEnv *env, jclass type) {

    transformImageWithText("/sdcard/src.jpeg", "HHH", "/sdcard/result.jpeg");

    getTextFormImage("/sdcard/result.jpeg", "/sdcard/mark.jpeg");
    // TODO

}