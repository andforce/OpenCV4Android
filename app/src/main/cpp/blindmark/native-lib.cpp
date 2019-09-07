#include <jni.h>
#include <string>


#include <iostream>
#include <cstdlib>
#include <sstream>
using namespace std;

#include <opencv2/opencv.hpp>

#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#define TAG "andforce_opencv"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG ,__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG ,__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG ,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG ,__VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL,TAG ,__VA_ARGS__)


#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <android/bitmap.h>
#include <opencv2/core.hpp>

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
    imwrite(path, src);
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


//--------------------------------------------------------------------

using namespace cv;
std::vector<Mat> sAllPlanes;
std::vector<Mat> planes;
Mat complexImage;

/**
 * 检查了没问题
 * @param image
 */
void shiftDFT(Mat image) {
    image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
    int cx = image.cols / 2;
    int cy = image.rows / 2;

    Mat q0 = Mat(image, Rect(0, 0, cx, cy));
    Mat q1 = Mat(image, Rect(cx, 0, cx, cy));
    Mat q2 = Mat(image, Rect(0, cy, cx, cy));
    Mat q3 = Mat(image, Rect(cx, cy, cx, cy));

    Mat tmp = Mat();
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/**
 * OK
 * @param image
 * @return
 */
Mat optimizeImageDim(Mat image) {
//    return image;
//    // init
    Mat padded = Mat();
    // get the optimal rows size for dft
    int addPixelRows = getOptimalDFTSize(image.rows);
    // get the optimal cols size for dft
    int addPixelCols = getOptimalDFTSize(image.cols);
    // apply the optimal cols and rows size to the image
    copyMakeBorder(image, padded, 0, addPixelRows - image.rows, 0, addPixelCols - image.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    return padded;
}

/**
 * 检查了没问题
 * @param complexImage
 * @return
 */
Mat createOptimizedMagnitude(Mat complexImage) {
    // init
    std::vector<Mat> newPlanes;
    Mat mag = Mat();
    // split the comples image in two planes
    split(complexImage, newPlanes);
    // compute the magnitude
    magnitude(newPlanes[0], newPlanes[1], mag);

    // move to a logarithmic scale
    add(Mat::ones(mag.size(), CV_32F), mag, mag);
    log(mag, mag);
    // optionally reorder the 4 quadrants of the magnitude image
    shiftDFT(mag);
    // normalize the magnitude image for the visualization since both JavaFX
    // and OpenCV need images with value between 0 and 255
    // convert back to CV_8UC1
    mag.convertTo(mag, CV_8UC1);
    normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1);

    return mag;
}

Mat transformImage(Mat image) {
    // planes数组中存的通道数若开始不为空,需清空.
    if (!planes.empty()) {
        planes.clear();
    }
    // optimize the dimension of the loaded image
    Mat padded = optimizeImageDim(image);
    padded.convertTo(padded, CV_32F);
    // prepare the image planes to obtain the complex image
    planes.push_back(padded);
    Mat empty = Mat::zeros(padded.size(), CV_32F);
    planes.push_back(empty);
    // prepare a complex image for performing the dft
    merge(planes, complexImage);
    // dft
    dft(complexImage, complexImage);
    // optimize the image resulting from the dft operation
    Mat magnitude = createOptimizedMagnitude(complexImage);
    planes.clear();
    return magnitude;
}

//Mat transformImage(Mat image) {
//    // planes??????????????,???.
//    if (!planes.empty()) {
//        planes.clear();
//    }
//    // optimize the dimension of the loaded image
//    Mat padded = optimizeImageDim(image);
//    padded.convertTo(padded, CV_32F);
//    // prepare the image planes to obtain the complex image
//    planes.push_back(padded);
//    Mat empty = Mat::zeros(padded.size(), CV_32F);
//    planes.push_back(empty);
//    // prepare a complex image for performing the dft
//    merge(planes, complexImage);
//    // dft
//    printf("complexImage types %d\n", complexImage.type());
//    dft(complexImage, complexImage);
//
//    // optimize the image resulting from the dft operation
//    Mat magnitude = createOptimizedMagnitude(complexImage);
//    planes.clear();
//    return magnitude;
//}

/**
 * OK
 * @param mat
 * @return
 */
Mat splitSrc(Mat mat) {
    mat = optimizeImageDim(mat);
    vector<Mat> channels;
    split(mat, channels);

//    Mat padded = Mat();
//    if (sAllPlanes.size() > 1) {
//        for (int i = 0; i < sAllPlanes.size(); i++) {
//            padded = sAllPlanes[i];
//            break;
//        }
//    } else {
//        padded = mat;
//    }
    return channels[0];
}

void transformImageWithText(Mat image, String watermarkText, Point point, double fontSize, Scalar scalar) {
    // planes数组中存的通道数若开始不为空,需清空.
    if (planes.size() != 0) {
        planes.clear();
    }
    // optimize the dimension of the loaded image
    //Mat padded = this.optimizeImageDim(image);
    Mat padded = image;
    padded.convertTo(padded, CV_32F);
    // prepare the image planes to obtain the complex image
    planes.push_back(padded);
    Mat empty = Mat::zeros(padded.size(), CV_32F);
    planes.push_back(empty);
    // prepare a complex image for performing the dft
    merge(planes, complexImage);
    // dft
    dft(complexImage, complexImage);
    // 频谱图上添加文本
    putText(complexImage, watermarkText, point, FONT_HERSHEY_DUPLEX, fontSize, scalar,2);
    flip(complexImage, complexImage, -1);
    putText(complexImage, watermarkText, point, FONT_HERSHEY_DUPLEX, fontSize, scalar,2);
    flip(complexImage, complexImage, -1);

    planes.clear();
}

//void transformImageWithText(Mat image, String watermarkText,
//                            Point point, double fontSize, Scalar scalar) {
//    // planes??????????????,???.
//    if (!planes.empty()) {
//        planes.clear();
//    }
//    // optimize the dimension of the loaded image
//    Mat padded = optimizeImageDim(image);
//    padded.convertTo(padded, CV_32F);
//    printf("padded types %d CV_32FC1 %d\n", padded.type(), CV_32F);
//    // prepare the image planes to obtain the complex image
//    planes.push_back(padded);
//    planes.push_back(Mat::zeros(padded.size(), CV_32F));
//    // prepare a complex image for performing the dft
//    merge(planes, complexImage);
//    printf("complexImage types %d\n", complexImage.type());
//    // dft
//    dft(complexImage, complexImage);
//    // ????????
//    putText(complexImage, watermarkText, point, FONT_HERSHEY_DUPLEX, fontSize, scalar, 2);
//    flip(complexImage, complexImage, -1);
//    putText(complexImage, watermarkText, point, FONT_HERSHEY_DUPLEX, fontSize, scalar, 2);
//    flip(complexImage, complexImage, -1);
//
//    planes.clear();
//}

Mat antitransformImage() {
    Mat invDFT = Mat();
    idft(complexImage, invDFT, DFT_SCALE | DFT_REAL_OUTPUT, 0);
    Mat restoredImage = Mat();
    invDFT.convertTo(restoredImage, CV_8U);
    planes.clear();
    return restoredImage;
}

void encImage(Mat img1, String mark){
    Point point(50, 100);
    Scalar scalar(0, 0, 0, 0);

    imwrite("/storage/emulated/0/Download/1_src.jpg", img1);

    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", "encImage");
    //Mat img1 = imread(argv[2], IMREAD_GRAYSCALE);
    transformImageWithText(img1, mark, point, 2.0, scalar);
    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", "transformImageWithText");

    Mat img2 = createOptimizedMagnitude(complexImage);
    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", "createOptimizedMagnitude");

    Mat img3 = antitransformImage();
    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", "antitransformImage");

//    namedWindow("Matrix1", WINDOW_AUTOSIZE);
//    imshow("Matrix1", img1);
//
//    namedWindow("Matrix2", WINDOW_AUTOSIZE);
//    imshow("Matrix2", img2);
//
//    namedWindow("Matrix3", WINDOW_AUTOSIZE);
//    imshow("Matrix3", img3);

    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", "encode+++++++++++++++");
    imwrite("/storage/emulated/0/Download/1_orig.jpg", img1);
    imwrite("/storage/emulated/0/Download/1_watermark.jpg", img2);
    imwrite("/storage/emulated/0/Download/1_result.jpg", img3);
}


void decImage(Mat img1){
    Point point(50, 100);
    Scalar scalar(0, 0, 0, 0);

    //Mat img1 = imread(argv[2], IMREAD_GRAYSCALE);
    transformImage(img1);
    Mat img2 = createOptimizedMagnitude(complexImage);

    Mat img3 = antitransformImage();

//    namedWindow("Matrix1", WINDOW_AUTOSIZE);
//    imshow("Matrix1", img1);
//
//    namedWindow("Matrix2", WINDOW_AUTOSIZE);
//    imshow("Matrix2", img2);

    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", "decode");
    imwrite("/storage/emulated/0/Download/1_decode.jpg", img2);
}

/**
 * OK
 * @param complexImage
 * @param allPlanes
 * @return
 */
Mat antitransformImage(Mat complexImage, vector<Mat> allPlanes) {
//    return antitransformImage();

    Mat invDFT =  Mat();
    idft(complexImage, invDFT, DFT_SCALE | DFT_REAL_OUTPUT, 0);

    Mat restoredImage = Mat();
    invDFT.convertTo(restoredImage, CV_8U);
    allPlanes.erase(allPlanes.begin());
    allPlanes.insert(allPlanes.begin(), restoredImage);
    Mat lastImage = Mat();
    merge(allPlanes, lastImage);
    return lastImage;
}

Mat addImageWatermarkWithText(Mat image, const String &watermarkText){
    Mat complexImage = Mat();

    split(image, sAllPlanes);

    //优化图像的尺寸
//    Mat padded = image;//optimizeImageDim(image);
    Mat padded = splitSrc(image);
    padded.convertTo(padded, CV_32F);
    planes.push_back(padded);
    Mat empty = Mat::zeros(padded.size(), CV_32F);
    planes.push_back(empty);
    merge(planes, complexImage);
    // dft
    dft(complexImage, complexImage);
    // 添加文本水印
    Scalar scalar = Scalar(0, 0, 0);
    Point point =  Point(40, 40);
    putText(complexImage, watermarkText, point, FONT_HERSHEY_DUPLEX, 1.0, scalar);
    flip(complexImage, complexImage, -1);
    putText(complexImage, watermarkText, point, FONT_HERSHEY_DUPLEX, 1.0, scalar);
    flip(complexImage, complexImage, -1);
    return antitransformImage(complexImage, sAllPlanes);
}

/**
 * <pre>
 *     获取图片水印
 * <pre>
 * @author Yangxiaohui
 * @date 2018-10-25 19:58
 * @param image
 */
Mat getImageWatermarkWithText(Mat image){
    vector<Mat> planes;
    Mat complexImage = Mat();
    Mat padded = splitSrc(image);
    padded.convertTo(padded, CV_32F);
    planes.push_back(padded);
    Mat empty = Mat::zeros(padded.size(), CV_32F);
    planes.push_back(empty);
    merge(planes, complexImage);
    // dft
    dft(complexImage, complexImage);
    Mat magnitude = createOptimizedMagnitude(complexImage);
    planes.clear();
    return magnitude;
}

Mat start(Mat src) {
    src.convertTo(src, CV_32F);
//    MatVector planes = new MatVector(2);
    std::vector<Mat> planes;//2
    Mat com =  Mat();
    planes.push_back(src);
    Mat mat = Mat::zeros(src.size(), CV_32F);
    planes.push_back(mat);
    merge(planes, com);
    dft(com, com);
    return com;
}

void addWatermark(Mat com, const String &watermark) {
    Scalar s =  Scalar(0, 0, 0, 0);
    Point p =  Point(com.cols / 3, com.rows / 3);
    putText(com, watermark, p, FONT_HERSHEY_COMPLEX, 1.0, s, 3,8, false);
    flip(com, com, -1);
    putText(com, watermark, p, FONT_HERSHEY_COMPLEX, 1.0, s, 3,8, false);
    flip(com, com, -1);
}


void inverse(Mat com) {
//    std::vector<Mat> planes;
//    //MatVector planes = new MatVector(2);
//    idft(com, com);
//    split(com, planes);

//    com.convertTo(com, CV_8UC1);
//
//    normalize(planes[0], com, 0, 255, NORM_MINMAX, CV_8UC1, NULL);
}

void encode(const String &image, const String &watermark, const String &output) {
    Mat src = imread(image, CV_8S);
    std::vector<Mat> color;// 3
    split(src, color);

    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark----------------- split OK");
    for (int i = 0; i < color.size(); i++) {
        Mat com = start(color[i]);
        addWatermark(com, watermark);
        LOGD("Java_com_andforce_opencv4_BlindMark_blindMark----------------- addMarkOK");
        createOptimizedMagnitude(com);
        color.push_back(com);
    }

    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark----------------- addMarkOK---inverse");

    Mat res =  Mat();
    merge(color, res);

    if (res.rows != src.rows || res.cols != src.cols) {
        res = Mat(res, Rect(0, 0, src.cols & -2, src.rows & -2));
    }

    imwrite(output, res);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_andforce_opencv_android_BlindMark_blindMark(JNIEnv *env, jclass clazz, jstring path_,
                                                     jstring mark_) {
    const char *path = env->GetStringUTFChars(path_, 0);
    const char *mark = env->GetStringUTFChars(mark_, 0);

    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", path);

    Mat image = imread(path, IMREAD_COLOR);
    imwrite("/storage/emulated/0/Download/src.jpg", image);
    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", "start add mark");
    Mat watered = addImageWatermarkWithText(image,mark);
    imwrite("/storage/emulated/0/Download/watered.jpg", watered);
    //encImage(image, mark);

    LOGD("Java_com_andforce_opencv4_BlindMark_blindMark-----------------%s", "get mark");
    Mat water = getImageWatermarkWithText(watered);
    imwrite("/storage/emulated/0/Download/mark_text.jpg", water);

    env->ReleaseStringUTFChars(path_, path);
    env->ReleaseStringUTFChars(mark_, mark);
}


extern "C"
JNIEXPORT void JNICALL
Java_com_andforce_opencv_android_BlindMark_decBlindMark(JNIEnv *env, jclass clazz, jstring path_) {
    const char *path = env->GetStringUTFChars(path_, 0);

    // TODO

    env->ReleaseStringUTFChars(path_, path);
}