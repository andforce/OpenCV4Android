#include <jni.h>
#include <string>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include "cpp_java_utils.h"

#include <stdlib.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc/types_c.h"

using namespace cv;
using namespace std;

#define TAG "FIND_OBJ"
#define LOGD(...) __android_log_print(ANDROID_LOG_ERROR,TAG ,__VA_ARGS__)

bool isNormal(const vector<vector<Point>> &contours, int i);



extern "C"
JNIEXPORT jobject JNICALL
Java_com_andforce_opencv4_MainActivity_find_1objects(JNIEnv *env, jclass type, jstring imagePath_) {
    const char *imagePath = env->GetStringUTFChars(imagePath_, 0);

    LOGD("start find-----------------%s", imagePath);

    Mat image = imread(imagePath);
    Mat dst, dst1;
    cvtColor(image, dst, COLOR_BGR2BGRA);


    vector<Mat> channels;
    split(dst, channels);

    for (int i = 0; i < channels[3].rows; i++) {
        for (int j = 0; j < channels[3].cols; j++) {
            channels[3].at<uchar>(i, j) = 0;
        }
    }

    merge(channels, dst1);

    //imwrite("/Users/apple/Desktop/dst1.png", dst1);

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Mat thresh;
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    imwrite("/storage/emulated/0/Download/bin.png", thresh);

    morphologyEx(thresh, thresh, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(15, 15)),
                 Point(-1, -1), 2);
    //morphologyEx(thresh, thresh, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1,-1), 2);

    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    LOGD("contours.size() = %u\n", contours.size());

    int count = 0;
    vector<string> result;
    for (int i = 0; i < contours.size(); i++) {
        LOGD("contourArea = %f\n", contourArea(contours[i]));
        if (isNormal(contours, i)) {
            continue;
        }

        for (int k = 0; k < dst1.rows; k++) {
            for (int j = 0; j < dst1.cols; j++) {
                if (pointPolygonTest(contours[i], Point2f(j, k), false) > 0) {
                    dst1.at<Vec4b>(k, j)[3] = 255;
                }

            }
        }

        Rect rect = boundingRect(contours[i]);
        //rectangle(image, rect, Scalar(0, 0, 255), 1);

        ostringstream stream;
        stream << i;
        count++;
        string path = "/storage/emulated/0/Download/" + stream.str() + ".png";
        result.push_back(path);

        imwrite(path, dst1(rect));
    }

    env->ReleaseStringUTFChars(imagePath_, imagePath);

    return vector2java_util_ArrayList(env, result);
}

bool isNormal(const vector<vector<Point>> &contours, int i) {
    return contourArea(contours[i]) < 1000;
}

