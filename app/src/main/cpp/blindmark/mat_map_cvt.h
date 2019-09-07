//
// Created by andforce on 8/30/18.
//
#include <jni.h>
#include <string>
#include <math.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <android/bitmap.h>

#include <android/bitmap.h>

//#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "error", __VA_ARGS__))
//#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "debug", __VA_ARGS__))

using namespace cv;
using namespace std;

#ifndef OPENCV3_MAT_MAP_CVT_H
#define OPENCV3_MAT_MAP_CVT_H


class mat_map_cvt {

};


#endif //OPENCV3_MAT_MAP_CVT_H

void BitmapToMat2(JNIEnv *env, jobject &bitmap, Mat &mat, jboolean needUnPremultiplyAlpha, int _type);

void BitmapToMat(JNIEnv *env, jobject &bitmap, Mat &mat, int _type);

void MatToBitmap2(JNIEnv *env, Mat &mat, jobject &bitmap, jboolean needPremultiplyAlpha, int _type);

//void MatToBitmap(JNIEnv *env, Mat &mat, jobject &bitmap, int _type);