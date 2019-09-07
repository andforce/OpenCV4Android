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


extern "C" JNIEXPORT jstring JNICALL
Java_com_andforce_opencv_android_MainActivity_stringFromJNI(JNIEnv *env, jobject thiz) {
    std::string hello = "Hello from C++";

    ostringstream oss;

    oss << CV_VERSION << endl;

    std::string hell_opencv = hello + " with " + oss.str();
    LOGI("OpenCV version: %s", CV_VERSION);

    return env->NewStringUTF(hell_opencv.c_str());
}