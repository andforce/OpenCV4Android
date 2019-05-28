//
// Created by andforce on 8/30/18.
//
#include <jni.h>
#include <string>
#include <vector>

#ifndef OPENCV3_CPP_JAVA_UTILS_H
#define OPENCV3_CPP_JAVA_UTILS_H


class cpp_java_utils {

};


#endif //OPENCV3_CPP_JAVA_UTILS_H

jobject vector2java_util_ArrayList(JNIEnv *env, std::vector<std::string> vector);