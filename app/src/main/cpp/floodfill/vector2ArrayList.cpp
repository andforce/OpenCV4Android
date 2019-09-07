//
// Created by 迪远 王 on 2019-09-07.
//

#include "vector2ArrayList.h"

jobject vector2java_util_ArrayList(JNIEnv *env, std::vector<std::string> vector){

    static jclass java_util_ArrayList;
    static jmethodID java_util_ArrayList_;
    jmethodID java_util_ArrayList_size;
    jmethodID java_util_ArrayList_get;
    jmethodID java_util_ArrayList_add;

    java_util_ArrayList = static_cast<jclass>(env->NewGlobalRef(env->FindClass("java/util/ArrayList")));
    java_util_ArrayList_ = env->GetMethodID(java_util_ArrayList, "<init>", "(I)V");
    java_util_ArrayList_size = env->GetMethodID(java_util_ArrayList, "size", "()I");
    java_util_ArrayList_get = env->GetMethodID(java_util_ArrayList, "get", "(I)Ljava/lang/Object;");
    java_util_ArrayList_add = env->GetMethodID(java_util_ArrayList, "add", "(Ljava/lang/Object;)Z");


    jobject result = env->NewObject(java_util_ArrayList, java_util_ArrayList_, vector.size());
    for (std::string s: vector) {
        jstring element = env->NewStringUTF(s.c_str());
        env->CallBooleanMethod(result, java_util_ArrayList_add, element);
        env->DeleteLocalRef(element);
    }
    return result;

}