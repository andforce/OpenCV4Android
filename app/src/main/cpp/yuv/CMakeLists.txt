# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

set(CMAKE_VERBOSE_MAKEFILE on)

#添加头文件位置
include_directories(${CMAKE_SOURCE_DIR}/../../../../native/jni/include)

#引入libopencv_java4.so文件
add_library(opencv_java4 SHARED IMPORTED)

set_target_properties(opencv_java4
        PROPERTIES
        # ${PROJECT_SOURCE_DIR}
        # ${CMAKE_CURRENT_SOURCE_DIR}
        # ${CMAKE_SOURCE_DIR}
        IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../../native/libs/${ANDROID_ABI}/libopencv_java4.so
        )

message(STATUS ">>>>>>>>>>>>>>>>>>>>>>IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../../native/libs/${ANDROID_ABI}/libopencv_java4.so")

add_library(libopencv_calib3d STATIC IMPORTED )
set_target_properties(libopencv_calib3d PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_calib3d.a")

add_library(libopencv_core STATIC IMPORTED )
set_target_properties(libopencv_core PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_core.a")

add_library(libopencv_dnn STATIC IMPORTED )
set_target_properties(libopencv_dnn PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_dnn.a")

add_library(libopencv_features2d STATIC IMPORTED )
set_target_properties(libopencv_features2d PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_features2d.a")

add_library(libopencv_flann STATIC IMPORTED )
set_target_properties(libopencv_flann PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_flann.a")

add_library(libopencv_highgui STATIC IMPORTED )
set_target_properties(libopencv_highgui PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_highgui.a")

add_library(libopencv_imgcodecs STATIC IMPORTED )
set_target_properties(libopencv_imgcodecs PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_imgcodecs.a")

add_library(libopencv_imgproc STATIC IMPORTED )
set_target_properties(libopencv_imgproc PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_imgproc.a")

add_library(libopencv_ml STATIC IMPORTED )
set_target_properties(libopencv_ml PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_ml.a")

add_library(libopencv_objdetect STATIC IMPORTED )
set_target_properties(libopencv_objdetect PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_objdetect.a")

add_library(libopencv_photo STATIC IMPORTED )
set_target_properties(libopencv_photo PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_photo.a")

#add_library(libopencv_shape STATIC IMPORTED )
#set_target_properties(libopencv_shape PROPERTIES
#        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_shape.a")

add_library(libopencv_stitching STATIC IMPORTED )
set_target_properties(libopencv_stitching PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_stitching.a")

#add_library(libopencv_superres STATIC IMPORTED )
#set_target_properties(libopencv_superres PROPERTIES
#        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_superres.a")

add_library(libopencv_video STATIC IMPORTED )
set_target_properties(libopencv_video PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_video.a")

add_library(libopencv_videoio STATIC IMPORTED )
set_target_properties(libopencv_videoio PROPERTIES
        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_videoio.a")

#add_library(libopencv_videostab STATIC IMPORTED )
#set_target_properties(libopencv_videostab PROPERTIES
#        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_videostab.a")
#
#add_library(libopencv_ts STATIC IMPORTED )
#set_target_properties(libopencv_ts PROPERTIES
#        IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../native/staticlibs/${ANDROID_ABI}/libopencv_ts.a")

add_library( # Sets the name of the library.
        yuv-converter

        # Sets the library as a shared library.
        SHARED

        vector2ArrayList.cpp
        find_objects.cpp
        mat_map_cvt.cpp
        yuv.cpp
        )


# 支持操作Bitmap的库
find_library( # Sets the name of the path variable.
        jnigraphics-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        jnigraphics )

find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.

        yuv-converter
        opencv_java4

        libopencv_calib3d
        libopencv_core
        libopencv_dnn
        libopencv_features2d
        libopencv_flann
        libopencv_highgui
        libopencv_imgcodecs
        libopencv_imgproc
        libopencv_ml
        libopencv_objdetect
        libopencv_photo
        libopencv_stitching
        libopencv_video
        libopencv_videoio
        ###

        # Links the target library to the log library
        # included in the NDK.
        ${jnigraphics-lib}
        ${log-lib}
        )