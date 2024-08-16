![](https://repository-images.githubusercontent.com/188975362/fce2f300-d188-11e9-8267-57f67f941f51)

# OpenCV4Android
Native OpenCV Library Sample for Android Studio - https://opencv.org/android/

![](https://img.shields.io/badge/OpenCV-v4.10.0-red.svg)
![](https://img.shields.io/badge/Android%20Studio-v2024.1.1-blue.svg)
![](https://img.shields.io/badge/Gradle-v7.5-blue.svg)
![](https://img.shields.io/badge/Android%20Gradle%20Plugin-v7.4.2-blue.svg)


## Upgrade OpenCV
1. Download OpenCV Android SDK from https://opencv.org/releases/
2. Extract the downloaded file: `unzip opencv-4.10.0-android-sdk.zip`
3. Copy the extracted folder `OpenCV-android-sdk/sdk/native` to `native` directory thant in the project.
![img.png](img.png)


## Example
```shell
➜  app tree -L 4
.
└── src
    ├── main
    │   ├── cpp
    │   │   ├── CMakeLists.txt
    │   │   ├── blindmark   # Blind Mark
    │   │   ├── floodfill   # Flood Fill for remove background
    │   │   ├── helloopencv # Hello OpenCV
    │   │   └── yuv         # bitmap to yuv

```


## Others
Android Studio CMake - shared library missing libc++_shared.so? Can CMake bundle this?
https://stackoverflow.com/questions/39620739/android-studio-cmake-shared-library-missing-libc-shared-so-can-cmake-bundle

Adding OpenCV to Native C code through CMake on Android Studio
https://stackoverflow.com/questions/54967251/how-can-i-integrate-opencv-4-0-into-a-pure-c-android-ndk-project

How can I integrate OpenCV 4.0 into a pure C++ Android NDK project?
https://stackoverflow.com/questions/54967251/how-can-i-integrate-opencv-4-0-into-a-pure-c-android-ndk-project

opencv4.1.0+contrib4.1.0+opencl在mac下编译so
https://blog.csdn.net/kkae8643150/article/details/99357273
