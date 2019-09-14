#!/bin/bash

export ANDROID_HOME=/home/andforce/Android/Sdk
export ANDROID_SDK=${ANDROID_HOME}
export ANDROID_NDK_HOME=/home/andforce/Downloads/android-ndk-r20
export ANDROID_NDK=${ANDROID_NDK_HOME}

CMAKE_ARGS="-DINSTALL_ANDROID_EXAMPLES=OFF \
		-DANDROID_EXAMPLES_WITH_LIBS=OFF \
		-DBUILD_EXAMPLES=OFF \
		-DBUILD_DOCS=OFF \
		-DWITH_OPENCL=OFF \
		-DWITH_IPP=OFF \
		-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake \
		-DANDROID_TOOLCHAIN=clang \
		-DBUILD_ANDROID_PROJECTS=OFF \
		-DBUILD_ANDROID_EXAMPLES=OFF \
		-DBUILD_PREF_TESTS=OFF \
		-DBUILD_TESTS=OFF \
		-DANDROID_SDK_TARGET=21 \
		-DBUILD_ZLIB=ON"
TYPE=$1

if [[ "$1" == "static" ]]; then
        echo "build static"
        CMAKE_ARGS+=" -DANDROID_STL=c++_static"
        echo ${CMAKE_ARGS}
elif [[ "$1" == "shared" ]]; then
        echo "build shared"
        CMAKE_ARGS+=" -DANDROID_STL=c++_shared -DBUILD_SHARED_LIBS=ON"
        echo ${CMAKE_ARGS}
else
        echo "build error"
	return
fi

echo ${CMAKE_ARGS}

mkdir -p build/install

supports=("x86" "x86_64" "arm64-v8a" "armeabi-v7a")

abi_arr=()
i=2
count=$#
let count+=1
while [[ ${i} -lt ${count} ]]
do
  eval arg="\$$i"
  index=${i}
  let index-=2
  abi_arr[${index}]=${arg}
  let i+=1
done

function build() {
    ABI=$1

    echo "Building Opencv for $ABI"
    mkdir build_${TYPE}_${ABI}
    pushd build_${TYPE}_${ABI}

    cmake -DANDROID_ABI=${ABI} \
        ${CMAKE_ARGS} ..

    make -j8
    make install

    cp -r install/* ../build/install/

    popd
}


for var in ${abi_arr[@]};
do
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>$var"
    build ${var}
done

echo "build success"