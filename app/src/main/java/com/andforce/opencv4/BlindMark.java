package com.andforce.opencv4;

import android.graphics.Bitmap;

public class BlindMark {
    public static native void blindMark(String path, String mark);
    public static native void decBlindMark(String path);
}
