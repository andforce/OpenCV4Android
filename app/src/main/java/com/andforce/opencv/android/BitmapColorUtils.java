package com.andforce.opencv.android;

import android.graphics.Bitmap;

public class BitmapColorUtils {

    public static native void convertBitmap2YUV420SP(Bitmap srcBitmap, String dirPath);
}
