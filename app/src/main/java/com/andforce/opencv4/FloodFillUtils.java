package com.andforce.opencv4;

import android.graphics.Bitmap;

public class FloodFillUtils {
    public static native void floodFill(Bitmap bitmap, int x, int y, int low, int up);

    public static native int[] floodFillBitmapWithMask(Bitmap bitmap, Bitmap maskBitmap, Bitmap resultBitmap, int x, int y, int low, int up);
}
