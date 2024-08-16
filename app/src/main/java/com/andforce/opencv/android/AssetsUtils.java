package com.andforce.opencv.android;

import android.app.Application;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.IOException;
import java.io.InputStream;

public class AssetsUtils {
    public static Bitmap getBitmapFromAssets(Application application, String fileName) {
        try {
            InputStream is = application.getAssets().open(fileName);
            return BitmapFactory.decodeStream(is);
        } catch (IOException e) {
            // ignore
        }
        return null;
    }
}
