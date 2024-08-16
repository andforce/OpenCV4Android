package com.andforce.opencv.android;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("hello_opencv");
        System.loadLibrary("yuv_converter");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());
        // 从Assets中读取图片，转成bitmap
        Bitmap bitmap = AssetsUtils.getBitmapFromAssets(this.getApplication(), "flower.jpg");

        String path = Objects.requireNonNull(getExternalFilesDir("opencv")).getAbsolutePath();
        BitmapColorUtils.convertBitmap2YUV420SP(bitmap, path);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public void showFloodFill(View view) {
        Intent intent = new Intent(this, FloodFillActivity.class);
        startActivity(intent);
    }

    public void showBlindMark(View view) {
        Intent intent = new Intent(this, BlindMarkActivity.class);
        startActivity(intent);
    }
}
