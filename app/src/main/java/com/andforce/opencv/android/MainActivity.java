package com.andforce.opencv.android;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

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

        Bitmap bitmap = BitmapFactory.decodeFile("/sdcard/hua.jpg");

        BitmapColorUtils.convertBitmap2YUV420SP(bitmap);
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
