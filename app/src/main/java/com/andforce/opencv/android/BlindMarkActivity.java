package com.andforce.opencv.android;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

public class BlindMarkActivity extends AppCompatActivity {

    static {
        System.loadLibrary("blind-mark");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_blind_mark);

        BlindMark.blindMark("/sdcard/DCIM/Camera/IMG_20190907_123106.jpg", "AAA");
    }
}
