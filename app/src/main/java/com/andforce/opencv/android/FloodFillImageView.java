package com.andforce.opencv.android;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.widget.ImageView;

import androidx.annotation.Nullable;

@SuppressLint("AppCompatCustomView")
public class FloodFillImageView extends ImageView {

    public FloodFillImageView(Context context) {
        super(context);
    }

    public FloodFillImageView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
    }

    public FloodFillImageView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    private Rect mRect = new Rect();

    public Rect getImageDisplayRect(){
        Matrix m = this.getImageMatrix();
        float[] values = new float[9];
        m.getValues(values);

        Rect rect = getDrawable().getBounds();
        mRect.left = (int) values[2];
        mRect.top = (int) values[5];
        mRect.right = (int) (mRect.left + rect.width() * values[0]);
        mRect.bottom = (int) (mRect.top + rect.height() * values[0]);
        return mRect;
    }

    int[] mRealPoint = new int[2];
    public int[] pointOnReadImage(int x, int y){
        Matrix m = this.getImageMatrix();
        float[] values = new float[9];
        m.getValues(values);

        //Image在绘制过程中的变换矩阵，从中获得x和y方向的缩放系数
        float sx = values[0];
        float sy = values[4];

        Rect display = getImageDisplayRect();

        mRealPoint[0] = (int) ((x - display.left) / sx);
        mRealPoint[1] = (int) ((y - display.top) / sy);
        return mRealPoint;
    }
}
