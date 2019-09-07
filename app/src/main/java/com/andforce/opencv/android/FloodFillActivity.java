package com.andforce.opencv.android;

import android.app.ProgressDialog;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.ArrayList;

public class FloodFillActivity extends AppCompatActivity {

    private static final int REQUEST_OPEN_IMAGE = 1;

    static {
        System.loadLibrary("flood-fill");
    }


    private ProgressDialog dlg;

    private int mStartY = -1;

    private int value = 50;

    Bitmap floodFillBitmap;
    Bitmap orgBitmap;
    Bitmap maskBitmap;
    Bitmap resultBitmap;

    FloodFillImageView mSrcImageView;
    ImageView mMaskImageView;
    ImageView mResultImageView;

    ArrayList<String> mResult;

    int[] realPoint;

    int[] orgPixels;
    int[] changedPixels;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_flood_fill);
//        new Thread(new Runnable() {
//            @Override
//            public void run() {
//                FloodFillUtils.blindWaterMark();
//            }
//        }).start();

        mMaskImageView = findViewById(R.id.maskImageView);

        mResultImageView = findViewById(R.id.resultImageView);

        mSrcImageView = findViewById(R.id.imageView);
        mSrcImageView.post(new Runnable() {
            @Override
            public void run() {
                mStartY = mSrcImageView.getMeasuredHeight() / 2;
            }
        });

        mSrcImageView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN: {
                        mStartY = (int) event.getY();

                        Rect rectDis = mSrcImageView.getImageDisplayRect();

                        Matrix matrix = mSrcImageView.getImageMatrix();

                        Log.d("ACTION_MOVE realSize->", "DiaplyRect:" + rectDis + " Matrix:" + matrix);

                        if (rectDis.contains((int) event.getX(), (int) event.getY())) {


                            realPoint = mSrcImageView.pointOnReadImage((int) event.getX(), (int) event.getY());

                            Log.d("ACTION_MOVE imageSize->", "点到了 " + "X:" + event.getX() + " Y:" + event.getY() + " realX:" + realPoint[0] + " realY:" + realPoint[1]);

                            Bitmap maskBitmap = orgBitmap.copy(Bitmap.Config.ARGB_8888, true);

                            Log.d("ACTION_MOVE imageSize->", "Bitmap W:" + maskBitmap.getWidth() + " Bitmap H:" + maskBitmap.getHeight());

                            //Log.d("ACTION_MOVE", "NEW:" + maskBitmap + " ORG:" + orgBitmap);

                            if (resultBitmap == null) {
                                resultBitmap = Bitmap.createBitmap(orgBitmap.getWidth(), orgBitmap.getHeight(), Bitmap.Config.ARGB_8888);
                            }

                            int[] pixels = FloodFillUtils.floodFillBitmapWithMask(orgBitmap, maskBitmap, resultBitmap, realPoint[0], realPoint[1], value, value);

                            int[] maskPixels = new int[maskBitmap.getWidth() * maskBitmap.getHeight()];
                            maskBitmap.getPixels(maskPixels, 0, maskBitmap.getWidth(), 0, 0, maskBitmap.getWidth(), maskBitmap.getHeight());

//                            for (int i = 0; i < maskPixels.length; i++){
//                                Log.d("ACTION_MOVE maskPixels", "Pix: "+ maskPixels[i]);
//                            }

                            //Log.d("ACTION_MOVE Count->", "Count: "+ count);

//                            mSrcImageView.setImageBitmap(maskBitmap);
                            mMaskImageView.setImageBitmap(maskBitmap);
                            resultBitmap.setPixels(pixels, 0, resultBitmap.getWidth(), 0, 0, resultBitmap.getWidth(), resultBitmap.getHeight());
                            mResultImageView.setImageBitmap(resultBitmap);

                        } else {
                            Log.d("ACTION_MOVE imageSize->", "没点到 " + "X:" + event.getX() + " Y:" + event.getY() + " display:" + rectDis);
//                            mSrcImageView.setImageBitmap(orgBitmap);
                            mMaskImageView.setImageBitmap(orgBitmap);
                        }

                        break;
                    }

                    case MotionEvent.ACTION_MOVE: {

                        if (Math.abs(event.getY() - mStartY) > 100) {

                            if (event.getY() > mStartY) {
                                // Scroll Down
                                value += 1;

                            } else {
                                // Scroll Up
                                value -= 1;
                            }

                            if (value > 255) {
                                value = 255;
                            }

                            if (value < 0) {
                                value = 0;
                            }

                            //Log.d("ACTION_MOVE", "floodFill" + value + " StartY:" + mStartY);

//                            Bitmap maskBitmap = orgBitmap.copy(Bitmap.Config.ARGB_8888, true);
                            if (maskBitmap == null) {
                                maskBitmap = Bitmap.createBitmap(orgBitmap.getWidth(), orgBitmap.getHeight(), Bitmap.Config.ARGB_8888);
                            }

                            if (resultBitmap == null) {
                                resultBitmap = Bitmap.createBitmap(orgBitmap.getWidth(), orgBitmap.getHeight(), Bitmap.Config.ARGB_8888);
                            }

                            //Log.d("ACTION_MOVE", "NEW:" + maskBitmap + " ORG:" + orgBitmap);

                            int[] pixels = FloodFillUtils.floodFillBitmapWithMask(orgBitmap, maskBitmap, resultBitmap, realPoint[0], realPoint[1], value, value);

//                            Log.d("ACTION_MOVE Count->", "Count: "+ count+ " maskBitmap:" + maskBitmap);

//                            mSrcImageView.setImageBitmap(maskBitmap);
                            mMaskImageView.setImageBitmap(maskBitmap);

                            resultBitmap.setPixels(pixels, 0, resultBitmap.getWidth(), 0, 0, resultBitmap.getWidth(), resultBitmap.getHeight());
                            mResultImageView.setImageBitmap(resultBitmap);
                        }

                        Log.d("ACTION_MOVE", ">>>>>> floodFill value:" + value + " StartY:" + mStartY + " Y:" + event.getY());
                        break;
                    }

                    case MotionEvent.ACTION_UP: {
                        value = 20;
                        break;
                    }
                }
                return true;
            }
        });

        dlg = new ProgressDialog(this);

        findViewById(R.id.choose_image).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent getPictureIntent = new Intent(Intent.ACTION_GET_CONTENT);
                getPictureIntent.setType("image/*");
                Intent pickPictureIntent = new Intent(Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                Intent chooserIntent = Intent.createChooser(getPictureIntent, "Select Image");
                chooserIntent.putExtra(Intent.EXTRA_INITIAL_INTENTS, new Intent[]{
                        pickPictureIntent
                });
                startActivityForResult(chooserIntent, REQUEST_OPEN_IMAGE);
            }
        });

        findViewById(R.id.show_result).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mResult == null || mResult.isEmpty()) {
                    Toast.makeText(getApplicationContext(), "结果空", Toast.LENGTH_SHORT).show();
                    return;
                }
                Intent intent = new Intent(FloodFillActivity.this, ResultActivity.class);
                intent.putStringArrayListExtra("result", mResult);
                startActivity(intent);
            }
        });
    }


    private void setPic(String pic) {
        int targetW = mSrcImageView.getWidth();
        int targetH = mSrcImageView.getHeight();

        BitmapFactory.Options bmOptions = new BitmapFactory.Options();
        bmOptions.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(pic, bmOptions);
        int photoW = bmOptions.outWidth;
        int photoH = bmOptions.outHeight;

        int scaleFactor = Math.min(photoW / targetW, photoH / targetH);

        bmOptions.inJustDecodeBounds = false;
        bmOptions.inSampleSize = scaleFactor;
        bmOptions.inPurgeable = true;

        mSrcImageView.setImageBitmap(BitmapFactory.decodeFile(pic, bmOptions));
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case REQUEST_OPEN_IMAGE:
                if (resultCode == RESULT_OK) {
                    Uri imgUri = data.getData();
                    String[] filePathColumn = {MediaStore.Images.Media.DATA};

                    Cursor cursor = getContentResolver().query(imgUri, filePathColumn,
                            null, null, null);
                    cursor.moveToFirst();

                    int colIndex = cursor.getColumnIndex(filePathColumn[0]);
                    final String imagePath = cursor.getString(colIndex);
                    cursor.close();
//                    setPic(imagePath);

                    int targetW = mSrcImageView.getWidth();
                    int targetH = mSrcImageView.getHeight();

                    BitmapFactory.Options bmOptions = new BitmapFactory.Options();
                    bmOptions.inJustDecodeBounds = true;
                    BitmapFactory.decodeFile(imagePath, bmOptions);
                    int photoW = bmOptions.outWidth;
                    int photoH = bmOptions.outHeight;

                    int scaleFactor = Math.min(photoW / targetW, photoH / targetH);

                    bmOptions.inJustDecodeBounds = false;
//                    bmOptions.inPreferredConfig = Bitmap.Config.RGB_565;
                    bmOptions.inSampleSize = scaleFactor;
                    bmOptions.inPurgeable = true;

                    orgBitmap = BitmapFactory.decodeFile(imagePath, bmOptions);

                    orgPixels = new int[orgBitmap.getWidth() * orgBitmap.getHeight()];
                    orgBitmap.getPixels(orgPixels, 0, orgBitmap.getWidth(), 0, 0, orgBitmap.getWidth(), orgBitmap.getHeight());

                    changedPixels = new int[orgPixels.length];

                    floodFillBitmap = BitmapFactory.decodeFile(imagePath, bmOptions);

                    mSrcImageView.setImageBitmap(orgBitmap);

//                    Bitmap bitmap1 = FloodFillUtils.floodFillBitmap(maskBitmap, 10, 10, 20, 20);
//
//                    mSrcImageView.setImageBitmap(bitmap1);

//                    mSrcImageView.postDelayed(new Runnable() {
//                        @Override
//                        public void run() {
//                            if (false) {
//                                FloodFillUtils.floodFill(floodFillBitmap, 10, 10, 20, 20);
//                                mSrcImageView.setImageBitmap(floodFillBitmap);
//                            } else {
//                                Bitmap bitmap1 = FloodFillUtils.floodFillBitmap(floodFillBitmap, 10, 10, 20, 20);
//                                mSrcImageView.setImageBitmap(floodFillBitmap);
//                            }
//                        }
//                    }, 1000);


                    if (false) {
                        dlg.setMessage("正在查找");
                        dlg.setCancelable(false);
                        dlg.setIndeterminate(true);
                        dlg.show();

                        new Thread(new Runnable() {
                            @Override
                            public void run() {
                                Log.d("FIND_OBJ", "start find");
                                mResult = find_objects(imagePath);
                                Log.d("FIND_OBJ", "count: " + mResult.size());
                                getWindow().getDecorView().post(new Runnable() {
                                    @Override
                                    public void run() {
                                        dlg.dismiss();

                                        Toast.makeText(getApplicationContext(), "找到：" + mResult.size() + "个", Toast.LENGTH_SHORT).show();

                                        Intent intent = new Intent(FloodFillActivity.this, ResultActivity.class);
                                        intent.putStringArrayListExtra("result", mResult);
                                        startActivity(intent);


                                    }
                                });
                            }
                        }).start();
                    }
                }
                break;
        }
    }

    //public static native void gray();

    public static native ArrayList<String> find_objects(String imagePath);
}
