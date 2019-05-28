package com.andforce.opencv4;

import android.net.Uri;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.GridView;
import android.widget.ImageView;

import java.io.File;
import java.util.ArrayList;

public class ResultActivity extends AppCompatActivity {

    ArrayList<String> mImages;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        GridView gridView = findViewById(R.id.result_gridview);

        mImages = (getIntent().getExtras().getStringArrayList("result"));

        gridView.setAdapter(new MyAdapter());

    }


    class MyAdapter extends BaseAdapter{

        @Override
        public int getCount() {
            return mImages.size();
        }

        @Override
        public Object getItem(int position) {
            return mImages.get(position);
        }

        @Override
        public long getItemId(int position) {
            return position;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            ViewHolder viewHolder;
            if(convertView==null){
                LayoutInflater inflater = LayoutInflater.from(getBaseContext());
                convertView =inflater.inflate(R.layout.result_item,null);

                viewHolder = new ViewHolder();


                viewHolder.imageView = convertView.findViewById(R.id.result_imageview);
                convertView.setTag(viewHolder);
            }else {
                viewHolder =(ViewHolder)convertView.getTag();
            }

            viewHolder.imageView.setImageURI(Uri.fromFile(new File(mImages.get(position))));

            return convertView;
        }

        class ViewHolder{
            ImageView imageView;
        }
    }
}
