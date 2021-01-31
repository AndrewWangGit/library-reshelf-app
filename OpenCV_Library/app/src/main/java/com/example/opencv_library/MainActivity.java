package com.example.opencv_library;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.util.SparseArray;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    //Constants
    private static final String TAG = "OpenCV";
    private static final String FILE = "test1";

    //ImageViews
    ImageView imageConverted;

    //TextViews
    TextView ocrText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Define imageView
        imageConverted = findViewById(R.id.imageConverted);

        //Define TextViews
        ocrText = findViewById(R.id.ocrText);

        //Check if OpenCV loaded
        checkLibraryLoads();

        //Grabs the original image from drawable
        Bitmap bmp = convertImageToBitmap(FILE);

        //Converts bitmap to grayscale mat
        Mat mat = bitmapToMatGrayscale(bmp);

        //Sharpen image and convert to binary image
        Imgproc.GaussianBlur(mat, mat, new Size(1, 1), 0);
        Imgproc.threshold(mat, mat, 0, 255, Imgproc.THRESH_OTSU);
        Imgproc.adaptiveThreshold(mat, mat, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 40);

        //New image where result will be displayed
        Mat dest = Mat.zeros(mat.size(), CvType.CV_8UC3);

        //Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mat, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        //Draw contours to new image using white
        Scalar white = new Scalar(255, 255, 255);
        Imgproc.drawContours(dest, contours, -1, white, -1);

        //Define variables to identify correct rectangles
        Point[] points = new Point[4];
        List<Bitmap> croppedImages = new ArrayList<>();
        List<RotatedRect> goodRectangle = new ArrayList<>();
        Point topLeft;
        Point bottomRight;
        Bitmap croppedImage;
        RotatedRect rotatedRect;

        //Identify rectangles
        for(MatOfPoint contour : contours) {
            //Find rectangles
            rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));

            //Filter correct rectangles to the images
            if((rotatedRect.size.area() / mat.size().area() >= 0.05 && rotatedRect.size.area() / mat.size().area() <= 0.75) &&
                    (Math.abs((rotatedRect.angle + 360) % 90) <= 10 || Math.abs((rotatedRect.angle + 360) % 90) >= 80) &&
                    (rotatedRect.size.height / rotatedRect.size.width >= 0.75 && rotatedRect.size.height / rotatedRect.size.width <= 1.5)) {

                //Add rectangle
                goodRectangle.add(rotatedRect);

            }
        }

        //Add buffer rectangle at the end
        goodRectangle.add(new RotatedRect(new Point(0,0), new Size(1,1), 0));
        RotatedRect rect1, rect2;

        for(int i = 0; i < goodRectangle.size() - 1; i++) {
            rect1 = goodRectangle.get(i);
            rect2 = goodRectangle.get(i + 1);

            if(!(Math.abs(rect1.center.x - rect2.center.x) < 20)) {

                //Grab points from rectangle
                rect1.points(points);

                //Identify bottomRight and topLeft points based on angle
                if(rect1.angle <= 10 && rect1.angle >= -10) {
                    topLeft = points[1];
                    bottomRight = points[3];
                } else if (rect1.angle <= -80 && rect1.angle >= -100) {
                    topLeft = points[2];
                    bottomRight = points[0];
                } else if (rect1.angle >= 80 && rect1.angle <= 100) {
                    topLeft = points[0];
                    bottomRight = points[2];
                } else {
                    topLeft = points[3];
                    bottomRight = points[1];
                }

                //Pull the croppedImage and adds it to ArrayList
                croppedImage = Bitmap.createBitmap(bmp, (int) topLeft.x, (int) topLeft.y, (int) (bottomRight.x - topLeft.x), (int) (bottomRight.y - topLeft.y));
                croppedImages.add(croppedImage);
            } else {
                //Remove repeats
                goodRectangle.remove(rect2);

                //Account for index change
                i--;
            }
        }

        //Remove buffer rectangle at the end
        goodRectangle.remove(goodRectangle.size() - 1);

        List<String> ocrRecognized = new ArrayList<>();
        Frame frame;
        SparseArray<TextBlock> items;
        TextRecognizer textRecognizer = new TextRecognizer.Builder(getApplicationContext()).build();
        StringBuilder stringBuilder;
        TextBlock item;

        if(!textRecognizer.isOperational()) {
            Log.i("ERROR", "Detector dependencies are not yet available!");
        } else {
            for(Bitmap bitmap : croppedImages) {
                frame = new Frame.Builder().setBitmap(bitmap).build();
                items = textRecognizer.detect(frame);
                stringBuilder = new StringBuilder();
                for(int i = 0; i < items.size(); i++) {
                    item = items.valueAt(i);
                    stringBuilder.append(item.getValue());
                }
                ocrRecognized.add(stringBuilder.toString());
            }
        }

        Scalar green = new Scalar(0, 128, 0);
        Scalar red = new Scalar(255, 0, 0);
        String right;
        String left;
        int dewyOrder;
        int secondOrder;

        for(int i = 0; i < ocrRecognized.size() - 1; i++) {
            right = ocrRecognized.get(i);
            left = ocrRecognized.get(i + 1);

            if(right.contains("\n") && left.contains("\n")) {
                dewyOrder = right.substring(0, right.indexOf("\n")).compareTo(left.substring(0, left.indexOf("\n")));

                if(dewyOrder > 0) {
                    drawRotatedRect(dest, goodRectangle.get(i), green, 10);
                } else if (dewyOrder < 0) {
                    drawRotatedRect(dest, goodRectangle.get(i), red, 10);
                } else {
                    secondOrder = right.substring(right.indexOf("\n") + 1).compareTo(left.substring(left.indexOf("\n") + 1));
                    if(secondOrder >= 0) {
                        drawRotatedRect(dest, goodRectangle.get(i), green, 10);
                    } else {
                        drawRotatedRect(dest, goodRectangle.get(i), red, 10);
                    }
                }
            }
        }

        //Left-most rectangle is always correct
        drawRotatedRect(dest, goodRectangle.get(goodRectangle.size() - 1), green, 10);

        //Grabs image with drawn contours and green rectangles
        Bitmap resultBitmap = Bitmap.createBitmap(dest.cols(),  dest.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dest, resultBitmap);

        //Sets the converted image to imageConverted ImageView //resultBitmap typically is displayed
        setImageView(imageConverted, resultBitmap);
        for(String s : ocrRecognized) {
            ocrText.append(s);
            ocrText.append("\n--------------------------\n");
        }
        String filepath = saveToInternalStorage(resultBitmap);
        Log.i(TAG, filepath);
    }


    //ALL METHODS
    private static void checkLibraryLoads() {
        if(OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCV successfully loaded!");
        } else {
            Log.i(TAG, "OpenCV failed to load!");
        }
    }

    private Bitmap convertImageToBitmap(String name) {
        InputStream stream = null;
        Uri uri = Uri.parse("android.resource://com.example.opencv_library/drawable/" + name);
        try {
            stream = getContentResolver().openInputStream(uri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
        bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
        return BitmapFactory.decodeStream(stream, null, bmpFactoryOptions);
    }

    private static void setImageView(ImageView imageView, Bitmap bitmap) {
        imageView.setImageBitmap(bitmap);
    }

    private static Mat bitmapToMatGrayscale(Bitmap bitmap) {
        Mat matOriginal = new Mat();
        Utils.bitmapToMat(bitmap, matOriginal);
        Mat matGray = Mat.zeros(matOriginal.size(), CvType.CV_8UC3);
        Imgproc.cvtColor(matOriginal, matGray, Imgproc.COLOR_RGB2GRAY);
        return matGray;
    }

    public static void drawRotatedRect(Mat image, RotatedRect rotatedRect, Scalar color, int thickness) {
        Point[] vertices = new Point[4];
        rotatedRect.points(vertices);
        MatOfPoint points = new MatOfPoint(vertices);
        Imgproc.drawContours(image, Arrays.asList(points), -1, color, thickness);
    }

    private String saveToInternalStorage(Bitmap bitmapImage){
        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        // path to /data/data/yourapp/app_data/imageDir
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        // Create imageDir
        File mypath=new File(directory,"profile.jpg");

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mypath);
            // Use the compress method on the BitMap object to write image to the OutputStream
            bitmapImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return directory.getAbsolutePath();
    }
}