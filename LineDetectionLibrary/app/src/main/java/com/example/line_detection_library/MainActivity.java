package com.example.line_detection_library;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "OpenCV";
    private static final String FILE = "image1";

    private static final int KERNEL_SIZE = 3;
    private static final int ANGLE_CUTOFF = 75;

    private static final int THRESHOLD1_CANNY = 1;
    private static final int THRESHOLD2_CANNY = 200;

    private static final int THRESHOLD_HOUGH = 15;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        checkLibraryLoads();

        Bitmap imageBitmap = convertImageToBitmap(FILE);
        Mat original = new Mat();

        List<Mat> colors = new ArrayList<>();

        Mat gray = new Mat();
        Mat red;
        Mat green;
        Mat blue;

        Mat grayBlur = new Mat();
        Mat redBlur = new Mat();
        Mat greenBlur = new Mat();
        Mat blueBlur = new Mat();

        Mat edgesGray = new Mat();
        Mat edgesRed = new Mat();
        Mat edgesGreen = new Mat();
        Mat edgesBlue = new Mat();

        Mat lines = new Mat();

        Utils.bitmapToMat(imageBitmap, original);

        Size scale = new Size(original.size().width / 4, original.size(). height / 4);
        Bitmap display = Bitmap.createBitmap((int) scale.width, (int) scale.height, Bitmap.Config.ARGB_8888);

        Imgproc.cvtColor(original, gray, Imgproc.COLOR_RGB2GRAY);
        Core.split(original, colors);

        red = colors.get(0);
        green = colors.get(1);
        blue = colors.get(2);

        /*
        Imgproc.GaussianBlur(gray, grayBlur, new Size(KERNEL_SIZE, KERNEL_SIZE), 0);
        Imgproc.GaussianBlur(red, redBlur, new Size(KERNEL_SIZE, KERNEL_SIZE), 0);
        Imgproc.GaussianBlur(green, greenBlur, new Size(KERNEL_SIZE, KERNEL_SIZE), 0);
        Imgproc.GaussianBlur(blue, blueBlur, new Size(KERNEL_SIZE, KERNEL_SIZE), 0);
        */

        Imgproc.resize(original, original, scale);
        Imgproc.resize(gray, grayBlur, scale);
        Imgproc.resize(red, redBlur, scale);
        Imgproc.resize(green, greenBlur, scale);
        Imgproc.resize(blue, blueBlur, scale);

        Imgproc.Canny(grayBlur, edgesGray, THRESHOLD1_CANNY, THRESHOLD2_CANNY);

        Utils.matToBitmap(edgesGray, display);
        saveToInternalStorage(display, "grayCanny.jpg");

        Imgproc.Canny(redBlur, edgesRed, THRESHOLD1_CANNY, THRESHOLD2_CANNY);

        Utils.matToBitmap(edgesRed, display);
        saveToInternalStorage(display, "redCanny.jpg");

        Imgproc.Canny(greenBlur, edgesGreen, THRESHOLD1_CANNY, THRESHOLD2_CANNY);

        Utils.matToBitmap(edgesGreen, display);
        saveToInternalStorage(display, "greenCanny.jpg");

        Imgproc.Canny(blueBlur, edgesBlue, THRESHOLD1_CANNY, THRESHOLD2_CANNY);

        Utils.matToBitmap(edgesBlue, display);
        saveToInternalStorage(display, "blueCanny.jpg");

        //Do red computation

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edgesRed, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        List<MatOfPoint> hullList = new ArrayList<>();

        for(MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            Point[] contourArray = contour.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }

        double perimeter;
        Scalar color = new Scalar(255, 255, 255);
        Mat drawing = Mat.zeros(edgesRed.size(), CvType.CV_8UC1);
        for (int i = 0; i < contours.size(); i++) {

            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter > original.size().height / 3) {
                Imgproc.drawContours(drawing, contours, i, color);
                //Imgproc.drawContours(drawing, hullList, i, color);
            }
        }

        Utils.matToBitmap(drawing, display);
        saveToInternalStorage(display, "redRemove.jpg");

        Imgproc.HoughLinesP(drawing, lines, 1, Math.PI / 180, THRESHOLD_HOUGH,
                original.size().height / 4, original.size().height / 5);

        for (int x = 0; x < lines.rows(); x++) {
            double[] l = lines.get(x, 0);

            double angle = Math.atan((l[1] - l[3])/(l[0] - l[2]));
            angle = Math.abs(angle / Math.PI * 180);

            if(angle > ANGLE_CUTOFF) {
                Imgproc.line(original, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(255, 0, 0), 1, Imgproc.LINE_AA, 0);
            }
        }

        Utils.matToBitmap(original, display);
        saveToInternalStorage(display, "middleOriginal.jpg");

        //Do green computation

        contours = new ArrayList<>();
        hierarchy = new Mat();
        Imgproc.findContours(edgesGreen, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        hullList = new ArrayList<>();

        for(MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            Point[] contourArray = contour.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }

        drawing = Mat.zeros(edgesGreen.size(), CvType.CV_8UC1);
        for (int i = 0; i < contours.size(); i++) {

            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter > original.size().height / 3) {
                Imgproc.drawContours(drawing, contours, i, color);
                //Imgproc.drawContours(drawing, hullList, i, color);
            }
        }

        Utils.matToBitmap(drawing, display);
        saveToInternalStorage(display, "greenRemove.jpg");

        Imgproc.HoughLinesP(drawing, lines, 1, Math.PI / 180, THRESHOLD_HOUGH,
                original.size().height / 4, original.size().height / 5);

        for (int x = 0; x < lines.rows(); x++) {
            double[] l = lines.get(x, 0);

            double angle = Math.atan((l[1] - l[3])/(l[0] - l[2]));
            angle = Math.abs(angle / Math.PI * 180);

            if(angle > ANGLE_CUTOFF) {
                Imgproc.line(original, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(255, 0, 0), 10, Imgproc.LINE_AA, 0);
            }
        }

        //Do blue computation

        contours = new ArrayList<>();
        hierarchy = new Mat();
        Imgproc.findContours(edgesBlue, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        hullList = new ArrayList<>();

        for(MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            Point[] contourArray = contour.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }

        drawing = Mat.zeros(edgesBlue.size(), CvType.CV_8UC1);
        for (int i = 0; i < contours.size(); i++) {

            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter > original.size().height / 3) {
                Imgproc.drawContours(drawing, contours, i, color);
                //Imgproc.drawContours(drawing, hullList, i, color);
            }
        }

        Utils.matToBitmap(drawing, display);
        saveToInternalStorage(display, "blueRemove.jpg");

        Imgproc.HoughLinesP(drawing, lines, 1, Math.PI / 180, THRESHOLD_HOUGH,
                original.size().height / 4, original.size().height / 5);

        for (int x = 0; x < lines.rows(); x++) {
            double[] l = lines.get(x, 0);

            double angle = Math.atan((l[1] - l[3])/(l[0] - l[2]));
            angle = Math.abs(angle / Math.PI * 180);

            if(angle > ANGLE_CUTOFF) {
                Imgproc.line(original, new Point(l[0], l[1]), new Point(l[2], l[3]), new Scalar(255, 0, 0), 10, Imgproc.LINE_AA, 0);
            }
        }

        //Do gray comparison

        contours = new ArrayList<>();
        hierarchy = new Mat();
        Imgproc.findContours(edgesGray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        hullList = new ArrayList<>();

        for(MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            Point[] contourArray = contour.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }

        drawing = Mat.zeros(edgesGray.size(), CvType.CV_8UC1);
        for (int i = 0; i < contours.size(); i++) {

            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter > original.size().height / 3) {
                Imgproc.drawContours(drawing, contours, i, color);
                //Imgproc.drawContours(drawing, hullList, i, color);
            }
        }

        Imgproc.HoughLinesP(drawing, lines, 1, Math.PI / 180, THRESHOLD_HOUGH,
                original.size().height / 4, original.size().height / 5);


        Utils.matToBitmap(original, display);
        saveToInternalStorage(display, "finalProduct.jpg");
    }

    private static void checkLibraryLoads() {
        if(OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCV successfully loaded!");
        } else {
            Log.i(TAG, "OpenCV failed to load!");
        }
    }

    private Bitmap convertImageToBitmap(String name) {
        InputStream stream = null;
        Uri uri = Uri.parse("android.resource://com.example.line_detection_library/drawable/" + name);
        try {
            stream = getContentResolver().openInputStream(uri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
        bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
        return BitmapFactory.decodeStream(stream, null, bmpFactoryOptions);
    }

    private String saveToInternalStorage(Bitmap bitmapImage, String filename){
        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        // path to /data/data/yourapp/app_data/imageDir
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        // Create imageDir
        File mypath=new File(directory, filename);

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
