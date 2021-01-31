package com.example.opencv_library_video;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureRequest;
import android.media.ImageReader;
import android.os.Bundle;
import android.util.Log;
import android.util.SparseArray;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.TextureView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
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
import org.w3c.dom.Text;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OpenCV";
    CameraBridgeViewBase cameraBridgeViewBase;
    Mat matDisplay, matManipulate;
    ArrayList<MatOfPoint> contours = new ArrayList<>();
    ArrayList<RotatedRect> goodRectangle = new ArrayList<>();
    ArrayList<Bitmap> croppedImages = new ArrayList<>();
    ArrayList<String> ocrRecognized = new ArrayList<>();
    SparseArray<TextBlock> items = new SparseArray<>();
    Point[] points = new Point[4];
    RotatedRect rotatedRect, rect1, rect2;
    Point topLeft, bottomRight;
    Bitmap croppedImage, bmp;
    TextRecognizer textRecognizer;
    Frame frame;
    StringBuilder stringBuilder = new StringBuilder();
    TextBlock item;
    Scalar green = new Scalar(0, 128, 0), red = new Scalar(255, 0, 0);
    String right, left;
    int dewyOrder, secondOrder;

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            super.onManagerConnected(status);
            switch(status) {
                case BaseLoaderCallback.SUCCESS:
                    cameraBridgeViewBase.setMaxFrameSize(800,600);
                    cameraBridgeViewBase.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        checkLibraryLoads();

        cameraBridgeViewBase = findViewById(R.id.cameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        textRecognizer = new TextRecognizer.Builder(getApplicationContext()).build();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //Get mats
        matDisplay = inputFrame.rgba();
        matManipulate = inputFrame.gray();

        //Sharpen and convert to binary mat
        Imgproc.GaussianBlur(matManipulate, matManipulate, new Size(1, 1), 0);
        Imgproc.threshold(matManipulate, matManipulate, 0, 255, Imgproc.THRESH_OTSU);
        Imgproc.adaptiveThreshold(matManipulate, matManipulate, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 40);

        Utils.matToBitmap(matManipulate, bmp);

        //Find contours
        Imgproc.findContours(matManipulate, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        //Identify "good" rectangles
        for(MatOfPoint contour : contours) {
            //Find rectangles
            rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));

            //Filter correct rectangles to the images
            if((rotatedRect.size.area() / matManipulate.size().area() >= 0.01 && rotatedRect.size.area() / matManipulate.size().area() <= 0.05) &&
                    (Math.abs((rotatedRect.angle + 360) % 90) <= 10 || Math.abs((rotatedRect.angle + 360) % 90) >= 80) &&
                    (rotatedRect.size.height / rotatedRect.size.width >= 0.75 && rotatedRect.size.height / rotatedRect.size.width <= 1.5)) {

                //Add rectangle
                goodRectangle.add(rotatedRect);

            }
        }

        //Add buffer rectangle at the end
        goodRectangle.add(new RotatedRect(new Point(0,0), new Size(1,1), 0));

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
                //Bad lines of code - causing out of bound errors
                if(topLeft.x > 0 && topLeft.y > 0 && bottomRight.x <= bmp.getWidth() && bottomRight.y <= bmp.getHeight()) {
                    croppedImage = Bitmap.createBitmap(bmp, (int) topLeft.x, (int) topLeft.y, (int) (bottomRight.x - topLeft.x), (int) (bottomRight.y - topLeft.y));
                    croppedImages.add(croppedImage);
                }

            } else {
                //Remove repeats
                goodRectangle.remove(rect2);

                //Account for index change
                i--;
            }
        }

        /*

        for(Bitmap bitmap : croppedImages) {
            frame = new Frame.Builder().setBitmap(bitmap).build();
            items = textRecognizer.detect(frame);
            for(int i = 0; i < items.size(); i++) {
                item = items.valueAt(i);
                stringBuilder.append(item.getValue());
            }
            if(stringBuilder.toString().contains("\n")) {
                ocrRecognized.add(stringBuilder.toString());
            }
            stringBuilder.setLength(0);
        }

        for(int i = 0; i < ocrRecognized.size() - 1; i++) {
            right = ocrRecognized.get(i);
            left = ocrRecognized.get(i + 1);

            dewyOrder = right.substring(0, right.indexOf("\n")).compareTo(left.substring(0, left.indexOf("\n")));

            if(dewyOrder > 0) {
                drawRotatedRect(matDisplay, goodRectangle.get(i), green, 10);
            } else if (dewyOrder < 0) {
                drawRotatedRect(matDisplay, goodRectangle.get(i), red, 10);
            } else {
                secondOrder = right.substring(right.indexOf("\n") + 1).compareTo(left.substring(left.indexOf("\n") + 1));
                if(secondOrder >= 0) {
                    drawRotatedRect(matDisplay, goodRectangle.get(i), green, 10);
                } else {
                    drawRotatedRect(matDisplay, goodRectangle.get(i), red, 10);
                }
            }
        }

        if(goodRectangle.size() != 0) {
            drawRotatedRect(matDisplay, goodRectangle.get(goodRectangle.size() - 1), green, 10);
        }

        */
        goodRectangle.clear();
        ocrRecognized.clear();
        croppedImages.clear();
        contours.clear();
        return matDisplay;
    }

    @Override
    public void onCameraViewStopped() {
        matDisplay.release();
        matManipulate.release();
        bmp.recycle();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        matDisplay = new Mat(width, height, CvType.CV_8UC3);
        matManipulate = new Mat(width, height, CvType.CV_8UC3);
        bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCV failed to load!");
        } else {
            Log.i(TAG, "OpenCV successfully loaded!");
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    private static void checkLibraryLoads() {
        if(OpenCVLoader.initDebug()) {
            Log.i(TAG, "OpenCV successfully loaded!");
        } else {
            Log.i(TAG, "OpenCV failed to load!");
        }
    }

    private static void drawRotatedRect(Mat image, RotatedRect rotatedRect, Scalar color, int thickness) {
        Point[] vertices = new Point[4];
        rotatedRect.points(vertices);
        MatOfPoint points = new MatOfPoint(vertices);
        Imgproc.drawContours(image, Arrays.asList(points), -1, color, thickness);
    }
}