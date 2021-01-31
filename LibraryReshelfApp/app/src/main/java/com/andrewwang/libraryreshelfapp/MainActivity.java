package com.andrewwang.libraryreshelfapp;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE = 1;

    private static final double GRAY_SCALE_FACTOR = 0.25;
    private static final double TAG_HEIGHT_SCALE_FACTOR = 3;
    private static final double DISPLAY_SCALE_FACTOR = 0.5;
    private static final double CROPPED_IMAGE_SCALE_FACTOR = 0.5;

    private static final int THRESHOLD_CANNY1 = 1;
    private static final int THRESHOLD_CANNY2 = 200;
    private static final int THRESHOLD_HOUGH = 15;

    private static final Size KERNEL_SIZE_SCALED = new Size(1,1);
    private static final Size KERNEL_SIZE = new Size(5,5);

    private static final int CONTOURS_THICKNESS = 5;
    private static final int MIN_SHELF_THICKNESS = 25;
    private static final int MAX_SHELF_THICKNESS = 100;
    private static final int HOUGH_LINE_THICKNESS = 1;

    private static final int MIN_CONTOUR_PERIMETER = 300; //300 is good
    private static final int MIN_VERTICAL_LINE = 250;
    private static final int MIN_ANGLE = 80;
    private static final int MIN_TAG_WIDTH = 125;

    private static final int CROP_BUFFER = 30;

    ArrayList<String> ocrRecognized = new ArrayList<>();
    ArrayList<Rect> rectangles = new ArrayList<>();
    ArrayList<Rect> goodRectangles = new ArrayList<>();
    ArrayList<Mat> croppedTags = new ArrayList<>();

    Mat displayImage;

    ImageView imageView;
    RelativeLayout layout;
    TextView textView;

    String currentPhotoPath;
    File photoFile;

    double[] white = new double[]{255, 255, 255};
    double[] black = new double[]{0, 0, 0};

    Scalar whiteScalar = new Scalar(255, 255, 255);
    Scalar redScalar = new Scalar(255, 0, 0);
    Scalar yellowScalar = new Scalar(255, 255, 0);
    Scalar greenScalar = new Scalar(0, 255, 0);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        layout = findViewById(R.id.layout);
        textView = findViewById(R.id.textView);

        checkLibraryLoads();

        layout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                textView.setVisibility(View.GONE);
                getImage();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        /*
        SETUP FILES
        SETUP FILES
        SETUP FILES
         */

        //Get image from saved file
        Bitmap bmp = BitmapFactory.decodeFile(currentPhotoPath);

        //Create the original mat object
        Mat original = new Mat();
        Utils.bitmapToMat(bmp, original);

        //Create the image to display to the user
        displayImage = new Mat();
        Imgproc.resize(original, displayImage, new Size(), DISPLAY_SCALE_FACTOR, DISPLAY_SCALE_FACTOR);
        Imgproc.cvtColor(displayImage, displayImage, Imgproc.COLOR_RGBA2RGB);

        //Convert image to gray and created a scaled version of the mat, then blur it
        Mat gray = colorToGray(original);
        Mat grayScaled = new Mat();
        Imgproc.resize(gray, grayScaled, new Size(), GRAY_SCALE_FACTOR, GRAY_SCALE_FACTOR);

        //Blur both images
        Imgproc.GaussianBlur(gray, gray, KERNEL_SIZE, 0, 0);
        Imgproc.GaussianBlur(grayScaled, grayScaled, KERNEL_SIZE_SCALED, 0, 0);

        //Get edges from the gray Mat
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, THRESHOLD_CANNY1, THRESHOLD_CANNY2);

        //Get the edgesScaled from the grayScaled Mat
        Mat edgesScaled = new Mat();
        Imgproc.Canny(grayScaled, edgesScaled, THRESHOLD_CANNY1, THRESHOLD_CANNY2);

        /*
        ELIMINATE NOISE FROM SCALED IMAGE
        ELIMINATE NOISE FROM SCALED IMAGE
        ELIMINATE NOISE FROM SCALED IMAGE
         */

        //Get looped items
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        ArrayList<MatOfPoint> hullList = new ArrayList<>();
        Imgproc.findContours(edgesScaled, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
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

        //New mat where noise is removed from the edgesScaled
        Mat noiseReducedCanny1 = Mat.zeros(edgesScaled.size(), CvType.CV_8UC1);

        //Filtering metrics to filter out small segments leaving only the large segments; do actual filtering
        for (int i = 0; i < contours.size(); i++) {
            if(Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true) > grayScaled.rows()) {
                Imgproc.drawContours(noiseReducedCanny1, contours, i, whiteScalar, CONTOURS_THICKNESS);
            }
        }

        /*
        THIN LINES HORIZONTALLY
        THIN LINES HORIZONTALLY
        THIN LINES HORIZONTALLY
         */

        double[] rgb;
        double[] whiteThinning = new double[] {255};
        double[] blackThinning = new double[] {0};

        int middlePixel;
        int inACol = 0;

        Mat noiseReducedCannyHori1 = noiseReducedCanny1.clone();

        //Write bottom line to be black
        for(int i = 0; i < noiseReducedCannyHori1.cols(); i++) {
            noiseReducedCannyHori1.put(noiseReducedCannyHori1.rows() - 1, i, black);
        }

        //Preform the line thinning
        for(int i = 0; i < noiseReducedCannyHori1.cols(); i++) {
            for(int j = 0; j < noiseReducedCannyHori1.rows(); j++) {
                rgb = noiseReducedCannyHori1.get(j, i);
                if(rgb[0] == 255) {
                    inACol++;
                } else {
                    if(inACol % 2 == 1) {
                        middlePixel = (int) Math.ceil(j - inACol / 2.0);
                        for(int x = j - inACol; x < j; x++) {
                            noiseReducedCannyHori1.put(x, i, blackThinning);
                        }
                        noiseReducedCannyHori1.put(middlePixel, i, whiteThinning);
                    } else if(inACol != 0 && inACol % 2 == 0) {
                        middlePixel = j - inACol / 2;
                        for(int x = j - inACol; x < j; x++) {
                            noiseReducedCannyHori1.put(x, i, blackThinning);
                        }
                        noiseReducedCannyHori1.put(middlePixel, i, blackThinning);
                    }
                    inACol = 0;
                }
            }
        }

        /*
        CONNECT LINES HORIZONTALLY
        CONNECT LINES HORIZONTALLY
        CONNECT LINES HORIZONTALLY
         */

        Mat kernelHorizontal = Mat.ones(1, 50, CvType.CV_8UC1);
        Imgproc.dilate(noiseReducedCannyHori1, noiseReducedCannyHori1, kernelHorizontal);
        Imgproc.erode(noiseReducedCannyHori1, noiseReducedCannyHori1, kernelHorizontal);

        /*
        IDENTIFY LONGEST HORIZONTAL LINE
        IDENTIFY LONGEST HORIZONTAL LINE
        IDENTIFY LONGEST HORIZONTAL LINE
         */

        contours.clear();
        hullList.clear();
        Imgproc.findContours(noiseReducedCannyHori1, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
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
        int longestHorizontalLine = 0;
        int indexOfLine = 0;
        for(int i = 0; i < hullList.size(); i++) {
            if(longestHorizontalLine < Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true)) {
                longestHorizontalLine = (int) Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
                indexOfLine = i;
            }
        }
        Mat noiseReducedCannyHori2 = new Mat(noiseReducedCannyHori1.size(), CvType.CV_8UC1);
        Imgproc.drawContours(noiseReducedCannyHori2, contours, indexOfLine, whiteScalar, 1);

        /*
        IDENTIFY ENDPOINTS OF LONGEST LINE
        IDENTIFY ENDPOINTS OF LONGEST LINE
        IDENTIFY ENDPOINTS OF LONGEST LINE
         */

        //Identify points to get shelf height
        Point left = new Point();
        Point right = new Point();
        int minSearchHeight = noiseReducedCannyHori2.rows() / 2;
        int row = noiseReducedCannyHori2.rows() - 1;
        int col = 0;

        //Left
        rgb = noiseReducedCannyHori2.get(row , col);
        while(rgb[0] == 0) {
            row--;
            if(row == minSearchHeight) {
                row = noiseReducedCannyHori2.rows() - 1;
                col++;
            }
            rgb = noiseReducedCannyHori2.get(row, col);
        }
        left.x = row;
        left.y = col;

        //Right
        row = noiseReducedCannyHori2.rows() - 1;
        col = noiseReducedCannyHori2.cols() - 1;
        rgb = noiseReducedCannyHori2.get(row, col);
        while(rgb[0] == 0) {
            row--;
            if(row == minSearchHeight) {
                row = noiseReducedCannyHori2.rows() - 1;
                col--;
            }
            rgb = noiseReducedCannyHori2.get(row, col);
        }
        right.x = row;
        right.y = col;

        /*
        //New points algorithm
        ArrayList<Integer> shelfHeights = new ArrayList<>();

        int shelfBottomHeight = (int) ((right.x + left.x) / 2);
        row = shelfBottomHeight - MIN_SHELF_THICKNESS;

        for(int i = (int) left.y; i < right.y; i+= 2) {
            rgb = noiseReducedCannyHori2.get(row, i);
            while(rgb[0] == 0) {
                row--;
                rgb = noiseReducedCannyHori2.get(row, i);
            }
            if(shelfBottomHeight - row < MAX_SHELF_THICKNESS) {
                shelfHeights.add(shelfBottomHeight - row);
            }
            row = shelfBottomHeight - MIN_SHELF_THICKNESS;
        }
        int shelfHeight = arrayAverage(shelfHeights);
        */

        /*
        GET REMAINING SHELF POINTS
        GET REMAINING SHELF POINTS
        GET REMAINING SHELF POINTS
         */

        List<Point> points = new ArrayList<>();
        points.add(left);

        //Get every other point
        row = noiseReducedCannyHori2.rows() - 1;
        for(int i = (int) left.y + 1; i < right.y; i++) {
            rgb = noiseReducedCannyHori2.get(row, i);
            while(rgb[0] == 0) {
                row--;
                rgb = noiseReducedCannyHori2.get(row, i);
            }
            points.add(new Point(row, i));
            row = noiseReducedCannyHori2.rows() - 1;
        }

        //Add last point
        points.add(right);

        /*
        CALCULATE SHELF HEIGHT
        CALCULATE SHELF HEIGHT
        CALCULATE SHELF HEIGHT
         */

        ArrayList<Integer> shelfHeights = new ArrayList<>();
        for(Point p : points) {
            row = (int) p.x - MIN_SHELF_THICKNESS;
            col = (int) p.y;
            rgb = edgesScaled.get(row, col);
            while(rgb[0] == 0) {
                row--;
                rgb = edgesScaled.get(row, col);
            }
            if(p.x - row < MAX_SHELF_THICKNESS) {
                shelfHeights.add((int) (p.x - row));
            }
        }

        int shelfHeight = arrayAverage(shelfHeights);

        /*
        CROP BOTTOM STRIP
        CROP BOTTOM STRIP
        CROP BOTTOM STRIP
         */

        int cropBottomRow = (int) left.x - shelfHeight;
        int cropTopRow = (int) (cropBottomRow - shelfHeight * TAG_HEIGHT_SCALE_FACTOR);
        cropBottomRow = (int) (cropBottomRow / GRAY_SCALE_FACTOR);
        cropTopRow = (int) (cropTopRow / GRAY_SCALE_FACTOR);
        Mat tagStrip = edges.submat(cropTopRow, cropBottomRow, 0, edges.cols() - 1);

        /*
        REDUCE NOISE FROM STRIP
        REDUCE NOISE FROM STRIP
        REDUCE NOISE FROM STRIP
         */

        Mat noiseReducedTagStrip1 = new Mat(tagStrip.size(), CvType.CV_8UC1);

        contours.clear();
        hullList.clear();
        Imgproc.findContours(tagStrip, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
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
        for(int i = 0; i < hullList.size(); i++) {
            if(Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true) > MIN_CONTOUR_PERIMETER) {
                Imgproc.drawContours(noiseReducedTagStrip1, contours, i, whiteScalar);
            }
        }

        Mat noiseReducedTagStrip2 = new Mat(noiseReducedTagStrip1.size(), CvType.CV_8UC1);
        Mat linesDrawn = new Mat(noiseReducedTagStrip2.size(), CvType.CV_8UC1);

        /*
        ELONGATE VERTICALLY
        ELONGATE VERTICALLY
        ELONGATE VERTICALLY
         */

        int verticalSize = noiseReducedTagStrip1.rows() / 150;
        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, verticalSize));
        Imgproc.erode(noiseReducedTagStrip1, noiseReducedTagStrip1, verticalStructure);
        Imgproc.dilate(noiseReducedTagStrip1, noiseReducedTagStrip1, verticalStructure);
        Mat kernelVertical = Mat.ones(50, 1, CvType.CV_8UC1);
        Imgproc.dilate(noiseReducedTagStrip1, noiseReducedTagStrip1, kernelVertical);
        Imgproc.erode(noiseReducedTagStrip1, noiseReducedTagStrip1, kernelVertical);

        /*
        REDUCE NOISE FROM STRIP AGAIN
        REDUCE NOISE FROM STRIP AGAIN
        REDUCE NOISE FROM STRIP AGAIN
         */

        contours.clear();
        hullList.clear();
        Imgproc.findContours(noiseReducedTagStrip1, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
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
        for(int i = 0; i < hullList.size(); i++) {
            if(Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true) > MIN_VERTICAL_LINE) {
                Imgproc.drawContours(noiseReducedTagStrip2, contours, i, whiteScalar);
            }
        }

        /*
        EXTEND VERTICAL LINES VERTICALLY TO FIT WHOLE HEIGHT
        EXTEND VERTICAL LINES VERTICALLY TO FIT WHOLE HEIGHT
        EXTEND VERTICAL LINES VERTICALLY TO FIT WHOLE HEIGHT
         */

        Mat lines = new Mat();
        Imgproc.HoughLinesP(noiseReducedTagStrip2, lines, 1, Math.PI / 180, THRESHOLD_HOUGH);
        double[] l;
        double angle;
        double slope;
        double yIntercept;
        for (int x = 0; x < lines.rows(); x++) {
            l = lines.get(x, 0);
            slope = (l[1] - l[3])/(l[0] - l[2]);
            angle = Math.abs(Math.atan(slope)) / Math.PI * 180;

            if(angle > MIN_ANGLE) {

                if(l[0] == l[2]) {
                    l[1] = 5;
                    l[3] = noiseReducedTagStrip2.rows() - 5;
                } else {
                    yIntercept = l[1] - slope*l[0];
                    l[0] = (5 - yIntercept) / slope;
                    l[1] = 5;
                    l[2] = (noiseReducedTagStrip2.rows() - 5 - yIntercept) / slope;
                    l[3] = noiseReducedTagStrip2.rows() - 5;
                }
                Imgproc.line(linesDrawn, new Point(l[0], l[1]), new Point(l[2], l[3]), whiteScalar, HOUGH_LINE_THICKNESS, Imgproc.LINE_AA);
            }
        }

        /*
        GET COORDINATES TO CROP FROM
        GET COORDINATES TO CROP FROM
        GET COORDINATES TO CROP FROM
         */
        ArrayList<Integer> coordinates = new ArrayList<>();
        coordinates.add(0);
        for(int i = 0; i < linesDrawn.cols(); i++) {
            rgb = linesDrawn.get(linesDrawn.rows() - 20, i);
            if(rgb[0] != 0 && i - coordinates.get(coordinates.size() - 1) > MIN_TAG_WIDTH) {
                coordinates.add(i);
            }
        }
        coordinates.add(linesDrawn.cols() - 1);

        /*
        NEEDS TO BE OPTIMIZED
        NEEDS TO BE OPTIMIZED
        NEEDS TO BE OPTIMIZED
         */

        Mat croppedImage;
        double[] hsv;
        int whiteInARow;
        int topSecond = 0;
        int bottomSecond = 0;
        Rect r;

        for(int i = 0; i < coordinates.size() - 1; i++) {
            r = new Rect();
            if(coordinates.get(i + 1) - coordinates.get(i) > MIN_TAG_WIDTH) {
                if(coordinates.get(i + 1) + CROP_BUFFER > original.cols()) {
                    croppedImage = original.submat(cropTopRow, cropBottomRow, coordinates.get(i), coordinates.get(i + 1));
                    Imgproc.resize(croppedImage, croppedImage, new Size(), CROPPED_IMAGE_SCALE_FACTOR, CROPPED_IMAGE_SCALE_FACTOR);

                    r.x = (int) (coordinates.get(i) * DISPLAY_SCALE_FACTOR);
                    r.y = (int) (cropTopRow * DISPLAY_SCALE_FACTOR);
                    r.height = (int) ((cropBottomRow - cropTopRow) * DISPLAY_SCALE_FACTOR);
                    r.width = (int) ((coordinates.get(i + 1) - coordinates.get(i)) * DISPLAY_SCALE_FACTOR);

                    rectangles.add(r);

                    Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_RGBA2BGR);
                    Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_BGR2HSV);

                    top:
                    for(int x = 0; x < croppedImage.rows(); x++) {
                        whiteInARow = 0;
                        for(int y = 0; y < croppedImage.cols(); y++) {
                            hsv = croppedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                whiteInARow++;
                            } else {
                                if(whiteInARow >= croppedImage.cols() / 2) {
                                    topSecond = x;
                                    break top;
                                }
                                whiteInARow = 0;
                            }
                        }
                    }

                    bottom:
                    for(int x = croppedImage.rows() - 1; x >= 0; x--) {
                        whiteInARow = 0;
                        for(int y = 0; y < croppedImage.cols(); y++) {
                            hsv = croppedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                whiteInARow++;
                            } else {
                                if(whiteInARow >= croppedImage.cols() / 2) {
                                    bottomSecond = x;
                                    break bottom;
                                }
                                whiteInARow = 0;
                            }
                        }
                    }

                    if(topSecond != bottomSecond) {
                        croppedImage = croppedImage.submat(topSecond, bottomSecond, 0, croppedImage.cols() - 1);
                    }

                    for(int x = 0; x < croppedImage.rows(); x++) {
                        for(int y = 0; y < croppedImage.cols(); y++) {
                            hsv = croppedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 100) {
                                croppedImage.put(x, y, white);
                            } else {
                                croppedImage.put(x, y, black);
                            }
                        }
                    }

                    croppedTags.add(croppedImage);
                } else {
                    croppedImage = original.submat(cropTopRow, cropBottomRow, coordinates.get(i), coordinates.get(i + 1) + CROP_BUFFER);
                    Imgproc.resize(croppedImage, croppedImage, new Size(), CROPPED_IMAGE_SCALE_FACTOR, CROPPED_IMAGE_SCALE_FACTOR);

                    r.x = (int) (coordinates.get(i) * DISPLAY_SCALE_FACTOR);
                    r.y = (int) (cropTopRow * DISPLAY_SCALE_FACTOR);
                    r.height = (int) ((cropBottomRow - cropTopRow) * DISPLAY_SCALE_FACTOR);
                    r.width = (int) ((coordinates.get(i + 1) - coordinates.get(i)) * DISPLAY_SCALE_FACTOR);

                    rectangles.add(r);

                    Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_RGBA2BGR);
                    Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_BGR2HSV);

                    top:
                    for(int x = 0; x < croppedImage.rows(); x++) {
                        whiteInARow = 0;
                        for(int y = 0; y < croppedImage.cols(); y++) {
                            hsv = croppedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                whiteInARow++;
                            } else {
                                if(whiteInARow >= croppedImage.cols() / 2) {
                                    topSecond = x;
                                    break top;
                                }
                                whiteInARow = 0;
                            }
                        }
                    }

                    bottom:
                    for(int x = croppedImage.rows() - 1; x >= 0; x--) {
                        whiteInARow = 0;
                        for(int y = 0; y < croppedImage.cols(); y++) {
                            hsv = croppedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                whiteInARow++;
                            } else {
                                if(whiteInARow >= croppedImage.cols() / 2) {
                                    bottomSecond = x;
                                    break bottom;
                                }
                                whiteInARow = 0;
                            }
                        }
                    }

                    if(topSecond != bottomSecond) {
                        croppedImage = croppedImage.submat(topSecond, bottomSecond, 0, croppedImage.cols() - 1);
                    }

                    for(int x = 0; x < croppedImage.rows(); x++) {
                        for(int y = 0; y < croppedImage.cols(); y++) {
                            hsv = croppedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 100) {
                                croppedImage.put(x, y, white);
                            } else {
                                croppedImage.put(x, y, black);
                            }
                        }
                    }

                    croppedTags.add(croppedImage);
                }
            }
        }

        OCRThread ocrThread = new OCRThread();
        ocrThread.start();
    }

    class OCRThread extends Thread {

        @Override
        public void run() {

            Mat mat;
            Bitmap bitmap;
            FirebaseVisionImage image;
            FirebaseVisionTextRecognizer detector = FirebaseVision.getInstance().getOnDeviceTextRecognizer();
            Task<FirebaseVisionText> task;
            String text;

            //Read text from cropped images
            for(int i = 0; i < croppedTags.size(); i++) {
                mat = croppedTags.get(i);
                bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mat, bitmap);
                image = FirebaseVisionImage.fromBitmap(bitmap);
                task = detector.processImage(image);

                try {
                    text = Tasks.await(task).getText();
                    text = text.toLowerCase();
                    text = text.replaceAll("\\s","");
                    text = text.replace('.', ' ');
                    text = text.replace(',', ' ');

                    if(text.length() > 8 && text.length() < 19) {
                        for(int x = 0; x < text.length(); x++) {
                            char c = text.charAt(x);
                            if(c != ' '&& c != 'a' && c != 'b' && c != 'c' && c != 'd' && c != 'e' && c != 'f' && c != 'g' && c != 'h' && c != 'i'
                                    && c != 'j' && c != 'k' && c != 'l' && c != 'm' && c != 'n' && c != 'o' && c != 'p' && c != 'q' && c != 'r' && c != 's'
                                    && c != 't' && c != 'u' && c != 'v' && c != 'w' && c != 'x' && c != 'y' && c != 'z' && c != '0' && c != '1' && c != '2'
                                    && c != '3' && c != '4' && c != '5' && c != '6' && c != '7' && c != '8' && c != '9' && c != '.' && c != ',' && c != '\n') {
                                text = text.replace(c, ' ');
                            }
                            text = text.replaceAll(" ", "");
                        }
                        ocrRecognized.add(text);
                        goodRectangles.add(rectangles.get(i));
                        Log.i("SUCCESS", text + " Index: " + i);
                    } else {
                        //drawRect(displayImage, rectangles.get(i), yellowScalar, 5);
                        //drawRect(displayImage, rectangles.get(i), new Scalar(0, 0, 0), 5);
                        Log.i("SUCCESS", "Index: " + i);
                    }
                } catch (ExecutionException e) {
                    //drawRect(displayImage, rectangles.get(i), yellowScalar, 5);
                    //drawRect(displayImage, rectangles.get(i), new Scalar(255, 255, 255), 5);
                    Log.i("SUCCESS", "Index: " + i);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }

            //Close the detector
            try {
                detector.close();
            } catch(IOException e) {
                e.printStackTrace();
            }

            //Run comparisons
            for(int i = 0; i < ocrRecognized.size() - 1; i++) {
                if(i == 0) {
                    drawRect(displayImage, goodRectangles.get(i), greenScalar, 5);
                }
                if(ocrRecognized.get(i).compareTo(ocrRecognized.get(i+1)) <= 0) {
                    drawRect(displayImage, goodRectangles.get(i + 1), greenScalar, 5);
                } else {
                    drawRect(displayImage, goodRectangles.get(i + 1), redScalar, 5);
                }
            }

            //UPDATE IMAGEVIEW
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Bitmap bitmap = Bitmap.createBitmap(displayImage.cols(), displayImage.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(displayImage, bitmap);
                    displayImage.release();
                    ocrRecognized.clear();
                    rectangles.clear();
                    goodRectangles.clear();
                    croppedTags.clear();
                    imageView.setImageBitmap(bitmap);
                }
            });
        }
    }

    private static void checkLibraryLoads() {
        if(OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "OpenCV successfully loaded!");
        } else {
            Log.i("OpenCV", "OpenCV failed to load!");
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    private void getImage() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if(intent.resolveActivity(getPackageManager()) != null) {
            photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
            if(photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(this, "andrewwang.provider", photoFile);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(intent, REQUEST_CODE);
            }
        }
    }

    private Mat colorToGray(Mat color) {
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_RGB2GRAY);
        return gray;
    }

    private void drawRect(Mat image, Rect rect, Scalar color, int thickness) {
        Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), color, thickness);
    }

    private int arrayAverage(ArrayList<Integer> arr) {
        int sum = 0;
        for(int i = 0; i < arr.size(); i++) {
            sum += arr.get(i);
        }
        return sum / arr.size();
    }
}