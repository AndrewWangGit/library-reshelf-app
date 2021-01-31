package com.andrewwang.linedetectionv3;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;

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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    private static final String FILE = "image_16";
    private static final double SCALE_FACTOR = 0.25;
    private static final int THRESHOLD1_CANNY = 1;
    private static final int THRESHOLD2_CANNY = 200;

    private static final Size KERNEL_SIZE_SCALED = new Size(1,1);
    private static final Size KERNEL_SIZE = new Size(5,5);

    private static final int THICKNESS_CONTOURS = 5;

    private static final int MIN_SHELF_THICKNESS = 25;
    private static final double TAG_HEIGHT_FACTOR = 3;

    private static final int MAX_SHELF_THICKNESS = 100;

    private static final int MIN_CONTOUR_PERIMETER = 300; //300 is good
    private static final int MIN_VERTICAL_LINE = 250;

    private static final int THRESHOLD_HOUGH = 15;
    private static final int HOUGH_LINE_THICKNESS = 1;
    private static final int ANGLE_CUTOFF = 80;
    private static final int MIN_TAG_WIDTH = 125;
    private static final int CROP_BUFFER = 30;

    private static final double DISPLAY_SCALE_FACTOR = 0.5;

    ArrayList<Rect> rectangles = new ArrayList<>();
    ArrayList<String> ocrRecognized = new ArrayList<>();
    ArrayList<Rect> goodRectangles = new ArrayList<>();
    ArrayList<Mat> croppedTags = new ArrayList<>();
    Mat displayImage;

    Scalar scalarYellow = new Scalar(255, 255, 0);
    Scalar scalarRed = new Scalar(255, 0, 0);
    Scalar scalarGreen = new Scalar(0, 255, 0);
    Scalar scalarWhite = new Scalar(255, 255, 255);

    double[] black = new double[]{0, 0, 0};
    double[] white = new double[]{255, 255, 255};
    
    long startTime;
    long endTime;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        startTime = System.nanoTime();
        
        checkLibraryLoads();

        displayImage = new Mat();

        //Get gray image and create new scaled Mat
        Mat original = fileToMat(FILE);

        Mat gray = colorToGray(original);
        Mat grayScaled = new Mat();

        Imgproc.resize(original, displayImage, new Size(), DISPLAY_SCALE_FACTOR, DISPLAY_SCALE_FACTOR);
        Imgproc.cvtColor(displayImage, displayImage, Imgproc.COLOR_RGBA2RGB);

        //Set the scaled Mat and blur it
        Imgproc.resize(gray, grayScaled, new Size(), SCALE_FACTOR, SCALE_FACTOR);
        Imgproc.GaussianBlur(grayScaled, grayScaled, KERNEL_SIZE_SCALED, 0, 0);
        Imgproc.GaussianBlur(gray, gray, KERNEL_SIZE, 0, 0);

        //Get the edgesScaled from the grayScaled Mat
        Mat edges = new Mat();
        Mat edgesScaled = new Mat();
        Imgproc.Canny(grayScaled, edgesScaled, THRESHOLD1_CANNY, THRESHOLD2_CANNY);
        Imgproc.Canny(gray, edges, THRESHOLD1_CANNY, THRESHOLD2_CANNY);

        //Save edgesScaled to internal storage for viewing
        saveToInternalStorage(edgesScaled, "edgesScaled.jpg");

        //Find the contours from the edgesScaled detected
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edgesScaled, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        //Find the small segments and loop them together
        ArrayList<MatOfPoint> hullList = new ArrayList<>();
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

        //Filtering metrics to filter out small segments leaving only the large segments
        double perimeter;

        //New mat where noise is removed from the edgesScaled
        Mat noiseReducedCanny1 = Mat.zeros(edgesScaled.size(), CvType.CV_8UC1);

        //Do actual filtering
        for (int i = 0; i < contours.size(); i++) {
            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter > grayScaled.rows()) {
                Imgproc.drawContours(noiseReducedCanny1, contours, i, scalarWhite, THICKNESS_CONTOURS);
            }
        }

        //Save first iteration to internal storage
        saveToInternalStorage(noiseReducedCanny1, "noiseReducedCanny1.jpg");

        //Line thinning to get bottom shelf
        double[] rgbCanny;
        double[] whiteThinning = new double[] {255};
        double[] blackThinning = new double[] {0};
        int inACol = 0;
        Mat noiseReducedCannyHori1 = noiseReducedCanny1.clone();

        //Write bottom line to be black
        for(int i = 0; i < noiseReducedCannyHori1.cols(); i++) {
            noiseReducedCannyHori1.put(noiseReducedCannyHori1.rows() - 1, i, blackThinning);
        }

        for(int i = 0; i < noiseReducedCannyHori1.cols(); i++) {
            for(int j = 0; j < noiseReducedCannyHori1.rows(); j++) {
                rgbCanny = noiseReducedCannyHori1.get(j, i);
                if(rgbCanny[0] == 255) {
                    inACol++;
                } else {
                    if(inACol != 0 && inACol % 2 == 1) {
                        int middle = (int) Math.ceil(j - inACol / 2.0);
                        for(int x = j - inACol; x < j; x++) {
                            noiseReducedCannyHori1.put(x, i, blackThinning);
                        }
                        noiseReducedCannyHori1.put(middle, i, whiteThinning);
                    } else if(inACol != 0 && inACol % 2 == 0) {
                        int middle = j - inACol / 2;
                        for(int x = j - inACol; x < j; x++) {
                            noiseReducedCannyHori1.put(x, i, blackThinning);
                        }
                        noiseReducedCannyHori1.put(middle, i, blackThinning);
                    }
                    inACol = 0;
                }
            }
        }

        //Connect lines
        Mat kernel = Mat.ones(1, 50, CvType.CV_8UC1);
        Imgproc.dilate(noiseReducedCannyHori1, noiseReducedCannyHori1, kernel);
        Imgproc.erode(noiseReducedCannyHori1, noiseReducedCannyHori1, kernel);

        //Save new image
        saveToInternalStorage(noiseReducedCannyHori1, "noiseReducedCannyHori1.jpg");

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

        double longest = 0;
        int index = 0;

        for(int i = 0; i < hullList.size(); i++) {
            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(longest < perimeter) {
                longest = perimeter;
                index = i;
            }
        }

        Mat noiseReducedCannyHori2 = new Mat(noiseReducedCannyHori1.size(), CvType.CV_8UC1);
        Imgproc.drawContours(noiseReducedCannyHori2, contours, index, scalarWhite, 1);

        saveToInternalStorage(noiseReducedCannyHori2, "noiseReducedCannyHori2.jpg");

        //Identify points to get shelf height
        Point left = new Point();
        Point right = new Point();

        double[] rgb;
        int row = noiseReducedCannyHori2.rows() - 1;
        int col = 0;
        rgb = noiseReducedCannyHori2.get(row , col);
        while(rgb[0] == 0) {
            row--;
            if(row == 0) {
                row = noiseReducedCannyHori2.rows() - 1;
                col++;
            }
            rgb = noiseReducedCannyHori2.get(row, col);
        }
        left.x = row;
        left.y = col;

        row = noiseReducedCannyHori2.rows() - 1;
        col = noiseReducedCannyHori2.cols() - 1;
        rgb = noiseReducedCannyHori2.get(row, col);
        while(rgb[0] == 0) {
            row--;
            if(row == 0) {
                row = noiseReducedCannyHori2.rows() - 1;
                col--;
            }
            rgb = noiseReducedCannyHori2.get(row, col);
        }
        right.x = row;
        right.y = col;



        List<Point> points = new ArrayList<>();
        points.add(left);
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
        points.add(right);

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
        int cropBottomRow = (int) left.x - shelfHeight;
        int cropTopRow = (int) (cropBottomRow - shelfHeight * TAG_HEIGHT_FACTOR);

        cropBottomRow = (int) (cropBottomRow / SCALE_FACTOR);
        cropTopRow = (int) (cropTopRow / SCALE_FACTOR);

        Mat tagStrip = edges.submat(cropTopRow, cropBottomRow, 0, edges.cols() - 1);
        saveToInternalStorage(tagStrip, "tagStrip.jpg");

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
            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter > MIN_CONTOUR_PERIMETER) {
                Imgproc.drawContours(noiseReducedTagStrip1, contours, i, scalarWhite);
            }
        }

        Mat noiseReducedTagStrip2 = new Mat(noiseReducedTagStrip1.size(), CvType.CV_8UC1);

        Mat linesDrawn = new Mat(noiseReducedTagStrip2.size(), CvType.CV_8UC1);

        int verticalSize = noiseReducedTagStrip1.rows() / 150;
        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, verticalSize));
        Imgproc.erode(noiseReducedTagStrip1, noiseReducedTagStrip1, verticalStructure);
        Imgproc.dilate(noiseReducedTagStrip1, noiseReducedTagStrip1, verticalStructure);

        kernel = Mat.ones(50, 1, CvType.CV_8UC1);
        Imgproc.dilate(noiseReducedTagStrip1, noiseReducedTagStrip1, kernel);
        Imgproc.erode(noiseReducedTagStrip1, noiseReducedTagStrip1, kernel);

        saveToInternalStorage(noiseReducedTagStrip1, "noiseReducedTagStrip1.jpg");

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
            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter > MIN_VERTICAL_LINE) {
                Imgproc.drawContours(noiseReducedTagStrip2, contours, i, scalarWhite);
            }
        }

        saveToInternalStorage(noiseReducedTagStrip2, "noiseReducedTagStrip2.jpg");

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

            if(angle > ANGLE_CUTOFF) {

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

                Imgproc.line(linesDrawn, new Point(l[0], l[1]), new Point(l[2], l[3]), scalarWhite, HOUGH_LINE_THICKNESS, Imgproc.LINE_AA);
            }
        }

        saveToInternalStorage(linesDrawn, "linesDrawn.jpg");

        ArrayList<Integer> coordinates = new ArrayList<>();
        coordinates.add(0);
        for(int i = 0; i < linesDrawn.cols(); i++) {
            rgb = linesDrawn.get(linesDrawn.rows() - 20, i);
            if(rgb[0] != 0 && i - coordinates.get(coordinates.size() - 1) > MIN_TAG_WIDTH) {
                coordinates.add(i);
            }
        }
        coordinates.add(linesDrawn.cols() - 1);

        Mat croppedImage;
        double[] hsv;
        int whiteInARow;
        int topSecond = 0;
        int bottomSecond = 0;
        Rect r;
        Mat resizedImage = new Mat();

        for(int i = 0; i < coordinates.size() - 1; i++) {
            r = new Rect();
            if(coordinates.get(i + 1) - coordinates.get(i) > MIN_TAG_WIDTH) {
                if(coordinates.get(i + 1) + CROP_BUFFER > original.cols()) {
                    croppedImage = original.submat(cropTopRow, cropBottomRow, coordinates.get(i), coordinates.get(i + 1));
                    Imgproc.resize(croppedImage, resizedImage, new Size(), 0.25, 0.25);
                    Imgproc.resize(croppedImage, croppedImage, new Size(), 0.5, 0.5);

                    r.x = (int) (coordinates.get(i) * DISPLAY_SCALE_FACTOR);
                    r.y = (int) (cropTopRow * DISPLAY_SCALE_FACTOR);
                    r.height = (int) ((cropBottomRow - cropTopRow) * DISPLAY_SCALE_FACTOR);
                    r.width = (int) ((coordinates.get(i + 1) - coordinates.get(i)) * DISPLAY_SCALE_FACTOR);

                    rectangles.add(r);

                    Imgproc.cvtColor(resizedImage, resizedImage, Imgproc.COLOR_RGBA2BGR);
                    Imgproc.cvtColor(resizedImage, resizedImage, Imgproc.COLOR_BGR2HSV);
                    Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_RGBA2BGR);
                    Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_BGR2HSV);

                    top:
                    for(int x = 0; x < resizedImage.rows(); x++) {
                        whiteInARow = 0;
                        for(int y = 0; y < resizedImage.cols(); y++) {
                            hsv = resizedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                whiteInARow++;
                            } else {
                                if(whiteInARow >= resizedImage.cols() / 2) {
                                    topSecond = x;
                                    break top;
                                }
                                whiteInARow = 0;
                            }
                        }
                    }

                    bottom:
                    for(int x = resizedImage.rows() - 1; x >= 0; x--) {
                        whiteInARow = 0;
                        for(int y = 0; y < resizedImage.cols(); y++) {
                            hsv = resizedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                whiteInARow++;
                            } else {
                                if(whiteInARow >= resizedImage.cols() / 2) {
                                    bottomSecond = x;
                                    break bottom;
                                }
                                whiteInARow = 0;
                            }
                        }
                    }

                    bottomSecond *= 2;
                    topSecond *= 2;

                    if(topSecond != bottomSecond) {
                        croppedImage = croppedImage.submat(topSecond, bottomSecond, 0, croppedImage.cols() - 1);
                    }

                    for(int x = 0; x < croppedImage.rows(); x++) {
                        for(int y = 0; y < croppedImage.cols(); y++) {
                            hsv = croppedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                croppedImage.put(x, y, white);
                            } else {
                                croppedImage.put(x, y, black);
                            }
                        }
                    }

                    croppedTags.add(croppedImage);
                } else {
                    croppedImage = original.submat(cropTopRow, cropBottomRow, coordinates.get(i), coordinates.get(i + 1) + CROP_BUFFER);
                    Imgproc.resize(croppedImage, resizedImage, new Size(), 0.3, 0.3);
                    Imgproc.resize(croppedImage, croppedImage, new Size(), 0.5, 0.5);

                    r.x = (int) (coordinates.get(i) * DISPLAY_SCALE_FACTOR);
                    r.y = (int) (cropTopRow * DISPLAY_SCALE_FACTOR);
                    r.height = (int) ((cropBottomRow - cropTopRow) * DISPLAY_SCALE_FACTOR);
                    r.width = (int) ((coordinates.get(i + 1) - coordinates.get(i)) * DISPLAY_SCALE_FACTOR);

                    rectangles.add(r);

                    Imgproc.cvtColor(resizedImage, resizedImage, Imgproc.COLOR_RGBA2BGR);
                    Imgproc.cvtColor(resizedImage, resizedImage, Imgproc.COLOR_BGR2HSV);
                    Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_RGBA2BGR);
                    Imgproc.cvtColor(croppedImage, croppedImage, Imgproc.COLOR_BGR2HSV);


                    top:
                    for(int x = 0; x < resizedImage.rows(); x++) {
                        whiteInARow = 0;
                        for(int y = 0; y < resizedImage.cols(); y++) {
                            hsv = resizedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                whiteInARow++;
                            } else {
                                if(whiteInARow >= resizedImage.cols() / 2) {
                                    topSecond = x;
                                    break top;
                                }
                                whiteInARow = 0;
                            }
                        }
                    }

                    bottom:
                    for(int x = resizedImage.rows() - 1; x >= 0; x--) {
                        whiteInARow = 0;
                        for(int y = 0; y < resizedImage.cols(); y++) {
                            hsv = resizedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
                                whiteInARow++;
                            } else {
                                if(whiteInARow >= resizedImage.cols() / 2) {
                                    bottomSecond = x;
                                    break bottom;
                                }
                                whiteInARow = 0;
                            }
                        }
                    }

                    bottomSecond = bottomSecond * 5 / 3;
                    topSecond = topSecond * 5 / 3;

                    if(topSecond != bottomSecond) {
                        croppedImage = croppedImage.submat(topSecond, bottomSecond, 0, croppedImage.cols() - 1);
                    }

                    for(int x = 0; x < croppedImage.rows(); x++) {
                        for(int y = 0; y < croppedImage.cols(); y++) {
                            hsv = croppedImage.get(x, y);
                            if(hsv[2] - hsv[1] >= 115) {
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



        for(int i = 0; i < croppedTags.size(); i++) {
            if(i < 10) {
                saveToInternalStorage(croppedTags.get(i), "image" + i + ".jpg");
            } else {
                saveToInternalStorage(croppedTags.get(i), "image_" + i + ".jpg");
            }
        }



        OCRThread ocrThread = new OCRThread();
        ocrThread.start();
    }

    class OCRThread extends Thread {

        FirebaseVisionImage image;
        Bitmap bitmap;
        FirebaseVisionTextRecognizer detector = FirebaseVision.getInstance().getOnDeviceTextRecognizer();

        @Override
        public void run() {
            for(int i = 0; i < croppedTags.size(); i++) {
                Mat mat = croppedTags.get(i);
                bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mat, bitmap);
                image = FirebaseVisionImage.fromBitmap(bitmap);
                Task<FirebaseVisionText> task = detector.processImage(image);

                try {
                    String text = Tasks.await(task).getText();
                    text = text.toLowerCase();
                    text = text.replaceAll("\\s","");
                    text = text.replace('.', ' ');
                    text = text.replace(',', ' ');

                    if(text.length() >= 8 && text.length() <= 21) {
                        //char firstChar = text.charAt(0);
                        //while(firstChar != '0' && firstChar != '1' && firstChar != '2' && firstChar != '3' && firstChar != '4' &&
                        //        firstChar != '5' && firstChar != '6' && firstChar != '7' && firstChar != '8' && firstChar != '9') {
                        //    text = text.substring(1);
                        //}

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
                        drawRect(displayImage, rectangles.get(i), new Scalar(0, 0 ,0), 5);
                        Log.i("SUCCESS", "Index: " + i);
                    }
                } catch (ExecutionException e) {
                    drawRect(displayImage, rectangles.get(i), scalarYellow, 5);
                    Log.i("SUCCESS", "Index: " + i);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }

            for(int i = 0; i < ocrRecognized.size() - 1; i++) {
                if(i == 0) {
                    drawRect(displayImage, goodRectangles.get(i), scalarGreen, 5);
                }
                if(ocrRecognized.get(i).compareTo(ocrRecognized.get(i+1)) <= 0) {
                    drawRect(displayImage, goodRectangles.get(i + 1), scalarGreen, 5);
                } else {
                    drawRect(displayImage, goodRectangles.get(i + 1), scalarRed, 5);
                }
            }
            
            endTime = System.nanoTime();
            
            Log.i("TIME", "RUNTIME: " + (endTime - startTime));

            saveToInternalStorage(displayImage, "displayImage.jpg");
        }
    }

    private void checkLibraryLoads() {
        if(OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "OpenCV successfully loaded!");
        } else {
            Log.i("OpenCV", "OpenCV failed to load!");
        }
    }

    private Mat fileToMat(String name) {
        InputStream stream = null;
        Uri uri = Uri.parse("android.resource://com.andrewwang.linedetectionv3/drawable/" + name);
        try {
            stream = getContentResolver().openInputStream(uri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
        bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap imgBmp = BitmapFactory.decodeStream(stream, null, bmpFactoryOptions);
        Mat image = new Mat();
        Utils.bitmapToMat(imgBmp, image);
        return image;
    }

    private Mat colorToGray(Mat color) {
        Mat gray = new Mat();
        Imgproc.cvtColor(color, gray, Imgproc.COLOR_RGB2GRAY);
        return gray;
    }


    private void saveToInternalStorage(Mat mat, String filename){
        Bitmap bitmapImage = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmapImage);
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
