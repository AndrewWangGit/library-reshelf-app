package com.andrewwang.linedetectionv2;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.util.SparseArray;
import android.widget.TextView;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
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

public class MainActivity extends AppCompatActivity {

    private static final String FILE = "image9";
    private static final double SCALE_FACTOR = 0.25;
    private static final int THRESHOLD1_CANNY = 1;
    private static final int THRESHOLD2_CANNY = 200;

    private static final int THRESHOLD_HOUGH = 15;

    private static final Size KERNEL_SIZE = new Size(1,1);
    private static final int THICKNESS_CONTOURS = 20;
    private static final int THICKNESS_RED_LINE = 1;

    private static final int MIN_LINE_LENGTH_ROUND_2 = 75;
    private static final int ANGLE_CUTOFF = 75;

    private static final int CROP_BUFFER = 0;
    private static final int MIN_TAG_WIDTH = 50;
    private static final int COLOR_DIFF_THRESH = 30;

    private static final double FACTOR_PAGE_DOWN = 3.0 / 4.0;

    TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textView = findViewById(R.id.textView);

        checkLibraryLoads();

        //Get gray image and create new scaled Mat
        Mat original = fileToMat(FILE);

        Mat gray = colorToGray(original);
        Mat grayScaled = new Mat();

        //Set the scaled Mat and blur it
        Imgproc.resize(gray, grayScaled, new Size(), SCALE_FACTOR, SCALE_FACTOR);
        Imgproc.GaussianBlur(grayScaled, grayScaled, KERNEL_SIZE, 0, 0);

        //Get the edges from the grayScaled Mat
        Mat edges = new Mat();
        Imgproc.Canny(grayScaled, edges, THRESHOLD1_CANNY, THRESHOLD2_CANNY);

        //Save edges to internal storage for viewing
        saveToInternalStorage(edges, "edges.jpg");

        //Find the contours from the edges detected
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edges, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

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
        Scalar color = new Scalar(255, 255, 255); //white

        //New mat where noise is removed from the edges
        Mat noiseReducedCanny = Mat.zeros(edges.size(), CvType.CV_8UC1);

        //Do actual filtering
        for (int i = 0; i < contours.size(); i++) {
            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter > grayScaled.size().height) {
                Imgproc.drawContours(noiseReducedCanny, contours, i, color, THICKNESS_CONTOURS);
            }
        }

        //Save first iteration to internal storage
        saveToInternalStorage(noiseReducedCanny, "noiseReducedCanny1.jpg");

        //Line thinning
        double[] rgbCanny;
        double[] white = new double[]{255};
        double[] black = new double[]{0};
        int inARow = 0;

        for(int i = 0; i < noiseReducedCanny.size().height; i++) {
            for(int j = 0; j < noiseReducedCanny.size().width; j++) {
                rgbCanny = noiseReducedCanny.get(i, j);
                if(rgbCanny[0] == 255) {
                    inARow++;
                } else {
                    if(inARow != 0 && inARow % 2 == 1) {
                        int middle = (int) Math.ceil(j - inARow / 2.0);
                        for(int x = j - inARow; x < j; x++) {
                            noiseReducedCanny.put(i, x, black);
                        }
                        noiseReducedCanny.put(i, middle, white);
                    } else if (inARow != 0 && inARow % 2 == 0) {
                        int middle = j - inARow / 2;
                        for(int x = j - inARow; x < j; x++) {
                            noiseReducedCanny.put(i, x, black);
                        }
                        noiseReducedCanny.put(i, middle, white);
                    }
                    inARow = 0;
                }
            }
        }

        //Save second iteration to internal storage
        saveToInternalStorage(noiseReducedCanny, "noiseReducedCanny2.jpg");

        //Fill small segments to be black
        contours.clear();
        hullList.clear();
        Imgproc.findContours(noiseReducedCanny, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

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

        color = new Scalar(0, 0, 0); //black

        for (int i = 0; i < contours.size(); i++) {

            perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter < MIN_LINE_LENGTH_ROUND_2) {
                Imgproc.drawContours(noiseReducedCanny, hullList, i, color, -1);
            }
        }

        //Save final iteration to internal storage
        saveToInternalStorage(noiseReducedCanny, "noiseReducedCanny3.jpg");

        Mat lines = new Mat();
        Mat linesDrawn = new Mat(original.size(), CvType.CV_8UC3);

        Imgproc.HoughLinesP(noiseReducedCanny, lines, 1, Math.PI / 180, THRESHOLD_HOUGH, original.size().height / 12, original.size().height / 8);

        //Convert channels and Mat type to make red drawable on it
        Imgproc.cvtColor(original, original, Imgproc.COLOR_RGBA2RGB);
        original.convertTo(original, CvType.CV_8UC3);

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
                    l[3] = original.size().height - 5;
                } else {
                    yIntercept = l[1] - slope*l[0];
                    l[0] = (5 - yIntercept) / slope;
                    l[1] = 5;
                    l[2] = (original.size().height - 5 - yIntercept) / slope;
                    l[3] = original.size().height - 5;
                }

                Imgproc.line(linesDrawn, new Point(l[0] / SCALE_FACTOR, l[1] / SCALE_FACTOR), new Point(l[2] / SCALE_FACTOR, l[3] / SCALE_FACTOR), new Scalar(255, 0, 0), THICKNESS_RED_LINE, Imgproc.LINE_AA);
            }
        }

        saveToInternalStorage(linesDrawn, "linesDrawn.jpg");


        double[] rgbLinesDrawn;
        double[] rgbSubMat;

        double rbDiff;
        double rgDiff;
        double gbDiff;

        int cord1 = 0;
        int cord2;

        Mat subMat;
        ArrayList<Mat> croppedImages = new ArrayList<>();

        for(int i = 0; i < original.size().width; i++) {
            rgbLinesDrawn = linesDrawn.get((int) original.size().height / 3 * 2, i);

            if(rgbLinesDrawn[0] != 0) {
                cord2 = i;

                if(cord1 > CROP_BUFFER && (cord2 - cord1) > MIN_TAG_WIDTH) {
                    subMat = original.submat((int) (original.size().height * FACTOR_PAGE_DOWN), (int) original.size().height, cord1 - CROP_BUFFER, cord2 + CROP_BUFFER);

                    for(int x = 0; x < subMat.size().width; x++) {
                        for(int y = 0; y < subMat.size().height; y++) {
                            rgbSubMat = subMat.get(y, x);
                            rbDiff = Math.abs(rgbSubMat[0] - rgbSubMat[2]);
                            rgDiff = Math.abs(rgbSubMat[0] - rgbSubMat[1]);
                            gbDiff = Math.abs(rgbSubMat[1] - rgbSubMat[2]);
                            if(rbDiff > 20 || rgDiff > 20 || gbDiff > 20) {
                                subMat.put(y, x, new double[]{255, 255, 255});
                            }
                        }
                    }

                    croppedImages.add(subMat);
                } else if(cord1 <= CROP_BUFFER) {
                    subMat = original.submat((int) (original.size().height * FACTOR_PAGE_DOWN), (int) original.size().height, cord1, cord2 + CROP_BUFFER);

                    for(int x = 0; x < subMat.size().width; x++) {
                        for(int y = 0; y < subMat.size().height; y++) {
                            rgbSubMat = subMat.get(y, x);

                            rbDiff = Math.abs(rgbSubMat[0] - rgbSubMat[2]);
                            rgDiff = Math.abs(rgbSubMat[0] - rgbSubMat[1]);
                            gbDiff = Math.abs(rgbSubMat[1] - rgbSubMat[2]);
                            if(rbDiff > 20 || rgDiff > 20 || gbDiff > 20) {
                                subMat.put(y, x, new double[]{255, 255, 255});
                            }
                        }
                    }

                    croppedImages.add(subMat);
                }

                cord1 = i;
                i += 19;
            }

            if(i == (int) (original.size().width - 2)) {
                cord2 = (int) (original.size().width - 2);
                subMat = original.submat((int) (original.size().height * FACTOR_PAGE_DOWN), (int) original.size().height, cord1, cord2);

                for(int x = 0; x < subMat.size().width; x++) {
                    for(int y = 0; y < subMat.size().height; y++) {
                        rgbSubMat = subMat.get(y, x);
                        rbDiff = Math.abs(rgbSubMat[0] - rgbSubMat[2]);
                        rgDiff = Math.abs(rgbSubMat[0] - rgbSubMat[1]);
                        gbDiff = Math.abs(rgbSubMat[1] - rgbSubMat[2]);
                        if(rbDiff > 20 || rgDiff > 20 || gbDiff > 20) {
                            subMat.put(y, x, new double[]{255, 255, 255});
                        }
                    }
                }

                croppedImages.add(subMat);
            }

        }

        int imageCounter = 1;
        for(Mat mat : croppedImages) {

            if(imageCounter < 10) {
                saveToInternalStorage(mat, "image" + imageCounter + ".jpg");
            } else {
                saveToInternalStorage(mat, "image_" + imageCounter + ".jpg");
            }
            imageCounter++;
        }



        List<String> ocrRecognized = new ArrayList<>();
        Frame frame;
        SparseArray<TextBlock> items;
        TextRecognizer textRecognizer = new TextRecognizer.Builder(getApplicationContext()).build();
        StringBuilder stringBuilder;
        TextBlock item;
        Bitmap bitmap;

        if(!textRecognizer.isOperational()) {
            Log.i("ERROR", "Detector dependencies are not yet available!");
        } else {
            for(Mat mat : croppedImages) {
                bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mat, bitmap);

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

        for(String s : ocrRecognized) {
            if(s.equals("")) {
                textView.append("NO TEXT DETECTED");
                textView.append("\n--------------------------\n");
            } else {
                textView.append(s);
                textView.append("\n--------------------------\n");
            }
        }


        clearImgDir();
    }

    private static void checkLibraryLoads() {
        if(OpenCVLoader.initDebug()) {
            Log.i("OpenCV", "OpenCV successfully loaded!");
        } else {
            Log.i("OpenCV", "OpenCV failed to load!");
        }
    }

    private Mat fileToMat(String name) {
        InputStream stream = null;
        Uri uri = Uri.parse("android.resource://com.andrewwang.linedetectionv2/drawable/" + name);
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
        Bitmap bitmapImage = Bitmap.createBitmap((int) mat.size().width, (int) mat.size().height, Bitmap.Config.ARGB_8888);
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

    private void clearImgDir() {
        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        directory.delete();
    }
}
