package com.andrewwang.libraryreshelfappv3;

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
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE = 1;
    private static final int HSV_BINARY_THRESHOLD = 115;
    private static final int MIN_AREA_THRESHOLD = 500; //500 initially
    private static final int MAX_AREA_THRESHOLD = 1600; //1600 initially
    private static final double MIN_RATIO_THRESHOLD = 2.5; //4.5 initially
    private static final double MAX_RATIO_THRESHOLD = 12; //9.5 initially

    private static final double WIDTH_EXTENSION = 1.9;
    private static final double HEIGHT_EXTENSION = 1.9; //1.7 initially

    private static final int MIN_RECTANGLE_PERIMETER = 275;
    private static final int MIN_RECTANGLE_AREA = 10000;

    private static final double CROP_TOP_HEIGHT = 3.0 / 4.0;
    private static final double CROP_BOTTOM_HEIGHT = 12.0 / 13.0;
    private static final int CROP_BUFFER = 10;

    double[] black = new double[] {0, 0, 0};
    double[] white = new double[] {255, 255, 255};

    Scalar whiteScalar = new Scalar(255, 255, 255);
    Scalar blackScalar = new Scalar(0, 0, 0);
    Scalar greenScalar = new Scalar(0, 255, 0);
    Scalar redScalar = new Scalar(255, 0, 0);

    ArrayList<RotatedRect> tagsOrderedFiltered = new ArrayList<>();
    ArrayList<Mat> croppedImagesOriginal = new ArrayList<>();
    ArrayList<Mat> croppedImagesBinary = new ArrayList<>();
    ArrayList<Mat> croppedImagesGray = new ArrayList<>();
    ArrayList<String> ocrRecognized = new ArrayList<>();

    //FILE STUFF
    String currentPhotoPath;
    File photoFile;

    //LAYOUT STUFF
    ImageView imageView;
    RelativeLayout layout;
    TextView textView;

    //OPENCV IMAGE MANIPULATION
    Mat displayImage;
    int cropTop;
    int cropBottom;

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
            public void onClick(View v) {
                textView.setVisibility(View.GONE);
                getImage();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        Bitmap bitmapOriginal = BitmapFactory.decodeFile(currentPhotoPath);

        Mat original = new Mat();
        Utils.bitmapToMat(bitmapOriginal, original);

        displayImage = original.clone();

        Imgproc.cvtColor(displayImage, displayImage, Imgproc.COLOR_RGBA2RGB);

        cropTop = (int) (original.rows() * CROP_TOP_HEIGHT);
        cropBottom = (int) (original.rows() * CROP_BOTTOM_HEIGHT);

        original = original.submat(cropTop, cropBottom, 0, original.cols());

        Mat cropStrip = original.clone();

        Imgproc.cvtColor(original, original, Imgproc.COLOR_RGBA2BGR);
        Imgproc.cvtColor(original, original, Imgproc.COLOR_BGR2HSV);

        //Binary threshold
        double[] hsv;
        for(int i = 0; i < original.rows(); i++) {
            for(int j = 0; j < original.cols(); j++) {
                hsv = original.get(i, j);
                if(hsv[2] - hsv[1] >= HSV_BINARY_THRESHOLD) {
                    original.put(i, j, white);
                } else {
                    original.put(i, j, black);
                }
            }
        }

        Mat binary = original.clone();

        //Convert to grayscale
        Imgproc.cvtColor(original, original, Imgproc.COLOR_HSV2BGR);
        Imgproc.cvtColor(original, original, Imgproc.COLOR_BGR2GRAY);

        //Get contours from image
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(original, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat filterStrip = original.clone();
        Imgproc.cvtColor(filterStrip, filterStrip, Imgproc.COLOR_GRAY2RGB);

        //Get letter by letter from tags and fill them green
        RotatedRect rotatedRect;
        for(MatOfPoint contour : contours) {
            rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
            double perimeter = (rotatedRect.size.height + rotatedRect.size.width) * 2;
            if((rotatedRect.size.area() > MIN_AREA_THRESHOLD
                    && rotatedRect.size.area() < MAX_AREA_THRESHOLD
                    && rotatedRect.size.area() / perimeter > MIN_RATIO_THRESHOLD
                    && rotatedRect.size.area() / perimeter < MAX_RATIO_THRESHOLD)
                    || ((rotatedRect.size.height > 25)
                    && (rotatedRect.size.height < 50)
                    && (rotatedRect.size.width < 15))
                    || ((rotatedRect.size.width > 25)
                    && (rotatedRect.size.width < 50)
                    && (rotatedRect.size.height < 15))) {
                rotatedRect.size.width = rotatedRect.size.width * WIDTH_EXTENSION;
                rotatedRect.size.height = rotatedRect.size.height * HEIGHT_EXTENSION;
                drawRotatedRect(filterStrip, rotatedRect, greenScalar, -1);
            }
        }

        //Remove everything but blobs
        double[] rgb;
        for(int i = 0; i < filterStrip.rows(); i++) {
            for(int j = 0; j < filterStrip.cols(); j++) {
                rgb = filterStrip.get(i, j);
                if(rgb[1] != 255) {
                    filterStrip.put(i, j, black);
                }
            }
        }

        //Make gray
        Imgproc.cvtColor(filterStrip, filterStrip, Imgproc.COLOR_RGB2GRAY);

        contours.clear();
        Imgproc.findContours(filterStrip, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

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

        //New mat where noise is removed from the edges
        Mat noiseReducedStrip = Mat.zeros(filterStrip.size(), CvType.CV_8UC1);

        //Draw blobs to new Mat
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(noiseReducedStrip, hullList, i, whiteScalar, -1);
        }

        //Paint all sides black
        for(int i = 0; i < noiseReducedStrip.rows(); i++) {
            noiseReducedStrip.put(i, 0, black);
            noiseReducedStrip.put(i, noiseReducedStrip.cols() - 1, black);
        }
        for(int i = 0; i < noiseReducedStrip.cols(); i++) {
            noiseReducedStrip.put(0, i, black);
            noiseReducedStrip.put(noiseReducedStrip.rows() - 1, i, black);
        }

        //Remove extra rectangles
        contours.clear();
        Imgproc.findContours(noiseReducedStrip, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        hullList.clear();
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
        //If rectangle is too small, remove it from the image
        for (int i = 0; i < contours.size(); i++) {
            double perimeter = Imgproc.arcLength(new MatOfPoint2f(hullList.get(i).toArray()), true);
            if(perimeter < MIN_RECTANGLE_PERIMETER) {
                Imgproc.drawContours(noiseReducedStrip, hullList, i, blackScalar, -1);
            }
        }

        //Add all the tags to a tags array
        contours.clear();
        Imgproc.findContours(noiseReducedStrip, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        ArrayList<RotatedRect> tags = new ArrayList<>();
        for(MatOfPoint contour : contours) {
            rotatedRect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));

            if(rotatedRect.size.area() > MIN_RECTANGLE_AREA && (Math.abs((rotatedRect.angle + 360) % 90) <= 10 || Math.abs((rotatedRect.angle + 360) % 90) >= 80)) {
                drawRotatedRect(noiseReducedStrip, rotatedRect, whiteScalar, -1);
                tags.add(rotatedRect);
            } else {
                rotatedRect.size.height += CROP_BUFFER;
                rotatedRect.size.width += CROP_BUFFER;
                drawRotatedRect(noiseReducedStrip, rotatedRect, blackScalar, -1);
            }
        }


        //Put the tags in left to right order, omit the first and last detected "tags"
        ArrayList<RotatedRect> tagsOrdered = new ArrayList<>();
        int indexOfLeftMost = 0;
        while(tags.size() != 1) {
            RotatedRect leftMost = tags.get(0);
            for(int i = 1; i < tags.size(); i++) {
                RotatedRect r = tags.get(i);
                if(r.center.x < leftMost.center.x) {
                    leftMost = r;
                    indexOfLeftMost = i;
                }
            }
            tagsOrdered.add(leftMost);
            tags.remove(indexOfLeftMost);
            indexOfLeftMost = 0;
        }
        tagsOrdered.remove(0);

        Point[] points = new Point[4];
        Point topLeft;
        Point bottomRight;
        Point topLeft1;

        //Filter the tags by looking for doubles
        while(tagsOrdered.size() != 1) {
            RotatedRect r1 = tagsOrdered.get(0);
            RotatedRect r2 = tagsOrdered.get(1);
            r1.points(points);
            if(r1.angle <= 10 && r1.angle >= -10) {
                bottomRight = points[3];
            } else if (r1.angle <= -80 && r1.angle >= -100) {
                bottomRight = points[0];
            } else if (r1.angle >= 80 && r1.angle <= 100) {
                bottomRight = points[2];
            } else {
                bottomRight = points[1];
            }

            r2.points(points);
            if(r2.angle <= 10 && r2.angle >= -10) {
                topLeft1 = points[1];
            } else if (r2.angle <= -80 && r2.angle >= -100) {
                topLeft1 = points[2];
            } else if (r2.angle >= 80 && r2.angle <= 100) {
                topLeft1 = points[0];
            } else {
                topLeft1 = points[3];
            }

            if(bottomRight.x < topLeft1.x) {
                tagsOrderedFiltered.add(r1);
                tagsOrdered.remove(0);
            } else {
                if(r2.center.y < r1.center.y) {
                    tagsOrdered.remove(1);
                } else {
                    tagsOrdered.remove(0);
                }
            }
        }
        tagsOrderedFiltered.add(tagsOrdered.get(0));

        //Crop tags from original, gray, and binary images
        Mat croppedImage = new Mat();
        for(RotatedRect r : tagsOrderedFiltered) {
            r.points(points);
            if(r.angle <= 10 && r.angle >= -10) {
                topLeft = points[1];
                bottomRight = points[3];
            } else if (r.angle <= -80 && r.angle >= -100) {
                topLeft = points[2];
                bottomRight = points[0];
            } else if (r.angle >= 80 && r.angle <= 100) {
                topLeft = points[0];
                bottomRight = points[2];
            } else {
                topLeft = points[3];
                bottomRight = points[1];
            }

            if(topLeft.x < 0) {
                topLeft.x = 0;
            }

            if(bottomRight.x > binary.cols()) {
                bottomRight.x = binary.cols();
            }

            if(topLeft.y < 0) {
                topLeft.y = 0;
            }

            if(bottomRight.y > binary.rows()) {
                bottomRight.y = binary.rows();
            }

            if(topLeft.y > bottomRight.y) {
                int big = (int) topLeft.y;
                int small = (int) bottomRight.y;
                topLeft.y = small;
                bottomRight.y = big;
            }

            if(topLeft.x > bottomRight.x) {
                int big = (int) topLeft.x;
                int small = (int) bottomRight.x;
                topLeft.x = small;
                bottomRight.x  = big;
            }

            Log.i("CROP", topLeft.y + ", " + bottomRight.y + ", " + topLeft.x + ", " + bottomRight.x);

            croppedImage = cropStrip.submat((int) topLeft.y, (int) bottomRight.y, (int) topLeft.x, (int) bottomRight.x);
            croppedImagesOriginal.add(croppedImage);
            croppedImage = original.submat((int) topLeft.y, (int) bottomRight.y, (int) topLeft.x, (int) bottomRight.x);
            croppedImagesGray.add(croppedImage);
            croppedImage = binary.submat((int) topLeft.y, (int) bottomRight.y, (int) topLeft.x, (int) bottomRight.x);
            croppedImagesBinary.add(croppedImage);
        }

        //Rotate the tags as needed
        for(int i = 0; i < croppedImagesOriginal.size(); i++) {
            if(croppedImagesOriginal.get(i).height() > croppedImagesOriginal.get(i).width() * 1.5) {
                Core.rotate(croppedImagesOriginal.get(i), croppedImagesOriginal.get(i), Core.ROTATE_90_CLOCKWISE);
                Core.rotate(croppedImagesBinary.get(i), croppedImagesBinary.get(i), Core.ROTATE_90_CLOCKWISE);
                Core.rotate(croppedImagesGray.get(i), croppedImagesGray.get(i), Core.ROTATE_90_CLOCKWISE);
            }
        }

        croppedImage.release();
        filterStrip.release();
        noiseReducedStrip.release();
        original.release();
        cropStrip.release();
        binary.release();

        bitmapOriginal.recycle();

        OCRThread ocrThread = new OCRThread();
        ocrThread.start();
    }

    class OCRThread extends Thread {

        @Override
        public void run() {

            FirebaseVisionTextRecognizer detector = FirebaseVision.getInstance().getOnDeviceTextRecognizer();

            Log.i("SIZE", Integer.toString(croppedImagesOriginal.size()));
            Log.i("SIZE", Integer.toString(tagsOrderedFiltered.size()));

            //Read text from cropped images
            for(int i = 0; i < croppedImagesOriginal.size(); i++) {
                Mat mat = croppedImagesOriginal.get(i);
                String text = readText(mat, detector);

                Log.i("TEXT", text);

                if(!text.equals("ERROR") && text.length() > 4) {
                    ocrRecognized.add(text);
                } else {
                    tagsOrderedFiltered.remove(i);
                }

                mat.release();
            }

            //Transfer the rectangles from strip to displayImage
            for(int i = 0; i < tagsOrderedFiltered.size(); i++) {
                RotatedRect shifted = tagsOrderedFiltered.get(i);
                shifted.center.y = shifted.center.y + cropTop;
                tagsOrderedFiltered.set(i, shifted);
            }

            //Run comparisons
            int dewyOrder;
            int secondOrder;

            //Assume that the first is correct
            drawRotatedRect(displayImage, tagsOrderedFiltered.get(0), greenScalar, 5);

            //Compare the first of the tags
            for(int i = 1; i < ocrRecognized.size(); i++) {
                String right = ocrRecognized.get(i);
                String left = ocrRecognized.get(i - 1);

                if(right.contains("\n") && left.contains("\n")) {
                    String textFirstRight = right.substring(0, right.indexOf('\n'));
                    String textFirstLeft = left.substring(0, left.indexOf('\n'));
                    dewyOrder = textFirstRight.compareTo(textFirstLeft);

                    if(dewyOrder > 0) {
                        drawRotatedRect(displayImage, tagsOrderedFiltered.get(i), greenScalar, 5);
                    } else if (dewyOrder < 0) {
                        String rightBinary = readText(croppedImagesBinary.get(i), detector);
                        String leftBinary = readText(croppedImagesBinary.get(i - 1), detector);

                        String rightGray = readText(croppedImagesGray.get(i), detector);
                        String leftGray = readText(croppedImagesGray.get(i - 1), detector);

                        int dewyOrderBinary = rightBinary.substring(0, rightBinary.indexOf("\n")).compareTo(leftBinary.substring(0, leftBinary.indexOf("\n")));
                        int secondOrderBinary = rightBinary.substring(rightBinary.indexOf("\n") + 1).compareTo(leftBinary.substring(leftBinary.indexOf("\n") + 1));

                        int dewyOrderGray = rightGray.substring(0, rightGray.indexOf("\n")).compareTo(leftGray.substring(0, leftGray.indexOf("\n")));
                        int secondOrderGray = rightGray.substring(rightGray.indexOf("\n") + 1).compareTo(leftGray.substring(leftGray.indexOf("\n") + 1));

                        if(dewyOrderBinary > 0 || dewyOrderGray > 0) {
                            drawRotatedRect(displayImage, tagsOrderedFiltered.get(i), greenScalar, 5);
                        } else if(dewyOrderBinary < 0 && dewyOrderGray < 0) {
                            drawRotatedRect(displayImage, tagsOrderedFiltered.get(i), redScalar, 5);
                        } else {
                            if(secondOrderBinary >= 0 || secondOrderGray >= 0) {
                                drawRotatedRect(displayImage, tagsOrderedFiltered.get(i), greenScalar, 5);
                            } else {
                                drawRotatedRect(displayImage, tagsOrderedFiltered.get(i), redScalar, 5);
                            }
                        }
                    } else {
                        secondOrder = right.substring(right.indexOf("\n") + 1).compareTo(left.substring(left.indexOf("\n") + 1));
                        if(secondOrder >= 0) {
                            drawRotatedRect(displayImage, tagsOrderedFiltered.get(i), greenScalar, 5);
                        } else {
                            String rightBinary = readText(croppedImagesBinary.get(i), detector);
                            String leftBinary = readText(croppedImagesBinary.get(i - 1), detector);

                            String rightGray = readText(croppedImagesGray.get(i), detector);
                            String leftGray = readText(croppedImagesGray.get(i - 1), detector);

                            int secondOrderBinary = rightBinary.substring(rightBinary.indexOf("\n") + 1).compareTo(leftBinary.substring(leftBinary.indexOf("\n") + 1));
                            int secondOrderGray = rightGray.substring(rightGray.indexOf("\n") + 1).compareTo(leftGray.substring(leftGray.indexOf("\n") + 1));

                            if(secondOrderBinary >= 0 || secondOrderGray >= 0) {
                                drawRotatedRect(displayImage, tagsOrderedFiltered.get(i), greenScalar, 5);
                            } else {
                                drawRotatedRect(displayImage, tagsOrderedFiltered.get(i), redScalar, 5);
                            }
                        }
                    }
                }
            }

            //Close the detector
            try {
                detector.close();
            } catch(IOException e) {
                e.printStackTrace();
            }

            //Update all views and clear ArrayLists
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Bitmap bitmap = Bitmap.createBitmap(displayImage.cols(), displayImage.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(displayImage, bitmap);
                    imageView.setImageBitmap(bitmap);

                    tagsOrderedFiltered.clear();
                    croppedImagesOriginal.clear();
                    croppedImagesBinary.clear();
                    croppedImagesGray.clear();
                    ocrRecognized.clear();

                    displayImage.release();
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
                Uri photoURI = FileProvider.getUriForFile(this, "andrewwang.provider.v2", photoFile);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(intent, REQUEST_CODE);
            }
        }
    }

    private void drawRotatedRect(Mat image, RotatedRect rotatedRect, Scalar color, int thickness) {
        Point[] vertices = new Point[4];
        rotatedRect.points(vertices);
        MatOfPoint points = new MatOfPoint(vertices);
        Imgproc.drawContours(image, Arrays.asList(points), -1, color, thickness);
    }

    private String readText(Mat mat, FirebaseVisionTextRecognizer detector) {
        Bitmap bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        FirebaseVisionImage image = FirebaseVisionImage.fromBitmap(bitmap);
        Task<FirebaseVisionText> task = detector.processImage(image);

        try {
            String text = Tasks.await(task).getText();
            text = text.toLowerCase();
            text = text.replace('.', ' ');
            text = text.replace(',', ' ');

            for(int x = 0; x < text.length(); x++) {
                char c = text.charAt(x);
                if(c != ' ' && c != 'a' && c != 'b' && c != 'c' && c != 'd' && c != 'e' && c != 'f' && c != 'g' && c != 'h' && c != 'i'
                        && c != 'j' && c != 'k' && c != 'l' && c != 'm' && c != 'n' && c != 'o' && c != 'p' && c != 'q' && c != 'r' && c != 's'
                        && c != 't' && c != 'u' && c != 'v' && c != 'w' && c != 'x' && c != 'y' && c != 'z' && c != '0' && c != '1' && c != '2'
                        && c != '3' && c != '4' && c != '5' && c != '6' && c != '7' && c != '8' && c != '9' && c != '\n') {
                    text = text.replace(c, ' ');
                }
                text = text.replaceAll(" ", "");
            }

            String top = "";

            if(text.contains("\n")) {
                top = text.substring(0, text.indexOf('\n'));

                int numbers = 0;
                int letters = 0;

                for(int j = 0; j < top.length(); j++) {
                    if(top.charAt(j) >= 'a' && top.charAt(j) <= 'z') {
                        letters++;
                    } else {
                        numbers++;
                    }
                }

                if(numbers > letters && letters != 0) {
                    top = top.replace('s', '5');
                    top = top.replace('o', '0');
                    top = top.replace('g', '9');
                    top = top.replace('z', '2');
                    top = top.replace('l', '1');
                    top = top.replace('b', '6');
                } else if (letters > numbers && numbers != 0) {
                    top = top.replace('5', 's');
                    top = top.replace('0', 'o');
                    top = top.replace('9', 'g');
                    top = top.replace('2', 'z');
                    top = top.replace('1', 'l');
                    top = top.replace('6', 'b');
                }
            }
            text = top + "\n" + text.substring(text.indexOf("\n") + 1);

            bitmap.recycle();
            return text;
        } catch (ExecutionException e) {
            e.printStackTrace();
            bitmap.recycle();
            return "ERROR";
        } catch (InterruptedException e) {
            e.printStackTrace();
            bitmap.recycle();
            return "ERROR";
        }
    }
}
