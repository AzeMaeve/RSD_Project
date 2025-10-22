#include <stdio.h>
#include <stdlib.h>
#include <libserialport.h>
#include <chrono>

#define BAUD 9600

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Structure to store hole information
struct Hole {
    Point2f center;
    double area;
    int colour;
};

// Global variables for saved hole data
vector<Hole> savedHoles;
bool holesCalibrated = false;
bool calibrationMode = true; // Start in calibration mode

void printMenu() {
    cout << "\n=== Robot Control Menu ===" << endl;
    cout << "CALIBRATION MODE: Place empty board and press 'c'" << endl;
    cout << "1. Noughts & Crosses" << endl;
    cout << "2. Podium" << endl;
    cout << "3. Stack the Blocks" << endl;
    cout << "c. Calibrate Holes (detect empty board)" << endl;
    cout << "h. Home" << endl;
    cout << "s. Stop" << endl;
    cout << "r. Resume" << endl;
    cout << "q. Quit" << endl;
    cout << "Enter choice: ";
}

// Function to detect color at specific coordinates and return integer code
int detectColour(Mat& original, int x, int y) {
    // Check if coordinates are within image bounds
    if (x < 0 || x >= original.cols || y < 0 || y >= original.rows) {
        return 0;
    }
    
    // Convert to HSV for color detection
    Mat hsv;
    cvtColor(original, hsv, COLOR_BGR2HSV);

    // Get the pixel value at the specified coordinates
    Vec3b pixel = hsv.at<Vec3b>(y, x);

    int hue = pixel[0];
    int saturation = pixel[1];
    int value = pixel[2];

    // Check for red (wraps around 0-10 and 170-180 in HSV)
    if ((hue <= 10 || hue >= 170) && saturation > 100 && value > 50) {
        return 1; // Red
    }
    // Check for blue
    else if (hue >= 100 && hue <= 130 && saturation > 100 && value > 50) {
        return 2; // Blue
    }
    // Check for green
    else if (hue >= 40 && hue <= 80 && saturation > 100 && value > 50) {
        return 3; // Green
    }
    // Check for yellow
    else if (hue >= 20 && hue <= 35 && saturation > 100 && value > 50) {
        return 4; // Yellow
    }

    return 0; // No color detected
}

// Function to detect the largest dark object (board)
vector<Point> detectBoard(Mat& thresholded, Mat& original) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(thresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Find the largest contour
    double maxArea = 0;
    int maxAreaIdx = -1;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }

    if (maxAreaIdx >= 0 && maxArea > 10000) {
        // Draw the board contour
        drawContours(original, contours, maxAreaIdx, Scalar(0, 255, 255), 3);
        return contours[maxAreaIdx];
    }

    return vector<Point>();
}

// Function to detect holes within the board region during calibration
vector<Hole> calibrateHoles(Mat& thresholded, Mat& original, const vector<Point>& boardContour) {
    vector<Hole> holes;

    if (boardContour.empty()) return holes;

    // Create a mask for the board region
    Mat boardMask = Mat::zeros(thresholded.size(), CV_8UC1);
    vector<vector<Point>> boardContours = { boardContour };
    fillPoly(boardMask, boardContours, Scalar(255));

    // Find contours within the board mask
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // Use the inverted thresholded image to find holes (dark regions within board)
    Mat holesImage;
    bitwise_not(thresholded, holesImage);
    bitwise_and(holesImage, boardMask, holesImage);

    findContours(holesImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);

        // Filter by area - adjust these values based on your hole sizes
        if (area < 100 || area > 10000) continue;

        // Check if the contour is within the board
        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);

        // Verify the center is inside the board contour
        if (pointPolygonTest(boardContour, center, false) >= 0) {
            // Calculate circularity to identify round/square holes
            double perimeter = arcLength(contours[i], true);
            double circularity = 0;
            if (perimeter > 0) {
                circularity = (4 * CV_PI * area) / (perimeter * perimeter);
            }

            // Accept both circular and slightly elliptical shapes
            if (circularity > 0.5) {
                Hole hole;
                hole.center = center;
                hole.area = area;
                hole.colour = 0; // Default to no color during calibration
                holes.push_back(hole);

                // Draw the hole during calibration
                drawContours(original, contours, i, Scalar(0, 255, 0), 2);
                circle(original, center, 5, Scalar(0, 0, 255), -1);

                // Label the hole
                string label = "H" + to_string(holes.size()) + " (" +
                    to_string((int)center.x) + "," + to_string((int)center.y) + ")";
                putText(original, label, Point(center.x + 10, center.y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);
            }
        }
    }

    return holes;
}

// Function to save calibrated holes
void saveCalibratedHoles(const vector<Hole>& holes) {
    savedHoles = holes;
    holesCalibrated = true;
    calibrationMode = false;
    
    cout << "=== CALIBRATION COMPLETE ===" << endl;
    cout << "Saved " << savedHoles.size() << " holes:" << endl;
    for (size_t i = 0; i < savedHoles.size(); i++) {
        cout << "Hole " << (i + 1) << ": X=" << savedHoles[i].center.x
             << ", Y=" << savedHoles[i].center.y
             << ", Area=" << savedHoles[i].area << endl;
    }
    cout << "Now checking these coordinates for colors..." << endl;
}

// Function to check colors at saved hole coordinates
void checkSavedHoleColors(Mat& original) {
    if (!holesCalibrated || savedHoles.empty()) return;

    for (size_t i = 0; i < savedHoles.size(); i++) {
        int colorResult = detectColour(original, savedHoles[i].center.x, savedHoles[i].center.y);
        savedHoles[i].colour = colorResult;

        // Draw the saved hole with color coding
        Scalar color;
        string colorText;
        switch (colorResult) {
            case 1: color = Scalar(0, 0, 255); colorText = "R"; break; // Red
            case 2: color = Scalar(255, 0, 0); colorText = "B"; break; // Blue
            case 3: color = Scalar(0, 255, 0); colorText = "G"; break; // Green
            case 4: color = Scalar(0, 255, 255); colorText = "Y"; break; // Yellow
            default: color = Scalar(128, 128, 128); colorText = "N"; break; // None
        }

        // Draw filled circle for the hole
        circle(original, savedHoles[i].center, 15, color, -1);
        // Draw outline
        circle(original, savedHoles[i].center, 15, Scalar(255, 255, 255), 2);
        
        // Label with hole number and color
        string label = to_string(i + 1) + ":" + colorText;
        putText(original, label, Point(savedHoles[i].center.x - 10, savedHoles[i].center.y + 5),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    }

    // Print color status to console occasionally
    static int frameCount = 0;
    if (frameCount++ % 30 == 0) { // Print every 30 frames
        cout << "Current colors: ";
        for (size_t i = 0; i < savedHoles.size(); i++) {
            string colorText;
            switch (savedHoles[i].colour) {
                case 1: colorText = "R"; break;
                case 2: colorText = "B"; break;
                case 3: colorText = "G"; break;
                case 4: colorText = "Y"; break;
                default: colorText = "-"; break;
            }
            cout << (i + 1) << ":" << colorText << " ";
        }
        cout << endl;
    }
}

int main(int argc, char* argv[])
{
    // ========= Variables for camera function ==========
    int n = 0;
    char filename[200];
    Mat frame;

    // ========= Camera Setup ===========================
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "cannot open camera";
        return -1;
    }

    // ========= Serial Comms ===========================
    struct sp_port* port;
    int err;
    char cmd = 0;

    // Set up port, check port usage
    if (argc < 2)
    {
        fprintf(stderr, " Port use\n");
        exit(1);
    }

    // Get port name
    err = sp_get_port_by_name("COM3", &port);
    if (err == SP_OK)
        err = sp_open(port, SP_MODE_WRITE);
    if (err != SP_OK)
    {
        fprintf(stderr, " Can't open port %s\n", argv[1]);
        exit(2);
    }
    sp_set_baudrate(port, BAUD);
    sp_set_bits(port, 8);

    // ========== COLOUR DETECTION ===================
    namedWindow("Control", WINDOW_AUTOSIZE);

    // Create trackbars for dark object detection
    int iLowH = 0;
    int iHighH = 179;
    int iLowS = 0;
    int iHighS = 255;
    int iLowV = 0;
    int iHighV = 100;

    createTrackbar("LowH", "Control", &iLowH, 179);
    createTrackbar("HighH", "Control", &iHighH, 179);
    createTrackbar("LowS", "Control", &iLowS, 255);
    createTrackbar("HighS", "Control", &iHighS, 255);
    createTrackbar("LowV", "Control", &iLowV, 255);
    createTrackbar("HighV", "Control", &iHighV, 255);

    // =========== MAIN LOOP ==============================
    printMenu();

    while (true) {
        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal);

        if (!bSuccess) {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        Mat imgHSV;
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);

        Mat imgThresholded;
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

        // Morphological operations to clean up the image
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, kernel);
        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, kernel);

        // Detect the largest dark object (board)
        vector<Point> boardContour = detectBoard(imgThresholded, imgOriginal);

        if (calibrationMode) {
            // CALIBRATION MODE: Detect holes on empty board
            vector<Hole> detectedHoles = calibrateHoles(imgThresholded, imgOriginal, boardContour);
            
            // Display calibration instructions
            string instruction = "CALIBRATION: Place empty board and press 'c'";
            putText(imgOriginal, instruction, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            putText(imgOriginal, "Detected holes: " + to_string(detectedHoles.size()), 
                    Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            
        } else {
            // NORMAL MODE: Check colors at saved coordinates
            checkSavedHoleColors(imgOriginal);
            
            // Display status
            string status = "MONITORING: " + to_string(savedHoles.size()) + " holes";
            putText(imgOriginal, status, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }

        // Display images
        imshow("Thresholded Image", imgThresholded);
        imshow("Original with Detection", imgOriginal);

        char key = (char)waitKey(25);

        // Handle serial communication
        if (key == '1' || key == '2' || key == '3' || key == 'h' || key == 's' || key == 'r') {
            sp_blocking_write(port, &key, 1, 100);
        }

        switch (key) {
        case 'c':  // Calibrate holes
            if (calibrationMode && !boardContour.empty()) {
                vector<Hole> detectedHoles = calibrateHoles(imgThresholded, imgOriginal, boardContour);
                if (!detectedHoles.empty()) {
                    saveCalibratedHoles(detectedHoles);
                } else {
                    cout << "No holes detected! Adjust threshold values." << endl;
                }
            } else if (!calibrationMode) {
                cout << "Recalibrating..." << endl;
                calibrationMode = true;
                holesCalibrated = false;
            }
            break;

        case '1':  // Noughts & Crosses
            cmd = 1;
            cout << "Sent CMD 1 (Noughts & Crosses)" << endl;
            printMenu();
            break;

        case '2':  // Podium
            cmd = 2;
            cout << "Sent CMD 2 (Podium)" << endl;
            printMenu();
            break;

        case '3':  // Stack the Blocks
            cmd = 4;
            cout << "Sent CMD 3 (Stack the Blocks)" << endl;
            printMenu();
            break;

        case 'h':  // Home
            cmd = 8;
            cout << "Sent CMD 4 (Home)" << endl;
            printMenu();
            break;

        case 's':  // Stop
            cmd = 16;
            cout << "STOP (bit4=1)" << endl;
            printMenu();
            break;

        case 'r':  // Resume
            cmd = 32;
            cout << "RESUME (bit5=1)" << endl;
            printMenu();
            break;

        case 'q':
        case 'Q':
        case 27: //escape key
            sp_close(port);
            return 0;
        case ' ': //Save an image
            sprintf_s(filename, "filename%.3d.jpg", n++);
            imwrite(filename, imgOriginal);
            cout << "Saved " << filename << endl;
            break;
        default:
            break;
        }
    }
    sp_close(port);
    return 0;
}
