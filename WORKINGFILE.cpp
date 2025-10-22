#include <stdio.h>
#include <stdlib.h>
#include <libserialport.h>

#define BAUD 9600

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;

// Structure to store hole information
struct Hole {
    Point2f center;
    double area;
    int colour;
};

// Global variables
vector<Hole> savedHoles;
bool holesCalibrated = false;
Mat emptyFrame;
bool emptyFrameCaptured = false;

void printMenu() {
    cout << "\n=== Robot Control Menu ===" << endl;
    cout << "1. Capture Empty Frame & Detect Holes" << endl;
    cout << "2. Check Colors at Hole Positions" << endl;
    cout << "3. Print Hole Coordinates" << endl;
    cout << "4. Noughts & Crosses" << endl;
    cout << "5. Podium" << endl;
    cout << "6. Stack the Blocks" << endl;
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

// Function to detect holes within the board region
vector<Hole> detectHolesInBoard(Mat& thresholded, Mat& original, const vector<Point>& boardContour) {
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
                hole.colour = 0; // Default to no color
                holes.push_back(hole);
            }
        }
    }

    return holes;
}

// Function to capture and process empty frame
bool captureEmptyFrame(VideoCapture& cap) {
    Mat frame;
    if (!cap.read(frame)) {
        cout << "Cannot read frame from camera" << endl;
        return false;
    }
    
    emptyFrame = frame.clone();
    emptyFrameCaptured = true;
    
    cout << "Empty frame captured! Processing holes..." << endl;
    
    // Process the empty frame to detect holes
    Mat imgHSV;
    cvtColor(emptyFrame, imgHSV, COLOR_BGR2HSV);

    // Threshold values for dark board detection
    Mat imgThresholded;
    inRange(imgHSV, Scalar(0, 0, 0), Scalar(179, 255, 100), imgThresholded);

    // Morphological operations to clean up the image
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, kernel);
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, kernel);

    // Detect board and holes
    vector<Point> boardContour = detectBoard(imgThresholded, emptyFrame);
    savedHoles = detectHolesInBoard(imgThresholded, emptyFrame, boardContour);
    
    if (!savedHoles.empty()) {
        holesCalibrated = true;
        cout << "Successfully detected " << savedHoles.size() << " holes!" << endl;
        
        // Draw holes on the empty frame for visualization
        for (size_t i = 0; i < savedHoles.size(); i++) {
            circle(emptyFrame, savedHoles[i].center, 8, Scalar(0, 255, 0), 2);
            string label = to_string(i + 1) + " (" + to_string((int)savedHoles[i].center.x) + 
                          "," + to_string((int)savedHoles[i].center.y) + ")";
            putText(emptyFrame, label, Point(savedHoles[i].center.x + 10, savedHoles[i].center.y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }
        
        imshow("Empty Frame with Holes", emptyFrame);
        return true;
    } else {
        cout << "No holes detected in empty frame!" << endl;
        return false;
    }
}

// Function to check colors at saved hole positions
void checkHoleColors(VideoCapture& cap) {
    if (!holesCalibrated) {
        cout << "Please capture empty frame first (option 1)" << endl;
        return;
    }
    
    Mat currentFrame;
    if (!cap.read(currentFrame)) {
        cout << "Cannot read frame from camera" << endl;
        return;
    }
    
    cout << "Checking colors at hole positions..." << endl;
    
    // Check color at each saved hole position
    for (size_t i = 0; i < savedHoles.size(); i++) {
        int colorResult = detectColour(currentFrame, savedHoles[i].center.x, savedHoles[i].center.y);
        savedHoles[i].colour = colorResult;
        
        string colorName;
        switch (colorResult) {
            case 1: colorName = "Red"; break;
            case 2: colorName = "Blue"; break;
            case 3: colorName = "Green"; break;
            case 4: colorName = "Yellow"; break;
            default: colorName = "None"; break;
        }
        
        cout << "Hole " << (i + 1) << " at (" << savedHoles[i].center.x << ", " 
             << savedHoles[i].center.y << "): " << colorName << endl;
    }
    
    // Display current frame with color indicators
    Mat displayFrame = currentFrame.clone();
    for (size_t i = 0; i < savedHoles.size(); i++) {
        Scalar color;
        switch (savedHoles[i].colour) {
            case 1: color = Scalar(0, 0, 255); break; // Red
            case 2: color = Scalar(255, 0, 0); break; // Blue
            case 3: color = Scalar(0, 255, 0); break; // Green
            case 4: color = Scalar(0, 255, 255); break; // Yellow
            default: color = Scalar(128, 128, 128); break; // None
        }
        
        circle(displayFrame, savedHoles[i].center, 15, color, -1);
        circle(displayFrame, savedHoles[i].center, 15, Scalar(255, 255, 255), 2);
        
        string label = to_string(i + 1);
        putText(displayFrame, label, Point(savedHoles[i].center.x - 5, savedHoles[i].center.y + 5),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    }
    
    putText(displayFrame, "Current Colors", Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    
    imshow("Current Frame with Colors", displayFrame);
}

// Function to print hole coordinates
void printHoleCoordinates() {
    if (!holesCalibrated) {
        cout << "No holes calibrated yet. Use option 1 first." << endl;
        return;
    }
    
    cout << "=== Hole Coordinates ===" << endl;
    for (size_t i = 0; i < savedHoles.size(); i++) {
        cout << "Hole " << (i + 1) << ": X = " << savedHoles[i].center.x 
             << ", Y = " << savedHoles[i].center.y 
             << ", Area = " << savedHoles[i].area << endl;
    }
}

int main(int argc, char* argv[])
{
    // ========= Camera Setup ===========================
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return -1;
    }

    // ========= Serial Comms ===========================
    struct sp_port* port;
    int err;
    char cmd = 0;

    // Set up port
    if (argc < 2) {
        fprintf(stderr, "Port use\n");
        exit(1);
    }

    err = sp_get_port_by_name("COM3", &port);
    if (err == SP_OK)
        err = sp_open(port, SP_MODE_WRITE);
    if (err != SP_OK) {
        fprintf(stderr, "Can't open port %s\n", argv[1]);
        exit(2);
    }
    sp_set_baudrate(port, BAUD);
    sp_set_bits(port, 8);

    // ========== MAIN LOOP ==============================
    cout << "Robot Control System Started" << endl;
    cout << "Make sure the board is empty and visible in the camera" << endl;
    printMenu();

    while (true) {
        // Display live feed
        Mat liveFrame;
        if (cap.read(liveFrame)) {
            if (holesCalibrated) {
                // Show saved hole positions on live feed
                for (size_t i = 0; i < savedHoles.size(); i++) {
                    circle(liveFrame, savedHoles[i].center, 5, Scalar(0, 255, 0), 2);
                }
                putText(liveFrame, "Holes Calibrated - " + to_string(savedHoles.size()) + " holes", 
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
            } else {
                putText(liveFrame, "Press '1' to capture empty frame and detect holes", 
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }
            imshow("Live Feed", liveFrame);
        }

        // Wait for key press with longer delay for better responsiveness
        int key = waitKey(100);
        
        if (key != -1) {
            switch (key) {
            case '1':  // Capture empty frame and detect holes
                if (captureEmptyFrame(cap)) {
                    cout << "Hole calibration successful!" << endl;
                } else {
                    cout << "Hole calibration failed. Adjust camera/view and try again." << endl;
                }
                printMenu();
                break;

            case '2':  // Check colors at hole positions
                checkHoleColors(cap);
                printMenu();
                break;

            case '3':  // Print hole coordinates
                printHoleCoordinates();
                printMenu();
                break;

            case '4':  // Noughts & Crosses
                cmd = '1';
                cout << "Sent CMD 1 (Noughts & Crosses)" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case '5':  // Podium
                cmd = '2';
                cout << "Sent CMD 2 (Podium)" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case '6':  // Stack the Blocks
                cmd = '3';
                cout << "Sent CMD 3 (Stack the Blocks)" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case 'h':  // Home
                cmd = 'h';
                cout << "Sent CMD 4 (Home)" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case 's':  // Stop
                cmd = 's';
                cout << "STOP" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case 'r':  // Resume
                cmd = 'r';
                cout << "RESUME" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case 'q':
            case 'Q':
            case 27: // Escape key
                cout << "Quitting..." << endl;
                sp_close(port);
                return 0;

            default:
                // Ignore other keys
                break;
            }
        }
    }
    
    sp_close(port);
    return 0;
}
