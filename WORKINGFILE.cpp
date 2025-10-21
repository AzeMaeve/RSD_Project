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
    vector<Point> contour;
};

void printMenu() {
    cout << "\n=== Robot Control Menu ===\n";
    cout << "1. Noughts & Crosses\n";
    cout << "2. Podium\n";
    cout << "3. Stack the Blocks\n";
    cout << "h. Home\n";
    cout << "s. Stop\n";
    cout << "r. Resume\n";
    cout << "0. Quit\n";
    cout << "Enter choice: \n";
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

    if (maxAreaIdx >= 0 && maxArea > 10000) { // Minimum area for board
        // Draw the board contour
        drawContours(original, contours, maxAreaIdx, Scalar(0, 255, 255), 3);

        // Calculate and display board center
        Moments m = moments(contours[maxAreaIdx]);
        Point2f boardCenter(m.m10 / m.m00, m.m01 / m.m00);
        circle(original, boardCenter, 8, Scalar(255, 255, 0), -1);
        putText(original, "Board Center", Point(boardCenter.x + 10, boardCenter.y),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);

        return contours[maxAreaIdx];
    }

    return vector<Point>(); // Return empty if no board found
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
            // Circularity of 1.0 = perfect circle, ~0.7-0.8 for squares
            if (circularity > 0.5) {
                Hole hole;
                hole.center = center;
                hole.area = area;
                hole.contour = contours[i];
                holes.push_back(hole);

                // Draw the hole
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
    char cmd;

    // Set up port, check port usage
    if (argc < 2)
    {
        fprintf(stderr, " Port use\n"); // Return error
        exit(1);
    }

    // Get port name
    err = sp_get_port_by_name("COM3", &port);
    if (err == SP_OK)
        err = sp_open(port, SP_MODE_WRITE); // Open port
    if (err != SP_OK)
    {
        fprintf(stderr, " Can't open port %s\n", argv[1]); // Return error
        exit(2);
    }
    sp_set_baudrate(port, BAUD); // Set BAUD rate
    sp_set_bits(port, 8); // Set num of bits

    // ========== COLOUR DETECTION ===================
    namedWindow("Control", WINDOW_AUTOSIZE);

    // Create trackbars for dark object detection
    int iLowH = 0;
    int iHighH = 179;
    int iLowS = 0;
    int iHighS = 255;
    int iLowV = 0;    // Lower value for dark objects
    int iHighV = 100; // Upper value for dark objects (adjust based on lighting)

    createTrackbar("LowH", "Control", &iLowH, 179);
    createTrackbar("HighH", "Control", &iHighH, 179);
    createTrackbar("LowS", "Control", &iLowS, 255);
    createTrackbar("HighS", "Control", &iHighS, 255);
    createTrackbar("LowV", "Control", &iLowV, 255);
    createTrackbar("HighV", "Control", &iHighV, 255);

    // =========== MAIN LOOP ==============================
    printMenu(); // Print Menu to console

    while (true) {
        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal); // read a new frame from video

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

        // Detect holes within the board
        vector<Hole> holes = detectHolesInBoard(imgThresholded, imgOriginal, boardContour);

        // Print results to console
        /*if (!boardContour.empty()) {
            Moments m = moments(boardContour);
            Point2f boardCenter(m.m10 / m.m00, m.m01 / m.m00);
            cout << "Board detected - Center: (" << boardCenter.x << ", " << boardCenter.y << ")" << endl;

            if (!holes.empty()) {
                cout << "Detected " << holes.size() << " holes:" << endl;
                for (size_t i = 0; i < holes.size(); i++) {
                    cout << "  Hole " << (i + 1) << ": X=" << holes[i].center.x
                        << ", Y=" << holes[i].center.y
                        << ", Area=" << holes[i].area << endl;
                }
            }
            else {
                cout << "No holes detected in board" << endl;
            }
            cout << "---" << endl;
        }
        else {
            cout << "No board detected - adjust threshold values" << endl;
        }*/

        // Display images
        imshow("Thresholded Image", imgThresholded);
        imshow("Original with Detection", imgOriginal);

        char key = (char)waitKey(25);

        // Handle serial communication
        sp_blocking_write(port, &cmd, 1, 100);

        switch (key) {
        case '1':  // Noughts & Crosses
            cmd = 1;
            cout << "Sent CMD 1 (Noughts & Crosses)\n";
            cout << "cmd = " << (int)cmd << "\n";
            printMenu();
            break;

        case '2':  // Podium
            cmd = 2;
            cout << "Sent CMD 2 (Podium)\n";
            cout << "cmd = " << (int)cmd << "\n";
            printMenu();
            break;

        case '3':  // Stack the Blocks
            cmd = 4;
            cout << "Sent CMD 3 (Stack the Blocks)\n";
            cout << "cmd = " << (int)cmd << "\n";
            printMenu();
            break;

        case 'h':  // Home
            cmd = 8;
            cout << "Sent CMD 4 (Home)\n";
            cout << "cmd = " << (int)cmd << "\n";
            printMenu();
            break;

        case 's':  // Stop
            cmd = 16;
            cout << "STOP (bit4=1)\n";
            cout << "cmd = " << (int)cmd << "\n";
            printMenu();
            break;

        case 'r':  // Resume
            cmd = 32;
            cout << "RESUME (bit5=1)\n";
            cout << "cmd = " << (int)cmd << "\n";
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
