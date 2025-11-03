// ========================================
//          ROBOT COLOR SORTING SYSTEM (REFERENCE VERSION)
// ========================================
// This reference version of the code includes additional comments explaining how
// each section works. The program uses OpenCV and libserialport to detect colored
// blocks on a 3x3 board and control a robot arm to move the blocks accordingly.
//
// The system flow is as follows:
// 1. Camera captures live video feed of the board.
// 2. User calibrates the board (detects grid spaces).
// 3. Colors are detected automatically or on demand.
// 4. GUI allows selecting a color and destination row.
// 5. Serial commands are sent to the robot to execute the move.
//
// Dependencies:
//   - OpenCV (for image capture and processing)
//   - libserialport (for serial communication with the robot)
// ========================================

#include <stdio.h>
#include <stdlib.h>
#include <libserialport.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <chrono>

#define BAUD 9600

using namespace cv;
using namespace std;
using namespace std::chrono;

// ----------------------------------------
// STRUCTURE DEFINITIONS
// ----------------------------------------
// Each 'Space' represents one of the 9 grid positions on the board.
// The program identifies and tracks these using camera input.
struct Space {
    Point2f center;  // Center pixel location of the space
    double area;     // Area of the detected space (used for filtering)
    int colour;      // Detected color code (0=None, 1=Red, 2=Blue, 3=Green)
    int row;         // Logical grid row (1 to 3)
    int col;         // Logical grid column (1 to 3)
    int position_id; // Unique ID assigned based on row & column
};

// ----------------------------------------
// GLOBAL VARIABLES AND STATE FLAGS
// ----------------------------------------
vector<Space> savedSpaces;          // List of calibrated board positions
bool spacesCalibrated = false;      // True when calibration successful
Mat emptyFrame;                     // Stores empty board snapshot
bool emptyFrameCaptured = false;
bool continuousColorDetection = false; // Enables live color tracking
VideoCapture global_cap(0);         // Opens the webcam feed

// GUI interaction state
int selectedColor = 0; // 1=Red, 2=Blue, 3=Green
int selectedRow = 0;   // 1–3 represent target rows

// ----------------------------------------
// GUI ELEMENT LAYOUT (button rectangles)
// ----------------------------------------
Rect calibrateBtn = Rect(50, 100, 300, 50);
Rect colorRedBtn = Rect(50, 200, 80, 30);
Rect colorBlueBtn = Rect(140, 200, 80, 30);
Rect colorGreenBtn = Rect(230, 200, 80, 30);
Rect row1Btn = Rect(50, 250, 80, 30);
Rect row2Btn = Rect(140, 250, 80, 30);
Rect row3Btn = Rect(230, 250, 80, 30);
Rect executeBtn = Rect(50, 300, 300, 50);
Rect resetBtn = Rect(50, 370, 140, 40);
Rect homeBtn = Rect(200, 370, 140, 40);
Rect colorDetectionBtn = Rect(50, 450, 300, 30);

// Map color IDs to names for display
map<int, string> colorNames = {
    {1, "Red"}, {2, "Blue"}, {3, "Green"}, {0, "None"}
};

// Mapping grid positions (row, col) → unique ID
map<pair<int, int>, int> positionMap = {
    {{1,1},1}, {{1,2},2}, {{1,3},3},
    {{2,1},4}, {{2,2},5}, {{2,3},6},
    {{3,1},7}, {{3,2},8}, {{3,3},9}
};

// Predefined reset commands (encoded robot instructions)
map<pair<int,int>, unsigned char> resetCmdMap = {
    {{1,1},129}, {{1,2},130}, {{1,3},131},
    {{2,1},133}, {{2,2},134}, {{2,3},135},
    {{3,1},137}, {{3,2},138}, {{3,3},139}
};

// ----------------------------------------
// FUNCTION DECLARATIONS
// ----------------------------------------
bool captureEmptyFrame(VideoCapture& cap);
void checkSpaceColorsLive(Mat& liveFrame);
int detectColour(Mat& original, int x, int y);
vector<Point> detectBoard(Mat& thresholded, Mat& original);
vector<Space> detectSpacesInBoard(Mat& thresholded, Mat& original, const vector<Point>& boardContour);
void onMouse(int event, int x, int y, int flags, void* userdata);
void createControlPanel();
void executeMoveFromGUI(struct sp_port* port);
void executeReset(struct sp_port* port);
int getPositionId(int row, int col);
Space* findBlockByColor(int colorCode);
vector<Space*> findBlocksInColumn3();
vector<Space*> findEmptyPositionsInColumn1();

// ----------------------------------------
// COLOR DETECTION FUNCTION
// ----------------------------------------
// Converts pixel color to HSV and classifies into Red, Blue, or Green.
int detectColour(Mat& original, int x, int y) {
    if (x < 0 || x >= original.cols || y < 0 || y >= original.rows) return 0;
    Mat hsv; cvtColor(original, hsv, COLOR_BGR2HSV);
    Vec3b pixel = hsv.at<Vec3b>(y, x);
    int hue = pixel[0], sat = pixel[1], val = pixel[2];

    // HSV thresholds tuned for each color range
    if ((hue >= 140 && hue <= 180) && sat > 100 && val > 50) return 1; // Red
    else if (hue >= 100 && hue <= 135 && sat > 100 && val > 50) return 2; // Blue
    else if (hue >= 30 && hue <= 80 && sat > 100 && val > 50) return 3; // Green
    return 0;
}

// ----------------------------------------
// CAPTURE AND CALIBRATION LOGIC
// ----------------------------------------
// Takes an image of the empty board, detects its boundary, and identifies grid spaces.
bool captureEmptyFrame(VideoCapture& cap) {
    Mat frame;
    if (!cap.read(frame)) {
        cout << "Cannot read frame from camera" << endl;
        return false;
    }

    emptyFrame = frame.clone(); // Store a reference frame for later color comparison
    Mat imgHSV, imgThresholded;
    cvtColor(emptyFrame, imgHSV, COLOR_BGR2HSV);
    inRange(imgHSV, Scalar(0, 0, 0), Scalar(179, 255, 100), imgThresholded);

    // Morphological filtering to clean noise
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, kernel);
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, kernel);

    // Detect outer board and internal circles (spaces)
    vector<Point> boardContour = detectBoard(imgThresholded, emptyFrame);
    savedSpaces = detectSpacesInBoard(imgThresholded, emptyFrame, boardContour);

    if (savedSpaces.empty()) {
        cout << "No spaces detected in empty frame!" << endl;
        return false;
    }

    // Sort detected spaces by Y then X to match grid layout
    sort(savedSpaces.begin(), savedSpaces.end(),
         [](const Space& a, const Space& b) {
             return (a.center.y == b.center.y) ? a.center.x < b.center.x : a.center.y < b.center.y;
         });

    // Assign logical grid coordinates and position IDs
    for (size_t i = 0; i < savedSpaces.size(); i++) {
        savedSpaces[i].row = (i / 3) + 1;
        savedSpaces[i].col = 3 - (i % 3);
        savedSpaces[i].position_id = positionMap[{savedSpaces[i].row, savedSpaces[i].col}];
    }

    spacesCalibrated = true;
    cout << "Successfully detected " << savedSpaces.size() << " spaces!" << endl;
    return true;
}

// ----------------------------------------
// LIVE COLOR DETECTION FUNCTION
// ----------------------------------------
// Updates each calibrated space's color reading in real time
void checkSpaceColorsLive(Mat& liveFrame) {
    if (!spacesCalibrated || savedSpaces.empty()) return;

    for (size_t i = 0; i < savedSpaces.size(); i++) {
        int colorResult = detectColour(liveFrame, savedSpaces[i].center.x, savedSpaces[i].center.y);
        savedSpaces[i].colour = colorResult;

        Scalar color;
        switch (colorResult) {
            case 1: color = Scalar(0, 0, 255); break;   // Red circle
            case 2: color = Scalar(255, 0, 0); break;   // Blue circle
            case 3: color = Scalar(0, 255, 0); break;   // Green circle
            default: color = Scalar(128, 128, 128); break; // Grey for none
        }

        // Draw circle on live feed to visualize detection
        circle(liveFrame, savedSpaces[i].center, 15, color, -1);
        circle(liveFrame, savedSpaces[i].center, 15, Scalar(255, 255, 255), 2);
    }
}

// ----------------------------------------
// MAIN FUNCTION
// ----------------------------------------
// Initializes camera and serial, creates GUI, and loops until exit.
int main(int argc, char* argv[]) {
    if (!global_cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return -1;
    }

    struct sp_port* port = nullptr;
    int err;

    // Initialize serial port for robot communication
    if (argc >= 2) {
        err = sp_get_port_by_name("COM3", &port);
        if (err == SP_OK && sp_open(port, SP_MODE_WRITE) == SP_OK) {
            sp_set_baudrate(port, BAUD);
            sp_set_bits(port, 8);
            cout << "Serial port initialized successfully" << endl;
        } else {
            cout << "Warning: Could not open serial port" << endl;
            port = nullptr;
        }
    } else {
        cout << "No serial port specified. Running without hardware control." << endl;
    }

    namedWindow("Control Panel", WINDOW_NORMAL);
    namedWindow("Live Feed", WINDOW_NORMAL);

    // ------------------------------
    // MAIN LOOP: Live Video + GUI
    // ------------------------------
    while (true) {
        Mat liveFrame;

        // Capture current frame from webcam
        if (global_cap.read(liveFrame)) {
            if (spacesCalibrated) {
                if (continuousColorDetection)
                    checkSpaceColorsLive(liveFrame); // Continuously analyze colors
                else {
                    // Display markers for calibrated points
                    for (auto& s : savedSpaces)
                        circle(liveFrame, s.center, 5, Scalar(0,255,0), 2);
                    putText(liveFrame, "Matrix Calibrated", Point(10, 30),
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);
                }

                // Overlay user selections (color + row)
                string colorName = (selectedColor > 0) ? colorNames[selectedColor] : "None";
                string selectionText = "Selection: " + colorName + " -> Row " + to_string(selectedRow);
                putText(liveFrame, selectionText, Point(10, 30),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 2);
            } else {
                // Prompt user to calibrate if not done yet
                putText(liveFrame, "Calibrate Matrix in Control Panel", Point(10, 30),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }

            // Display updated live camera feed
            imshow("Live Feed", liveFrame);
        }

        // Refresh GUI window (buttons, text, etc.)
        createControlPanel();

        // Wait for 30ms and handle quit keys
        int key = waitKey(30);
        if (key == 'q' || key == 'Q' || key == 27) {
            cout << "Exiting program..." << endl;
            if (port) sp_close(port);
            break;
        }
    }

    if (port) sp_close(port);
    return 0;
}
