#include <stdio.h>
#include <stdlib.h>
#include <libserialport.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <map>

#define BAUD 9600

using namespace cv;
using namespace std;

// Structure to store hole information
struct Hole {
    Point2f center;
    double area;
    int colour;
    int row;
    int col;
};

// Global variables
vector<Hole> savedHoles;
bool holesCalibrated = false;
Mat emptyFrame;
bool emptyFrameCaptured = false;
bool continuousColorDetection = false;

// Color mapping
map<int, string> colorNames = {
    {1, "Red"},
    {2, "Blue"}, 
    {3, "Green"},
    {0, "None"}
};

// Position mapping for 3x3 grid (adjust based on your hole layout)
map<int, Point2f> targetPositions = {
    {1, Point2f(100, 100)},  // Row 1, Col 1 (top-left)
    {2, Point2f(200, 100)},  // Row 1, Col 2 (top-middle)
    {3, Point2f(300, 100)},  // Row 1, Col 3 (top-right)
    {4, Point2f(100, 200)},  // Row 2, Col 1 (middle-left)
    {5, Point2f(200, 200)},  // Row 2, Col 2 (center)
    {6, Point2f(300, 200)},  // Row 2, Col 3 (middle-right)
    {7, Point2f(100, 300)},  // Row 3, Col 1 (bottom-left)
    {8, Point2f(200, 300)},  // Row 3, Col 2 (bottom-middle)
    {9, Point2f(300, 300)}   // Row 3, Col 3 (bottom-right)
};

void printMenu() {
    cout << "\n=== Robot Control Menu ===" << endl;
    cout << "1. Capture Empty Matrix" << endl;
    cout << "2. Toggle Continuous Color Detection" << endl;
    cout << "3. Print Matrix Coordinates" << endl;
    cout << "4. Move Block (e.g., 'r1' moves red to position 1)" << endl;
    cout << "5. Print Current Layout" << endl;
    cout << "6. Routine 2" << endl;
    cout << "7. Routine 3" << endl;
    cout << "h. Home" << endl;
    cout << "s. Stop" << endl;
    cout << "r. Resume" << endl;
    cout << "q. Quit" << endl;
    cout << "Enter choice: ";
}

// Function to detect color at specific coordinates and return integer code
int detectColour(Mat& original, int x, int y) {
    if (x < 0 || x >= original.cols || y < 0 || y >= original.rows) {
        return 0;
    }

    Mat hsv;
    cvtColor(original, hsv, COLOR_BGR2HSV);

    Vec3b pixel = hsv.at<Vec3b>(y, x);
    int hue = pixel[0];
    int saturation = pixel[1];
    int value = pixel[2];

    if ((hue >= 140 && hue <= 180) && saturation > 100 && value > 50) {
        return 1; // Red
    }
    else if (hue >= 100 && hue <= 135 && saturation > 100 && value > 50) {
        return 2; // Blue
    }
    else if (hue >= 30 && hue <= 80 && saturation > 100 && value > 50) {
        return 3; // Green
    }

    return 0;
}

// Function to detect the largest dark object (board)
vector<Point> detectBoard(Mat& thresholded, Mat& original) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(thresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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
        drawContours(original, contours, maxAreaIdx, Scalar(0, 255, 255), 3);
        return contours[maxAreaIdx];
    }

    return vector<Point>();
}

// Function to detect holes within the board region
vector<Hole> detectHolesInBoard(Mat& thresholded, Mat& original, const vector<Point>& boardContour) {
    vector<Hole> holes;

    if (boardContour.empty()) return holes;

    Mat boardMask = Mat::zeros(thresholded.size(), CV_8UC1);
    vector<vector<Point>> boardContours = { boardContour };
    fillPoly(boardMask, boardContours, Scalar(255));

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    Mat holesImage;
    bitwise_not(thresholded, holesImage);
    bitwise_and(holesImage, boardMask, holesImage);

    findContours(holesImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 100 || area > 10000) continue;

        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);

        if (pointPolygonTest(boardContour, center, false) >= 0) {
            double perimeter = arcLength(contours[i], true);
            double circularity = 0;
            if (perimeter > 0) {
                circularity = (4 * CV_PI * area) / (perimeter * perimeter);
            }

            if (circularity > 0.5) {
                Hole hole;
                hole.center = center;
                hole.area = area;
                hole.colour = 0;
                
                // Assign grid position based on hole index (you may need to sort these)
                hole.row = (i % 3) + 1;  // Temporary assignment - adjust based on your layout
                hole.col = (i / 3) + 1;
                
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

    Mat imgHSV;
    cvtColor(emptyFrame, imgHSV, COLOR_BGR2HSV);

    Mat imgThresholded;
    inRange(imgHSV, Scalar(0, 0, 0), Scalar(179, 255, 100), imgThresholded);

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, kernel);
    morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, kernel);

    vector<Point> boardContour = detectBoard(imgThresholded, emptyFrame);
    savedHoles = detectHolesInBoard(imgThresholded, emptyFrame, boardContour);

    if (!savedHoles.empty()) {
        holesCalibrated = true;
        cout << "Successfully detected " << savedHoles.size() << " holes!" << endl;

        // Sort holes by position (left to right, top to bottom)
        sort(savedHoles.begin(), savedHoles.end(), [](const Hole& a, const Hole& b) {
            if (a.center.y == b.center.y) return a.center.x < b.center.x;
            return a.center.y < b.center.y;
        });

        // Assign proper grid positions after sorting
        for (size_t i = 0; i < savedHoles.size(); i++) {
            savedHoles[i].row = (i / 3) + 1;  // Assuming 3x3 grid
            savedHoles[i].col = (i % 3) + 1;
        }

        for (size_t i = 0; i < savedHoles.size(); i++) {
            circle(emptyFrame, savedHoles[i].center, 8, Scalar(0, 255, 0), 2);
            string label = "R" + to_string(savedHoles[i].row) + "C" + to_string(savedHoles[i].col);
            putText(emptyFrame, label, Point(savedHoles[i].center.x + 10, savedHoles[i].center.y),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        imshow("Empty Frame with Holes", emptyFrame);
        return true;
    }
    else {
        cout << "No holes detected in empty frame!" << endl;
        return false;
    }
}

// Function to check colors at saved hole positions on live feed
void checkHoleColorsLive(Mat& liveFrame) {
    if (!holesCalibrated || savedHoles.empty()) return;

    for (size_t i = 0; i < savedHoles.size(); i++) {
        int colorResult = detectColour(liveFrame, savedHoles[i].center.x, savedHoles[i].center.y);
        savedHoles[i].colour = colorResult;

        Scalar color;
        string colorText;
        switch (colorResult) {
        case 1: color = Scalar(0, 0, 255); colorText = "R"; break;
        case 2: color = Scalar(255, 0, 0); colorText = "B"; break;
        case 3: color = Scalar(0, 255, 0); colorText = "G"; break;
        default: color = Scalar(128, 128, 128); colorText = "N"; break;
        }

        circle(liveFrame, savedHoles[i].center, 15, color, -1);
        circle(liveFrame, savedHoles[i].center, 15, Scalar(255, 255, 255), 2);
        
        string label = "R" + to_string(savedHoles[i].row) + "C" + to_string(savedHoles[i].col) + ":" + colorText;
        putText(liveFrame, label, Point(savedHoles[i].center.x - 15, savedHoles[i].center.y + 5),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    }

    static int frameCount = 0;
    if (frameCount++ % 60 == 0) {
        cout << "Current colors: ";
        for (size_t i = 0; i < savedHoles.size(); i++) {
            cout << "R" << savedHoles[i].row << "C" << savedHoles[i].col 
                 << ":" << colorNames[savedHoles[i].colour] << " ";
        }
        cout << endl;
    }
}

// Function to print hole coordinates
void printHoleCoordinates() {
    if (!holesCalibrated) {
        cout << "No holes calibrated yet. Use option 1 first." << endl;
        return;
    }

    cout << "=== Hole Coordinates ===" << endl;
    for (size_t i = 0; i < savedHoles.size(); i++) {
        cout << "R" << savedHoles[i].row << "C" << savedHoles[i].col 
             << ": X=" << savedHoles[i].center.x 
             << ", Y=" << savedHoles[i].center.y 
             << ", Area=" << savedHoles[i].area << endl;
    }
}

// Function to print current layout
void printCurrentLayout() {
    if (!holesCalibrated) {
        cout << "No holes calibrated yet. Use option 1 first." << endl;
        return;
    }

    cout << "=== Current Layout ===" << endl;
    for (int row = 1; row <= 3; row++) {
        for (int col = 1; col <= 3; col++) {
            // Find hole at this position
            string color = "Empty";
            for (const auto& hole : savedHoles) {
                if (hole.row == row && hole.col == col) {
                    color = colorNames[hole.colour];
                    break;
                }
            }
            cout << "R" << row << "C" << col << ": " << color << "\t";
        }
        cout << endl;
    }
}

// Function to find block position by color
Hole* findBlockByColor(int color) {
    for (auto& hole : savedHoles) {
        if (hole.colour == color) {
            return &hole;
        }
    }
    return nullptr;
}

// Function to parse move command and execute movement
void executeMoveCommand(struct sp_port* port, const string& command) {
    if (command.length() != 2) {
        cout << "Invalid command format. Use like 'r1' (color + position)" << endl;
        return;
    }

    char colorChar = tolower(command[0]);
    char positionChar = command[1];

    // Map color character to color code
    int targetColor = 0;
    string colorName;
    switch (colorChar) {
        case 'r': targetColor = 1; colorName = "Red"; break;
        case 'b': targetColor = 2; colorName = "Blue"; break;
        case 'g': targetColor = 3; colorName = "Green"; break;
        default: 
            cout << "Invalid color. Use r(ed), b(lue), or g(reen)" << endl;
            return;
    }

    // Map position character to target position (1-9)
    int targetPosition = positionChar - '0';
    if (targetPosition < 1 || targetPosition > 9) {
        cout << "Invalid position. Use 1-9" << endl;
        return;
    }

    cout << "Command: Move " << colorName << " block to position " << targetPosition << endl;

    // Find the block with the specified color
    Hole* sourceHole = findBlockByColor(targetColor);
    if (!sourceHole) {
        cout << colorName << " block not found on the board!" << endl;
        return;
    }

    // Find the target hole (empty position)
    Hole* targetHole = nullptr;
    for (auto& hole : savedHoles) {
        if (hole.row == ((targetPosition-1)/3 + 1) && hole.col == ((targetPosition-1)%3 + 1)) {
            targetHole = &hole;
            break;
        }
    }

    if (!targetHole) {
        cout << "Target position not found!" << endl;
        return;
    }

    cout << "Source: R" << sourceHole->row << "C" << sourceHole->col 
         << " (" << sourceHole->center.x << "," << sourceHole->center.y << ")" << endl;
    cout << "Target: R" << targetHole->row << "C" << targetHole->col 
         << " (" << targetHole->center.x << "," << targetHole->center.y << ")" << endl;

    // Calculate movement parameters (you'll need to map these to your robot's coordinate system)
    float deltaX = targetHole->center.x - sourceHole->center.x;
    float deltaY = targetHole->center.y - sourceHole->center.y;
    
    cout << "Movement required: X=" << deltaX << ", Y=" << deltaY << endl;

    // Generate command based on movement pattern
    char cmd = 0;
    
    // Simple movement logic - you'll need to customize this for your robot
    if (abs(deltaX) > abs(deltaY)) {
        // Horizontal movement dominant
        if (deltaX > 0) {
            cmd = 'R'; // Move right
            cout << "Sending command: Move Right" << endl;
        } else {
            cmd = 'L'; // Move left
            cout << "Sending command: Move Left" << endl;
        }
    } else {
        // Vertical movement dominant
        if (deltaY > 0) {
            cmd = 'D'; // Move down
            cout << "Sending command: Move Down" << endl;
        } else {
            cmd = 'U'; // Move up
            cout << "Sending command: Move Up" << endl;
        }
    }

    // Send command to robot
    sp_blocking_write(port, &cmd, 1, 100);
    cout << "Command sent: " << cmd << endl;

    // Update the board state (simulate movement)
    targetHole->colour = sourceHole->colour;
    sourceHole->colour = 0;
    
    cout << "Movement completed!" << endl;
}

int main(int argc, char* argv[])
{
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return -1;
    }

    struct sp_port* port;
    int err;
    char cmd = 0;

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

    cout << "Robot Control System Started" << endl;
    cout << "Press '1' to capture empty matrix first" << endl;
    printMenu();

    while (true) {
        Mat liveFrame;
        if (cap.read(liveFrame)) {
            if (holesCalibrated) {
                if (continuousColorDetection) {
                    checkHoleColorsLive(liveFrame);
                    putText(liveFrame, "Continuous Color Detection - ON", 
                            Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                } else {
                    for (size_t i = 0; i < savedHoles.size(); i++) {
                        circle(liveFrame, savedHoles[i].center, 5, Scalar(0, 255, 0), 2);
                    }
                    putText(liveFrame, "Holes Calibrated - " + to_string(savedHoles.size()) + " holes", 
                            Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                }
            } else {
                putText(liveFrame, "Press '1' to capture empty matrix and detect holes", 
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }
            
            imshow("Live Feed", liveFrame);
        }

        int key = waitKey(30);

        if (key != -1) {
            switch (key) {
            case '1':
                if (captureEmptyFrame(cap)) {
                    cout << "Hole calibration successful!" << endl;
                    continuousColorDetection = true;
                    cout << "Continuous color detection started automatically" << endl;
                }
                else {
                    cout << "Hole calibration failed. Adjust camera/view and try again." << endl;
                }
                printMenu();
                break;

            case '2':
                if (holesCalibrated) {
                    continuousColorDetection = !continuousColorDetection;
                    cout << "Continuous color detection: " 
                         << (continuousColorDetection ? "ON" : "OFF") << endl;
                } else {
                    cout << "Please calibrate holes first (option 1)" << endl;
                }
                printMenu();
                break;

            case '3':
                printHoleCoordinates();
                printMenu();
                break;

            case '4': {
                cout << "Enter move command (e.g., 'r1' to move red to position 1): ";
                string moveCommand;
                cin >> moveCommand;
                executeMoveCommand(port, moveCommand);
                printMenu();
                break;
            }

            case '5':
                printCurrentLayout();
                printMenu();
                break;

            case '6':
                cmd = '2';
                cout << "Sent CMD 2 (Routine 2)" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case '7':
                cmd = '3';
                cout << "Sent CMD 3 (Routine 3)" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case 'h':
                cmd = 'h';
                cout << "Sent CMD 4 (Home)" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case 's':
                cmd = 's';
                cout << "STOP" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case 'r':
                cmd = 'r';
                cout << "RESUME" << endl;
                sp_blocking_write(port, &cmd, 1, 100);
                printMenu();
                break;

            case 'q':
            case 'Q':
            case 27:
                cout << "Quitting..." << endl;
                sp_close(port);
                return 0;

            default:
                break;
            }
        }
    }

    sp_close(port);
    return 0;
}
