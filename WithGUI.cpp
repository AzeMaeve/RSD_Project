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

// Structure to store hole information
struct Hole {
    Point2f center;
    double area;
    int colour;
    int row;
    int col;
    int position_id; // 1-9 for 3x3 grid
};

// Global variables
vector<Hole> savedHoles;
bool holesCalibrated = false;
Mat emptyFrame;
bool emptyFrameCaptured = false;
bool continuousColorDetection = false;
VideoCapture global_cap(0); // Global camera object

// GUI state variables
int selectedColor = 0; // 0=None, 1=Red, 2=Blue, 3=Green
int selectedRow = 0;   // 0=None, 1-3=Row number

// Button regions for mouse clicks
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

// Color mapping
map<int, string> colorNames = {
    {1, "Red"},
    {2, "Blue"},
    {3, "Green"},
    {0, "None"}
};

// Position mapping: row and column to position_id
map<pair<int, int>, int> positionMap = {
    {{1, 1}, 1}, {{1, 2}, 2}, {{1, 3}, 3},
    {{2, 1}, 4}, {{2, 2}, 5}, {{2, 3}, 6},
    {{3, 1}, 7}, {{3, 2}, 8}, {{3, 3}, 9}
};

// Reset command mapping: (pick_row, place_row) -> cmd
map<pair<int, int>, unsigned char> resetCmdMap = {
    {{1, 1}, 129}, // C3R1 -> C1R1
    {{1, 2}, 130}, // C3R1 -> C1R2
    {{1, 3}, 131}, // C3R1 -> C1R3
    {{2, 1}, 133}, // C3R2 -> C1R1
    {{2, 2}, 134}, // C3R2 -> C1R2
    {{2, 3}, 135}, // C3R2 -> C1R3
    {{3, 1}, 137}, // C3R3 -> C1R1
    {{3, 2}, 138}, // C3R3 -> C1R2
    {{3, 3}, 139}  // C3R3 -> C1R3
};

// Forward declarations
bool captureEmptyFrame(VideoCapture& cap);
void executeMoveFromGUI(struct sp_port* port);
void executeReset(struct sp_port* port);
int getPositionId(int row, int col);
Hole* findBlockByColor(int colorCode);
vector<Hole*> findBlocksInColumn3();
vector<Hole*> findEmptyPositionsInColumn1();
void checkHoleColorsLive(Mat& liveFrame);
int detectColour(Mat& original, int x, int y);
vector<Point> detectBoard(Mat& thresholded, Mat& original);
vector<Hole> detectHolesInBoard(Mat& thresholded, Mat& original, const vector<Point>& boardContour);

// Mouse callback for control panel
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Point pt(x, y);
        struct sp_port* port = (struct sp_port*)userdata;
        
        // Check which button was clicked
        if (calibrateBtn.contains(pt)) {
            cout << "Calibrating matrix..." << endl;
            if (captureEmptyFrame(global_cap)) {
                cout << "Calibration successful!" << endl;
                continuousColorDetection = true;
                cout << "Continuous color detection started automatically" << endl;
            }
            else {
                cout << "Calibration failed. Adjust camera/view and try again." << endl;
            }
        }
        else if (colorRedBtn.contains(pt)) {
            selectedColor = 1;
            cout << "Selected: Red" << endl;
        }
        else if (colorBlueBtn.contains(pt)) {
            selectedColor = 2;
            cout << "Selected: Blue" << endl;
        }
        else if (colorGreenBtn.contains(pt)) {
            selectedColor = 3;
            cout << "Selected: Green" << endl;
        }
        else if (row1Btn.contains(pt)) {
            selectedRow = 1;
            cout << "Selected: Row 1" << endl;
        }
        else if (row2Btn.contains(pt)) {
            selectedRow = 2;
            cout << "Selected: Row 2" << endl;
        }
        else if (row3Btn.contains(pt)) {
            selectedRow = 3;
            cout << "Selected: Row 3" << endl;
        }
        else if (executeBtn.contains(pt)) {
            cout << "Executing move..." << endl;
            executeMoveFromGUI(port);
        }
        else if (resetBtn.contains(pt)) {
            cout << "Executing reset..." << endl;
            executeReset(port);
        }
        else if (homeBtn.contains(pt)) {
            cout << "Going home..." << endl;
            if (port) {
                unsigned char cmd = 64;
                sp_blocking_write(port, &cmd, 1, 100);
                this_thread::sleep_for(milliseconds(2000));
                cmd = 0;
                sp_blocking_write(port, &cmd, 1, 100);
                cout << "Home position set!" << endl;
            }
        }
        else if (colorDetectionBtn.contains(pt)) {
            if (holesCalibrated) {
                continuousColorDetection = !continuousColorDetection;
                cout << "Continuous color detection: " << (continuousColorDetection ? "ON" : "OFF") << endl;
            }
            else {
                cout << "Please calibrate matrix first!" << endl;
            }
        }
    }
}

// Function to create control panel GUI
void createControlPanel() {
    Mat controlPanel = Mat::zeros(600, 400, CV_8UC3);
    
    // Set background color
    controlPanel.setTo(Scalar(60, 60, 60));
    
    // Title
    putText(controlPanel, "Robot Control Panel", Point(20, 30), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    
    // Calibration section
    rectangle(controlPanel, calibrateBtn, Scalar(100, 100, 100), -1);
    rectangle(controlPanel, calibrateBtn, Scalar(200, 200, 200), 2);
    putText(controlPanel, "Calibrate Matrix", Point(60, 130), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    
    // Status
    string statusText = holesCalibrated ? "Status: CALIBRATED" : "Status: NOT CALIBRATED";
    Scalar statusColor = holesCalibrated ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
    putText(controlPanel, statusText, Point(20, 80), 
            FONT_HERSHEY_SIMPLEX, 0.5, statusColor, 1);
    
    // Color selection
    putText(controlPanel, "Select Color:", Point(20, 180), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    
    // Color buttons
    rectangle(controlPanel, colorRedBtn, selectedColor == 1 ? Scalar(0, 0, 255) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, colorRedBtn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Red", Point(65, 220), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    rectangle(controlPanel, colorBlueBtn, selectedColor == 2 ? Scalar(255, 0, 0) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, colorBlueBtn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Blue", Point(155, 220), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    rectangle(controlPanel, colorGreenBtn, selectedColor == 3 ? Scalar(0, 255, 0) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, colorGreenBtn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Green", Point(245, 220), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    // Row selection
    putText(controlPanel, "Select Target Row:", Point(20, 240), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    
    rectangle(controlPanel, row1Btn, selectedRow == 1 ? Scalar(100, 100, 200) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, row1Btn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Row 1", Point(65, 270), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    rectangle(controlPanel, row2Btn, selectedRow == 2 ? Scalar(100, 100, 200) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, row2Btn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Row 2", Point(155, 270), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    rectangle(controlPanel, row3Btn, selectedRow == 3 ? Scalar(100, 100, 200) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, row3Btn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Row 3", Point(245, 270), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    // Execute button
    bool canExecute = (selectedColor > 0 && selectedRow > 0 && holesCalibrated);
    rectangle(controlPanel, executeBtn, canExecute ? Scalar(0, 100, 0) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, executeBtn, Scalar(200, 200, 200), 2);
    putText(controlPanel, "EXECUTE MOVE", Point(80, 330), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    
    // Current selection display
    string selectionText = "Current: " + colorNames[selectedColor] + " -> Row " + to_string(selectedRow);
    putText(controlPanel, selectionText, Point(20, 360), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
    
    // Special commands
    rectangle(controlPanel, resetBtn, holesCalibrated ? Scalar(0, 0, 100) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, resetBtn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "RESET", Point(70, 395), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    rectangle(controlPanel, homeBtn, Scalar(100, 0, 0), -1);
    rectangle(controlPanel, homeBtn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "HOME", Point(230, 395), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    // Color detection toggle
    rectangle(controlPanel, colorDetectionBtn, continuousColorDetection ? Scalar(0, 100, 0) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, colorDetectionBtn, Scalar(200, 200, 200), 1);
    string detectionText = continuousColorDetection ? "Color Detection: ON" : "Color Detection: OFF";
    putText(controlPanel, detectionText, Point(60, 470), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    // Instructions
    putText(controlPanel, "Instructions:", Point(20, 520), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(controlPanel, "Click buttons to activate functions", Point(20, 540), 
            FONT_HERSHEY_SIMPLEX, 0.3, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Grayed out buttons = disabled", Point(20, 560), 
            FONT_HERSHEY_SIMPLEX, 0.3, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Press 'q' in any window to quit", Point(20, 580), 
            FONT_HERSHEY_SIMPLEX, 0.3, Scalar(200, 200, 200), 1);
    
    imshow("Control Panel", controlPanel);
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

        // Assign grid positions (1-9 for 3x3 grid)
        for (size_t i = 0; i < savedHoles.size(); i++) {
            savedHoles[i].row = (i / 3) + 1;
            savedHoles[i].col = 3 - (i % 3);
            savedHoles[i].position_id = positionMap[{savedHoles[i].row, savedHoles[i].col}];
        }

        for (size_t i = 0; i < savedHoles.size(); i++) {
            circle(emptyFrame, savedHoles[i].center, 8, Scalar(0, 255, 0), 2);
            string label = "R" + to_string(savedHoles[i].row) + "C" + to_string(savedHoles[i].col) +
                " (P" + to_string(savedHoles[i].position_id) + ")";
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

// Function to get position_id from row and column
int getPositionId(int row, int col) {
    auto it = positionMap.find({ row, col });
    if (it != positionMap.end()) {
        return it->second;
    }
    return -1; // Invalid position
}

// Function to find a block of specified color in column 1
Hole* findBlockByColor(int colorCode) {
    for (auto& hole : savedHoles) {
        if (hole.col == 1 && hole.colour == colorCode) {
            return &hole;
        }
    }
    return nullptr;
}

// Function to find blocks in column 3 (for reset operation)
vector<Hole*> findBlocksInColumn3() {
    vector<Hole*> blocks;
    for (auto& hole : savedHoles) {
        if (hole.col == 3 && hole.colour != 0) {
            blocks.push_back(&hole);
        }
    }
    return blocks;
}

// Function to find empty positions in column 1 (for reset operation)
vector<Hole*> findEmptyPositionsInColumn1() {
    vector<Hole*> emptyPositions;
    for (auto& hole : savedHoles) {
        if (hole.col == 1 && hole.colour == 0) {
            emptyPositions.push_back(&hole);
        }
    }
    return emptyPositions;
}

// Function to execute movement based on GUI selection
void executeMoveFromGUI(struct sp_port* port) {
    if (selectedColor == 0 || selectedRow == 0) {
        cout << "Please select both color and row first!" << endl;
        return;
    }

    if (!holesCalibrated || savedHoles.empty()) {
        cout << "Matrix not calibrated yet!" << endl;
        return;
    }

    string colorName = colorNames[selectedColor];
    cout << "Executing move: " << colorName << " block to row " << selectedRow << " column 3" << endl;

    // Find the block to pick (in column 1)
    Hole* pick_hole = findBlockByColor(selectedColor);
    if (!pick_hole) {
        cout << "No " << colorName << " block found in column 1!" << endl;
        return;
    }

    // Find the place position (target row, column 3)
    int place_position = getPositionId(selectedRow, 3);
    if (place_position == -1) {
        cout << "Error: Could not find position for row " << selectedRow << " column 3." << endl;
        return;
    }

    // Find the place hole
    Hole* place_hole = nullptr;
    for (auto& hole : savedHoles) {
        if (hole.position_id == place_position) {
            place_hole = &hole;
            break;
        }
    }

    if (!place_hole) {
        cout << "Error: Could not find hole for the specified place position." << endl;
        return;
    }

    // Check if place position is empty
    if (place_hole->colour != 0) {
        cout << "Place position R" << place_hole->row << "C" << place_hole->col
            << " is not empty! It contains " << colorNames[place_hole->colour] << " block." << endl;
        return;
    }

    cout << "Pick from: R" << pick_hole->row << "C" << pick_hole->col
        << " (Position " << pick_hole->position_id << ")" << endl;
    cout << "Place to: R" << place_hole->row << "C" << place_hole->col
        << " (Position " << place_hole->position_id << ")" << endl;
    cout << "Block color: " << colorName << endl;

    // Use the exact logic for command generation with row numbers
    int pick = pick_hole->row;  // Use row number (1-3)
    int place = place_hole->row; // Use row number (1-3)
    unsigned char cmd = (unsigned char)((((pick - 1) << 4) | (place - 1)) + 1);

    cout << "Generated command: pick_row=" << pick << ", place_row=" << place << ", cmd=" << int(cmd) << endl;

    // Send command sequence
    if (port) {
        sp_blocking_write(port, &cmd, 1, 100);
        cout << "Command sent: " << int(cmd) << endl;

        // Wait 2 seconds
        this_thread::sleep_for(milliseconds(2000));

        // Send zero command
        cmd = 0;
        sp_blocking_write(port, &cmd, 1, 100);
        sp_drain(port);

        cout << "Command reset" << endl;
    }
    else {
        cout << "Serial port not available!" << endl;
    }

    // Update the board state (simulate movement)
    place_hole->colour = pick_hole->colour;
    pick_hole->colour = 0;

    cout << "Movement completed!" << endl;

    // Reset GUI selection
    selectedColor = 0;
    selectedRow = 0;
}

// Function to execute reset operation (move all blocks from C3 to C1)
void executeReset(struct sp_port* port) {
    if (!holesCalibrated || savedHoles.empty()) {
        cout << "Matrix not calibrated yet!" << endl;
        return;
    }

    // Find blocks in column 3 and empty positions in column 1
    vector<Hole*> blocksInC3 = findBlocksInColumn3();
    vector<Hole*> emptyPositionsInC1 = findEmptyPositionsInColumn1();

    if (blocksInC3.empty()) {
        cout << "No blocks found in column 3 to reset!" << endl;
        return;
    }

    if (emptyPositionsInC1.empty()) {
        cout << "No empty positions available in column 1!" << endl;
        return;
    }

    cout << "Starting reset operation..." << endl;
    cout << "Found " << blocksInC3.size() << " blocks in column 3" << endl;
    cout << "Found " << emptyPositionsInC1.size() << " empty positions in column 1" << endl;

    // Move blocks from C3 to C1
    for (size_t i = 0; i < min(blocksInC3.size(), emptyPositionsInC1.size()); i++) {
        Hole* pick_hole = blocksInC3[i];
        Hole* place_hole = emptyPositionsInC1[i];

        cout << "Moving block from R" << pick_hole->row << "C" << pick_hole->col
            << " to R" << place_hole->row << "C" << place_hole->col << endl;
        cout << "Block color: " << colorNames[pick_hole->colour] << endl;

        // Get the command for this specific movement
        auto cmdIt = resetCmdMap.find({ pick_hole->row, place_hole->row });
        if (cmdIt == resetCmdMap.end()) {
            cout << "Error: No command found for movement from R" << pick_hole->row
                << " to R" << place_hole->row << endl;
            continue;
        }

        unsigned char cmd = cmdIt->second;
        cout << "Using command: " << int(cmd) << " for C3R" << pick_hole->row
            << " -> C1R" << place_hole->row << endl;

        // Send command sequence
        if (port) {
            sp_blocking_write(port, &cmd, 1, 100);
            cout << "Command sent: " << int(cmd) << endl;

            // Wait for operation to complete (8 seconds for reset movements)
            this_thread::sleep_for(milliseconds(8000));

            // Send zero command
            cmd = 0;
            sp_blocking_write(port, &cmd, 1, 100);
            sp_drain(port);

            cout << "Command reset" << endl;
        }
        else {
            cout << "Serial port not available!" << endl;
        }

        // Update the board state (simulate movement)
        place_hole->colour = pick_hole->colour;
        pick_hole->colour = 0;

        cout << "Movement " << (i + 1) << " completed!" << endl;

        // Small delay between movements
        if (i < min(blocksInC3.size(), emptyPositionsInC1.size()) - 1) {
            this_thread::sleep_for(milliseconds(1000));
        }
    }

    cout << "Reset operation completed! Moved "
        << min(blocksInC3.size(), emptyPositionsInC1.size()) << " blocks from C3 to C1." << endl;
}

int main(int argc, char* argv[])
{
    if (!global_cap.isOpened()) {
        cout << "Cannot open camera" << endl;
        return -1;
    }

    struct sp_port* port = nullptr;
    int err;

    // Initialize serial port (optional - can run without it)
    if (argc >= 2) {
        err = sp_get_port_by_name("COM3", &port);
        if (err == SP_OK) {
            err = sp_open(port, SP_MODE_WRITE);
            if (err == SP_OK) {
                sp_set_baudrate(port, BAUD);
                sp_set_bits(port, 8);
                cout << "Serial port initialized successfully" << endl;
            }
            else {
                cout << "Warning: Could not open serial port" << endl;
                port = nullptr;
            }
        }
        else {
            cout << "Warning: Could not find serial port" << endl;
            port = nullptr;
        }
    }
    else {
        cout << "Warning: No serial port specified. Running in simulation mode." << endl;
    }

    cout << "Robot Control System Started" << endl;
    cout << "Click 'Calibrate Matrix' button to start" << endl;

    // Display reset command table
    cout << "\n=== Reset Command Table (C3 -> C1) ===" << endl;
    cout << "Pick Position | Place Position | Value | Bits ON (LEDs)" << endl;
    cout << "--------------|----------------|-------|----------------" << endl;
    for (const auto& entry : resetCmdMap) {
        string bitsOn;
        unsigned char cmd = entry.second;
        if (cmd == 129) bitsOn = "DI-9, DI-16";
        else if (cmd == 130) bitsOn = "DI-10, DI-16";
        else if (cmd == 131) bitsOn = "DI-9, DI-10, DI-16";
        else if (cmd == 133) bitsOn = "DI-9, DI-11, DI-16";
        else if (cmd == 134) bitsOn = "DI-10, DI-11, DI-16";
        else if (cmd == 135) bitsOn = "DI-9, DI-10, DI-11, DI-16";
        else if (cmd == 137) bitsOn = "DI-9, DI-12, DI-16";
        else if (cmd == 138) bitsOn = "DI-10, DI-12, DI-16";
        else if (cmd == 139) bitsOn = "DI-9, DI-10, DI-12, DI-16";
        
        cout << "    C3R" << entry.first.first << "     |     C1R" << entry.first.second 
             << "      |  " << int(entry.second) << "   | " << bitsOn << endl;
    }
    cout << "===============================================" << endl << endl;

    // Create control panel window
    namedWindow("Control Panel", WINDOW_NORMAL);
    resizeWindow("Control Panel", 400, 600);
    setMouseCallback("Control Panel", onMouse, port);

    // Create live feed window
    namedWindow("Live Feed", WINDOW_NORMAL);

    // Ensure cmd = 0 first
    unsigned char cmd = 0;
    if (port) {
        sp_blocking_write(port, &cmd, 1, 100);
    }

    while (true) {
        Mat liveFrame;
        if (global_cap.read(liveFrame)) {
            if (holesCalibrated) {
                if (continuousColorDetection) {
                    checkHoleColorsLive(liveFrame);
                    putText(liveFrame, "Live Colour Detection - ON",
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                }
                else {
                    for (size_t i = 0; i < savedHoles.size(); i++) {
                        circle(liveFrame, savedHoles[i].center, 5, Scalar(0, 255, 0), 2);
                    }
                    putText(liveFrame, "Matrix Calibrated - " + to_string(savedHoles.size()) + " positions",
                        Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                }

                // Display current selection on live feed
                string colorName = (selectedColor > 0) ? colorNames[selectedColor] : "None";
                string selectionText = "Selection: " + colorName + " -> Row " + to_string(selectedRow);
                putText(liveFrame, selectionText, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 2);
            }
            else {
                putText(liveFrame, "Click 'Calibrate Matrix' in Control Panel",
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }

            imshow("Live Feed", liveFrame);
        }

        // Update control panel
        createControlPanel();

        int key = waitKey(30);

        if (key == 'q' || key == 'Q' || key == 27) {
            cout << "Quitting..." << endl;
            if (port) {
                sp_close(port);
            }
            break;
        }
    }

    if (port) {
        sp_close(port);
    }
    return 0;
}
