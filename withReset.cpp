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

// GUI state variables
int selectedColor = 0; // 0=None, 1=Red, 2=Blue, 3=Green
int selectedRow = 0;   // 0=None, 1-3=Row number
bool commandReady = false;

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
// Maps from C3 (pick) to C1 (place) movements
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

// Function to create GUI control window
void createGUI() {
    namedWindow("Control Panel", WINDOW_NORMAL);
    resizeWindow("Control Panel", 400, 450);

    // Create trackbars for selection
    createTrackbar("Colour", "Control Panel", &selectedColor, 3);
    createTrackbar("Row", "Control Panel", &selectedRow, 3);

    // Display instructions
    Mat controlPanel = Mat::zeros(300, 400, CV_8UC3);
    putText(controlPanel, "Control Instructions:", Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    putText(controlPanel, "1-3: Select Colour (R/B/G)", Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    putText(controlPanel, "1-3: Select Row (1-3)", Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    putText(controlPanel, "Space: Execute Move", Point(10, 130), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    putText(controlPanel, "c: Calibrate Matrix", Point(10, 160), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    putText(controlPanel, "h: Go Home", Point(10, 190), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    putText(controlPanel, "r: Reset (C3 -> C1)", Point(10, 220), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    putText(controlPanel, "q: Quit", Point(10, 250), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

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

    // Update trackbars
    setTrackbarPos("Colour", "Control Panel", 0);
    setTrackbarPos("Row", "Control Panel", 0);
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
        auto cmdIt = resetCmdMap.find({pick_hole->row, place_hole->row});
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
    VideoCapture cap(0);
    if (!cap.isOpened()) {
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
    cout << "Press 'c' to calibrate matrix first" << endl;

    // Create GUI
    createGUI();

    // Display reset command table
    cout << "\n=== Reset Command Table (C3 -> C1) ===" << endl;
    cout << "Pick Row | Place Row | Command" << endl;
    cout << "---------|-----------|--------" << endl;
    for (const auto& entry : resetCmdMap) {
        cout << "   C3R" << entry.first.first << "   |    C1R" << entry.first.second 
             << "    |   " << int(entry.second) << endl;
    }
    cout << "===================================" << endl << endl;

    // Ensure cmd = 0 first
    unsigned char cmd = 0;
    if (port) {
        sp_blocking_write(port, &cmd, 1, 100);
    }

    while (true) {
        Mat liveFrame;
        if (cap.read(liveFrame)) {
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
                putText(liveFrame, "Press 'c' to calibrate matrix",
                    Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }

            imshow("Live Feed", liveFrame);
        }

        int key = waitKey(30);

        if (key != -1) {
            switch (key) {
            case 'c':
            case 'C':
                if (captureEmptyFrame(cap)) {
                    cout << "Calibration successful!" << endl;
                    continuousColorDetection = true;
                    cout << "Continuous color detection started automatically" << endl;
                }
                else {
                    cout << "Calibration failed. Adjust camera/view and try again." << endl;
                }
                break;

            case 'r': // Reset C3 to C1
                executeReset(port);
                break;

            case 'h': // Set to home
                cmd = 64;
                if (port) {
                    sp_blocking_write(port, &cmd, 1, 100);
                    this_thread::sleep_for(milliseconds(2000));
                    cmd = 0;
                    sp_blocking_write(port, &cmd, 1, 100);
                }
                break;

            case '0':
                if (holesCalibrated) {
                    continuousColorDetection = !continuousColorDetection;
                    cout << "Continuous color detection: "
                        << (continuousColorDetection ? "ON" : "OFF") << endl;
                }
                else {
                    cout << "Please calibrate matrix first (press 'c')" << endl;
                }
                break;

            case '1': selectedColor = 1; cout << "Selected: Red" << endl; break;
            case '2': selectedColor = 2; cout << "Selected: Blue" << endl; break;
            case '3': selectedColor = 3; cout << "Selected: Green" << endl; break;
            case '4': selectedRow = 1; cout << "Selected: Row 1" << endl; break;
            case '5': selectedRow = 2; cout << "Selected: Row 2" << endl; break;
            case '6': selectedRow = 3; cout << "Selected: Row 3" << endl; break;
            case ' ':
                if (selectedColor > 0 && selectedRow > 0) {
                    executeMoveFromGUI(port);
                }
                else {
                    cout << "Please select both color and row first!" << endl;
                }
                break;

            case 'q':
            case 'Q':
            case 27:
                cout << "Quitting..." << endl;
                if (port) {
                    sp_close(port);
                }
                return 0;

            default:
                break;
            }
        }
    }

    if (port) {
        sp_close(port);
    }
    return 0;
}
