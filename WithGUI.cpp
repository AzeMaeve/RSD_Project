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
bool guiInitialized = false;

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

// Mouse callback for control panel
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        Point pt(x, y);
        
        // Check which button was clicked
        if (calibrateBtn.contains(pt)) {
            // Calibrate button - this will be handled in main loop
            cout << "Calibrate button clicked - press 'c' in live feed window" << endl;
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
            cout << "Execute button clicked - use Spacebar in live feed window" << endl;
        }
        else if (resetBtn.contains(pt)) {
            cout << "Reset button clicked - use 'r' in live feed window" << endl;
        }
        else if (homeBtn.contains(pt)) {
            cout << "Home button clicked - use 'h' in live feed window" << endl;
        }
        else if (colorDetectionBtn.contains(pt)) {
            if (holesCalibrated) {
                continuousColorDetection = !continuousColorDetection;
                cout << "Continuous color detection: " << (continuousColorDetection ? "ON" : "OFF") << endl;
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
    putText(controlPanel, "Calibrate Matrix (Press 'c')", Point(60, 130), 
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
    rectangle(controlPanel, executeBtn, Scalar(0, 100, 0), -1);
    rectangle(controlPanel, executeBtn, Scalar(200, 200, 200), 2);
    putText(controlPanel, "EXECUTE MOVE (Press Spacebar)", Point(80, 330), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    
    // Current selection display
    string selectionText = "Current: " + colorNames[selectedColor] + " -> Row " + to_string(selectedRow);
    putText(controlPanel, selectionText, Point(20, 360), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 0), 1);
    
    // Special commands
    rectangle(controlPanel, resetBtn, Scalar(0, 0, 100), -1);
    rectangle(controlPanel, resetBtn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "RESET (r)", Point(70, 395), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    rectangle(controlPanel, homeBtn, Scalar(100, 0, 0), -1);
    rectangle(controlPanel, homeBtn, Scalar(200, 200, 200), 1);
    putText(controlPanel, "HOME (h)", Point(230, 395), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    // Color detection toggle
    rectangle(controlPanel, colorDetectionBtn, continuousColorDetection ? Scalar(0, 100, 0) : Scalar(50, 50, 50), -1);
    rectangle(controlPanel, colorDetectionBtn, Scalar(200, 200, 200), 1);
    string detectionText = continuousColorDetection ? "Color Detection: ON (0 to toggle)" : "Color Detection: OFF (0 to toggle)";
    putText(controlPanel, detectionText, Point(60, 470), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    
    // Instructions
    putText(controlPanel, "Instructions:", Point(20, 520), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(controlPanel, "Use keyboard in Live Feed window for actions", Point(20, 540), 
            FONT_HERSHEY_SIMPLEX, 0.3, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Click buttons here for selection only", Point(20, 560), 
            FONT_HERSHEY_SIMPLEX, 0.3, Scalar(200, 200, 200), 1);
    putText(controlPanel, "Press 'q' in Live Feed to quit", Point(20, 580), 
            FONT_HERSHEY_SIMPLEX, 0.3, Scalar(200, 200, 200), 1);
    
    imshow("Control Panel", controlPanel);
}

// [Include all your existing functions here: detectColour, detectBoard, detectHolesInBoard, 
// captureEmptyFrame, checkHoleColorsLive, getPositionId, findBlockByColor, 
// findBlocksInColumn3, findEmptyPositionsInColumn1, executeMoveFromGUI, executeReset]
// ... (Include all your existing function implementations here)

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
    cout << "Press 'c' in Live Feed window to calibrate matrix first" << endl;

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
    setMouseCallback("Control Panel", onMouse);

    // Create live feed window
    namedWindow("Live Feed", WINDOW_NORMAL);

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

        // Update control panel
        createControlPanel();

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
