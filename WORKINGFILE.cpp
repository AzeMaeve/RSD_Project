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

vector<Point2f> detectHoles(const Mat& frame, int iLowV, int iHighV)
{
	vector<Point2f> holeCenters;

	// Convert to HSV
	Mat imgHSV;
	cvtColor(frame, imgHSV, COLOR_BGR2HSV);

	// We only care about brightness (Value channel)
	vector<Mat> hsvChannels;
	split(imgHSV, hsvChannels);
	Mat V = hsvChannels[2];

	// Threshold using the Control window's Value range
	Mat mask;
	inRange(V, Scalar(iLowV), Scalar(iHighV), mask);

	// The mask should highlight the darker region (board)
	// But we want to find the WHITE holes inside it → invert it
	bitwise_not(mask, mask);

	// Clean up noise
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	// Find contours of the white holes
	vector<vector<Point>> contours;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (auto& contour : contours) {
		double area = contourArea(contour);
		if (area < 100 || area > 10000) continue; // filter noise

		Rect bbox = boundingRect(contour);
		float aspect = (float)bbox.width / bbox.height;
		if (aspect > 0.8 && aspect < 1.2) {
			Moments m = moments(contour);
			Point2f center(m.m10 / m.m00, m.m01 / m.m00);
			holeCenters.push_back(center);
		}
	}

	// Sort into a 3x3 order (top-left to bottom-right)
	sort(holeCenters.begin(), holeCenters.end(), [](Point2f a, Point2f b) {
		if (abs(a.y - b.y) > 20) return a.y < b.y;
		return a.x < b.x;
		});

	// Visualize detection
	Mat vis = frame.clone();
	for (size_t i = 0; i < holeCenters.size(); ++i) {
		circle(vis, holeCenters[i], 6, Scalar(0, 0, 255), -1);
		putText(vis, to_string(i + 1), holeCenters[i] + Point2f(10, 0),
			FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	}

	imshow("Detected Holes", vis);
	imshow("Board Mask", mask);

	return holeCenters;
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

	}


	// ========= Serial Comms ===========================
	struct sp_port* port;
	int err;
	int key = 0;
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

	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 0;
	int iHighS = 255;

	int iLowV = 70;
	int iHighV = 200;

	// =========== TEST =================
	Mat calibrationFrame;
	cap >> calibrationFrame;
	vector<Point2f> holeCenters = detectHoles(calibrationFrame, iLowV, iHighV);

	// Create trackbars in "Control" window
	createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);

	Mat imgTmp;
	cap.read(imgTmp);

	Mat imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);

		// =========== MAIN LOOP ==============================
		printMenu(); // Print Menu to console

		while (key != 'q') {
			cap >> frame;
			/*Reads and outputs a single pixel value at (10,15)*/
			// Vec3b intensity = frame.at<Vec3b>(10, 15);
			// int blue = intensity.val[0];
			// int green = intensity.val[1];
			// int red = intensity.val[2];
			// cout << "Intensity = " << endl << " " << blue << " " << green << " " << red << endl << endl;

			/*Modify the pixels of the RGB image */
			//for (int i = 150; i < frame.rows; i++)
			//{
			//	for (int j = 150; j < frame.cols; j++)
			//	{
			//		/*The following lines make the red and blue channels zero
			//		(this section of the image will be shades of green)*/
			//		frame.at<Vec3b>(i, j)[0] = 0;
			//		frame.at<Vec3b>(i, j)[2] = 0;
			//	}
			//}

			Mat imgOriginal;

			bool bSuccess = cap.read(imgOriginal); // read a new frame from video

			if (!bSuccess) //if not success, break loop
			{
				cout << "Cannot read a frame from video stream" << endl;
				break;
			}

			Mat imgHSV;

			cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

			Mat imgThresholded;

			inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

			//morphological opening (remove small objects from the foreground)
			erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

			//morphological closing (fill small holes in the foreground)
			dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

			// ========= TEST =============================
			for (int i = 0; i < holeCenters.size(); ++i)
			{
				Point2f c = holeCenters[i];
				int roiSize = 5;
				Rect roi(c.x - roiSize, c.y - roiSize, roiSize * 2, roiSize * 2);
				roi &= Rect(0, 0, imgHSV.cols, imgHSV.rows); // keep within bounds

				Mat roiHSV = imgHSV(roi);
				Scalar avgHSV = mean(roiHSV);

				bool colorDetected =
					avgHSV[0] > iLowH && avgHSV[0] < iHighH &&
					avgHSV[1] > iLowS && avgHSV[1] < iHighS &&
					avgHSV[2] > iLowV && avgHSV[2] < iHighV;

				if (colorDetected)
					circle(imgOriginal, c, 10, Scalar(0, 255, 0), 2); // green = detected
				else
					circle(imgOriginal, c, 10, Scalar(0, 0, 255), 2); // red = empty
			}

			// Calculate the moments of the thresholded image
			Moments oMoments = moments(imgThresholded);

			double dM01 = oMoments.m01;
			double dM10 = oMoments.m10;
			double dArea = oMoments.m00;

			// if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
			if (dArea > 10000)
			{
				// Calculate the position of the object
				int posX = dM10 / dArea;
				int posY = dM01 / dArea;
				// Print coordinates to console
				cout << posX << ", " << posY << endl;
			}

			imshow("Thresholded Image", imgThresholded); //show the thresholded image
			imshow("Original", imgOriginal); //show the original image

			char key = (char)waitKey(25);
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
				return 0;
			case ' ': //Save an image
				sprintf_s(filename, "filename%.3d.jpg", n++);
				imwrite(filename, frame);
				cout << "Saved " << filename << endl;
				break;
			default:
				break;
			}
		}
		sp_close(port);
		return 0;

}
