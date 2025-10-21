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

	int iLowV = 50;
	int iHighV = 200;

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
			// cout << posX << ", " << posY << endl;
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
