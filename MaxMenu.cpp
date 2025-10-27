#include <stdio.h>
#include <stdlib.h>
#include <libserialport.h>

#define BAUD 9600

#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <thread>         
#include <chrono>  

using namespace cv;
using namespace std;
void showMenu() {
	cout << "\n";
	cout << "=== Simple Menu ===\n";
	cout << "1. Option One\n";
	cout << "2. Option Two\n";
	cout << "3. Option Three\n";
	cout << "4. Option Four\n";
	cout << "Choose an option (1-4): ";
}


int main(int argc, char* argv[])
{

	
	/*Variables for camera function*/
	int n = 0;
	char filename[200];
	string window_name = "video | q or esc to quit";
	Mat frame;
	/*Setup camera and check for camera*/
	namedWindow(window_name);
	VideoCapture cap(0);
	if (!cap.isOpened()) {

		cout << "cannot open camera";

	}

	/*Variables for serial comms */
	struct sp_port* port;
	int err;
	char key = 0;
	unsigned char cmd;

	int pick = 0, place = 0;
	unsigned char cmdPacked = 0;

	/* Set up and open the port */
	/* check port usage */
	if (argc < 2)
	{
		/* return error */
		fprintf(stderr, " Port use\n");
		exit(1);
	}

	/* get port name */
	err = sp_get_port_by_name(argv[1], &port);
	if (err == SP_OK)
		/* open port */
		err = sp_open(port, SP_MODE_WRITE);
	if (err != SP_OK)
	{
		/* return error */
		fprintf(stderr, " Can't open port %s\n", argv[1]);
		exit(2);
	}

	/* set Baud rate */
	sp_set_baudrate(port, BAUD);
	/* set the number of bits */
	sp_set_bits(port, 8);

	/* specify the comand to send to the port */
	
	/* set up to exit when q key is entered */
	cmd = 0;
	sp_blocking_write(port, &cmd, 1, 100);
	showMenu();
	while (key != 'q') {
		/*cap >> frame;*/
	

		/*The code contained here reads and outputs a single pixel value at (10,15)*/
		//Vec3b intensity = frame.at<Vec3b>(10, 15);
		//int blue = intensity.val[0];
		//int green = intensity.val[1];
		//int red = intensity.val[2];
		//cout << "Intensity = " << endl << " " << blue << " " << green << " " << red << endl << endl;
		/*End of modifying pixel values*/

		/*The code contained here modifies the output pixel values*/
			/* Modify the pixels of the RGB image */
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
		/*End of modifying pixel values*/

		//imshow(window_name, frame);
		/*char key = (char)waitKey(25);*/

		
		/* write the number "cmd" to the port */
		
		
		cin >> (key);
		if (!isdigit(key) && key != 'q' && key != 'Q' && key != 27) {
			cerr << "Invalid input. Please enter a number (1–4) or 'q' to quit.\n";
			showMenu();
			continue;  
		}
		switch (key) {
		
		case '1':
			cout << "Pick block from column 1 (1-3): ";  cin >> pick;
			cout << "Place block in column 3 (1-3): ";   cin >> place;
			if (pick < 1 || pick > 3 || place < 1 || place > 3) {
				cerr << "Invalid Input indexes must be 1..3\n";
				showMenu();
				break;
			}

			// pack and send the command byte
			cmd = (unsigned char)((((pick - 1) << 4) | (place - 1)) + 1);
			sp_blocking_write(port, &cmd, 1, 100);
			cout << int(cmd);
			std::this_thread::sleep_for(std::chrono::milliseconds(2000));
			cmd = 0;
			sp_blocking_write(port, &cmd, 1, 100); 
			sp_drain(port);
			showMenu();
			break;
		case '2':
			cout << "You chose Option 2." << endl;
			showMenu();
			break;
		case '3':
			sp_blocking_write(port, &cmd, 1, 100);
			showMenu();
			break;
			
		case '4':
			cout << "Exiting program..." << endl;
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
	/* close the port */
	sp_close(port);
	return 0;
}
