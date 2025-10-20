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
    erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
    dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
    dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
    erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

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










Mat calibrationFrame;
cap >> calibrationFrame;
vector<Point2f> holeCenters = detectHoles(calibrationFrame, iLowV, iHighV);


if (key == 'c') {
    cout << "Recalibrating holes..." << endl;
    vector<Point2f> newCenters = detectHoles(frame, iLowV, iHighV);
    if (newCenters.size() == 9) {
        holeCenters = newCenters;
        cout << "Recalibration complete." << endl;
    } else {
        cout << "Recalibration failed — detected " << newCenters.size() << " holes.\n";
    }
}


