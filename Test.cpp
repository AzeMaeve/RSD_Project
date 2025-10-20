vector<Point2f> detectHoles(const Mat& frame)
{
    vector<Point2f> holeCenters;

    Mat gray, blurImg, thresh;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(5, 5), 0);
    threshold(blurImg, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours) {
        double area = contourArea(contour);
        if (area < 200 || area > 10000) continue;

        Rect bbox = boundingRect(contour);
        float aspect = (float)bbox.width / bbox.height;

        if (aspect > 0.8 && aspect < 1.2) {
            Moments m = moments(contour);
            Point2f center(m.m10 / m.m00, m.m01 / m.m00);
            holeCenters.push_back(center);
        }
    }

    // Sort into 3×3 grid (top-left → bottom-right)
    sort(holeCenters.begin(), holeCenters.end(), [](Point2f a, Point2f b) {
        if (abs(a.y - b.y) > 20) return a.y < b.y;
        return a.x < b.x;
    });

    // Visualize detected holes
    Mat vis = frame.clone();
    for (size_t i = 0; i < holeCenters.size(); ++i) {
        circle(vis, holeCenters[i], 6, Scalar(0, 0, 255), -1);
        putText(vis, to_string(i + 1), holeCenters[i] + Point2f(10, 0),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }
    imshow("Detected Holes", vis);

    return holeCenters;
}







// === Detect hole positions once at startup ===
Mat calibrationFrame;
cap >> calibrationFrame;
vector<Point2f> holeCenters = detectHoles(calibrationFrame);

if (holeCenters.size() != 9)
    cout << "Warning: Detected " << holeCenters.size() << " holes instead of 9." << endl;
else
    cout << "Hole calibration complete.\n";




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
