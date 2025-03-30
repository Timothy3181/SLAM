#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat frame;
    frame = imread("/Users/timothy3181/Desktop/yuanshen.png");
    imshow("frame", frame);
    waitKey(0);
    cout << "Hello" << endl;
}
