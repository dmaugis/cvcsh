/*
#include <limits>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;
#include <opencv2/opencv.hpp>
using namespace cv;
#include <boost/filesystem.hpp>
*/
#include <tuple>
#include <iostream>
using namespace std;
#include <opencv2/opencv.hpp>
using namespace cv;
#include <boost/filesystem.hpp>

#include "utils.h"


int main(int argc, char** argv)
{
	cout << "CSH algorithm example script!!!" << endl;
	cout << "*******************************" << endl;

	// preparing the images
	boost::filesystem::path img1("Saba1.bmp");
	boost::filesystem::path img2("Saba2.bmp");

	Mat	A = imread(img1.string());
	if(A.empty()) {
		cerr << basename(argv[0]) << ": could not open file or invalid image file  '" << img1 << "'" << endl;
		exit(-1);
	}
	Mat	B = imread(img2.string());
	if(B.empty()) {
		cerr << basename(argv[0]) << ": could not open file or invalid image file  '" << img2 << "'" << endl;
		exit(-1);
	}

	int hA,wA,dA;
	tie(hA,wA,dA)=size(A);

	int hB,wB,dB;
	tie(hB,wB,dB)=size(B);

	float	mpa = floor(float(hA*wA)/1000.) / 1000.;
	float	mpb = floor(float(hB*wB)/1000.) / 1000.;
	
	cout << "Image A: " << img1 << ", size = "<< mpa << " MP" << endl;
	cout << "Image B: " << img2 << ", size = "<< mpb << " MP" << endl;
	
	
	
	cout << endl;
	return 0;
}
