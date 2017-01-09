/*
attempt to port matlab code in C++11 w/OpenCV3 based on 

http://www.eng.tau.ac.il/~simonk/CSH/

Coherency Sensitive Hashing (CSH)
Simon Korman and Shai Avidan

abstract
Coherency Sensitive Hashing (CSH) extends Locality Sensitivity Hashing (LSH) and PatchMatch to quickly find matching patches between two images.
 LSH relies on hashing, which maps similar patches to the same bin, in order to find matching patches. 
 PatchMatch, on the other hand, relies on the observation that images are coherent, to propagate good matches to their neighbors, in the image plane. 
 It uses random patch assignment to seed the initial matching. CSH relies on hashing to seed the initial patch matching and on image coherence to propagate 
 good matches. In addition, hashing lets it propagate information between patches with similar appearance (i.e., map to the same bin). 
 This way, information is propagated much faster because it can use similarity in appearance space or neighborhood in the image plane. 
 As a result, CSH is at least three to four times faster than PatchMatch and more accurate, especially in textured regions, where reconstruction 
 artifacts are most noticeable to the human eye. We verified CSH on a new, large scale, data set of 133 image pairs.
*/
#include <tuple>
#include <iostream>
using namespace std;
#include <opencv2/opencv.hpp>
using namespace cv;
#include <boost/filesystem.hpp>

#include "utils.h"
#include "csh.h"



int main(int argc, char** argv)
{
	cout << "CSH algorithm example script!!!" << endl;
	cout << "*******************************" << endl;

	// preparing the images
	boost::filesystem::path img1("Saba1.bmp");
	boost::filesystem::path img2("Saba2.bmp");

	Mat	A = imread(img1.string(),IMREAD_COLOR);
	if(A.empty()) {
		cerr << basename(argv[0]) << ": could not open file or invalid image file  '" << img1 << "'" << endl;
		exit(-1);
	}
	Mat	B = imread(img2.string(),IMREAD_COLOR);
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

	// warmup
	cout << endl << "--> Dummy run to warmup . . .  " << endl;

	pair<Mat,Mat>	nnf= CSH::nn(A,B);
	Mat&	CSH_ann=nnf.first;
	
	cout << "Done!!" << endl;

	cout << endl;
	return 0;
}
