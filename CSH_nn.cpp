#include <tuple>
#include <iostream>
using namespace std;
#include "csh.h"
using namespace cv;

namespace
{

class HashingSchemeParams
{
public:
	short	width;
	short	dist;
	uint		maxKernels;
	uint		TLboundary;
	uint		BRboundary;
	uint		br_boundary_to_ignore;
	short	numHashTables;
	uint		numHashs;
	class DescriptorParams
	{
	public:
		bool		descriptor_mode;
		bool		rotation_invariant;
		int		hA,wA,dA;
		int		hB,wB,dB;
	} descriptor_params;
};


//function [CFIs_A,CFIs_B,nSequencyOrder16u,nSequencyLevels16u , WHK_with_Cb_Cr] =
//GetResultsOfKernelApplication(A,B,TLboundary,BRboundary,width,classType,maxKernels,hA,wA,dA,hB,wB )
vector<Mat> GetResultsOfKernelApplication(Mat A,Mat B,uint TLboundary,uint BRboundary,uint width,int classType,uint maxKernels,uint hA=0,uint wA=0,uint dA=0,uint hB=0,uint wB=0)
{
	/*
	% the last 5 parameters are relevant to 'patch' mode, which means that A and B contain flattened
	% patches, i.e. they are of the size: % size(A) = [width^2,hA*wA,dA]
	% IMPORTANT: note that in this case, we will add at the bottom and right zero padding of (width-1)
	*/
	bool  patch_mode = (hA!=0);
	if (!patch_mode) {
		// A] Padding
		// pre padding of '2*width' is for the fast and correct initialization of the first 'width' rows/cols
		Mat preA,preB;
		copyMakeBorder(A,preA, TLboundary, 0, TLboundary, 0, BORDER_CONSTANT, 0 );		//preA = padarray(A,[TLboundary,TLboundary],0,'pre');
		copyMakeBorder(B,preB, TLboundary, 0, TLboundary, 0, BORDER_CONSTANT, 0 );

		// sizes
		uint	pad_hA=preA.rows;		//[pad_hA,pad_wA,dA] = size(preA);
		uint	pad_wA=preA.cols;
		uint	dA=A.depth();
		uint	pad_hB=preB.rows;		//[pad_hB,pad_wB,dB] = size(preB);
		uint	pad_wB=preB.cols;
		uint	dB=B.depth();
	} else {
		uint shrink_hA = hA - width + 1; // hA includes a BR boundary of size (width-1)
		uint shrink_wA = wA - width + 1;
		uint shrink_hB = hB - width + 1;
		uint shrink_wB = wB - width + 1;
	}
	int	ChannelColors = dA;		// tells if this is Y/Cb/Cr or Y only image
	// B] Traverse order and filter specifications
//% obtaining GCKs traverse data
//[GCKs2D,snakeOrder,deltas1D,alphaDirs1D] = GetGCKTraverseParams(width);
	// Define the sequence in which the kernels are used for candidate check
	list<ushort>	nSequencyOrder16u;
	list<ushort>	filters;
	list<ushort>	filtersY;
	list<int>		procFilterIndToUse;
	list<int>		procSnakeIndToUse;
	int				LastCbCrFilterIndex;
	switch(width) {
	case 2: {
		nSequencyOrder16u=list<ushort> {1,2,3,4,6,5};
		filters=list<ushort> {1,2,3,4,7,10};
		filtersY=list<ushort> {1,3,7,10};
		procFilterIndToUse= list<int> {0 ,1,4,5};
		procSnakeIndToUse=list<int> {0 ,1,2,3};
		LastCbCrFilterIndex = 3;
	}
	break;
	case 4: {
		nSequencyOrder16u=list<ushort> {1,4,10,2,3,13,17,7,16,14,15,5,11,6,12,8,9};
		filters=list<ushort> {1,2,3,4,5,6,7,8,9,10,11,12,13,16,19,22,25};
		filters.sort();
		filtersY=list<ushort> {1 ,4 ,7 ,10 ,13 ,16 ,19, 22, 25};
		procFilterIndToUse=list<int> {0,1, 4, 7, 10, 13, 14, 15, 16};
		procSnakeIndToUse=list<int> {0, 1, 2, 3, 4, 5, 6, 7, 8};
		LastCbCrFilterIndex = 12;
	}
	break;
	case 8: {
		nSequencyOrder16u=list<ushort> {1,  4,10,  2,3,  13,17,  7,18,21,16,14,15,5,6,11,12, 8,9,22,23,19,20};
		filters=list<ushort> {1,4,10,  2,3,13,25,  7,28,46,22,16,19,5,6,11,12,  8,9,49,73,31,43};
		filters.sort();
		filtersY=list<ushort> { 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 43, 46, 49 , 73 };
		procFilterIndToUse=list<int> {0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, -1,  -1,  -1,  14,  20,  21, -1, -1,  -1, -1, -1,  -1, -1,  18};
		procSnakeIndToUse=list<int> {0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10, -1,  -1,  -1,   6,  15,  16, -1, -1,  -1, -1, -1,  -1, -1,  10};
		LastCbCrFilterIndex = 12;
	}
	break;
	case 16: {
		nSequencyOrder16u=list<ushort> {1, 3,  5,  7,  17,  42};
		filters=list<ushort> { 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 97,100 ,103,106, 109, 112, 115, 139, 142, 145 };
		filters.sort();
		procFilterIndToUse=list<int> { 0, 1, 4, 7, 10, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,29,30 ,31 ,32,33,34,35,36,-1,-1,-1,28,38,39,40 ,41,42,43,-1,-1,-1 ,-1 ,-1 ,-1 ,-1 ,36 ,45 ,46};
		procSnakeIndToUse=list<int> { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,-1 ,-1 ,-1,20,33,34,35,36,37,38,-1,-1,-1,-1,-1,-1,-1,28,47,48	};
		LastCbCrFilterIndex = 12;
	}
	break;
	case 32:
			throw runtime_error("not implemented yet...");
	default:
		assert(0);
	}

	auto WHK_with_Cb_Cr = filters;

/* TBC */

}





pair<Mat,Mat>		HashingSchemeNew(Mat rgbA,Mat rgbB,short k,bool calcBnn,HashingSchemeParams& parameters,Mat mask=Mat(),bool patch_mode=false)
{
	if ((k>1) && calcBnn)
		throw runtime_error("This combination of ''(k>1) && (calcBnn)'', currently doesn''t work due to a bug. compute for each direction separately...");
// A] extracting input parameters
	ushort	width = parameters.width;
	bool		descriptor_mode = parameters.descriptor_params.descriptor_mode;
	ushort	maxKernels = parameters.maxKernels;
//useTics = parameters.useTicsInside;
	uint		TLboundary = parameters.TLboundary;
	uint		BRboundary = parameters.BRboundary;
	uint		br_boundary_to_ignore = parameters.br_boundary_to_ignore;
	uint		numTables = parameters.numHashTables;
	uint		numHashs = parameters.numHashs;
//insideInfo = parameters.insideInfo;
//CalcErrorImages = parameters.DebugCalcErrors;
	uint		dist = parameters.dist; // this is for CSH that keeps 'dist' pixels in x or y away from identity
//KNN_enrichment_mode = parameters.KNN_enrichment_mode;
	bool		rotation_invariant = false;
	Mat A,B;
	uint		hA,wA,dA,hB,wB,dB;
	if(!descriptor_mode) {
		if(!patch_mode) {
			hA=rgbA.rows;
			wA=rgbA.cols;
			dA=rgbA.depth();
			hB=rgbB.rows;
			wB=rgbA.cols;
			dB=rgbB.depth();
			if(dA==3)	cvtColor(rgbA, A, COLOR_BGR2YCrCb);
			else			A=rgbA;
			if(dB==3)	cvtColor(rgbB, B, COLOR_BGR2YCrCb);
			else			B=rgbB;
		} else {
			/* patch mode */
			A = rgbA;
			B = rgbB; // the input A has size: size(A) = [width^2,(hA-[width-1])*(wA-[width-1])*dA]
			hA = parameters.descriptor_params.hA;
			wA = parameters.descriptor_params.wA;
			hB = parameters.descriptor_params.hB;
			wB = parameters.descriptor_params.wB;
			dA = parameters.descriptor_params.dA;
			dB = dA;
			rotation_invariant = parameters.descriptor_params.rotation_invariant;
			/*
			        if (rotation_invariant && isequal(A,B) && k<=1)
			            dist = 1;
			        end
			*/
		}
	} else {	//	descriptor_mode
		/*
		if ((ndims(rgbA) ~= 2) || (ndims(rgbB) ~= 2))
			error('Input data (both A and B) must be of dimensions (Descripotr_Size*(W*H))');
		end
		*/
		if(rgbA.type()!=CV_32F)	throw	runtime_error("Input data A must be of type float");
		if(rgbB.type()!=CV_32F)	throw	runtime_error("Input data B must be of type float");
		A = rgbA;
		B = rgbB;
		hA = parameters.descriptor_params.hA;
		wA = parameters.descriptor_params.wA;
		hB = parameters.descriptor_params.hB;
		wB = parameters.descriptor_params.wB;
		dA = 1;
		dB = 1;
		width = 1;
	}
	if (dA != dB)
		throw	runtime_error("Image color channels must be the same: both RGB or both gray level images");
	ushort	ColorChannels = dA;
// long term initializations
	Size	sizA(hA,wA);
	Size sizB(hB,wB);

	if(!mask.empty()) {
		uint	hm=mask.rows;
		uint	wm=mask.cols;
		uint	dm=mask.depth();
		if((hm!= hB) || (wm != wB) | (dm != 1))
			throw	runtime_error("mask image must have the same dimensions as target image (image B)");
		if (calcBnn)
			if((hm!= hA) || (wm != wA) | (dm != 1))
				throw	runtime_error("mask image must have the same dimensions as target image (image A)");
	}
// C] PARAMETERS/INITIALIZATIONS 2 - Actual things
// depending on the patch width - how many kernels (maximum) do we want to compute
	uint maxBits = 0;
	if (!descriptor_mode) {
		set<ushort> SupportedWidth { 2, 4, 8, 16, 32};
		if(SupportedWidth.find(width)==SupportedWidth.end())
			throw	runtime_error("input patch width not supported");
		// maxBits = the number of bits in the code
		switch(width) {
		case 2:
			maxKernels = 2*2;
			maxBits = 15;
			break;
		case 4:
			maxKernels = 3*3;
			maxBits = 17;
			break;
		case 8:
			maxKernels = 5*5;
			maxBits = 18;
			break;
		case 16:
			maxKernels = 7*7;
			maxBits = 18;
			break;
		case 32:
			maxKernels = 9*9;
			maxBits = 20;
			break;
		}
		maxKernels = maxKernels * ColorChannels;
	} else {	// descriptor_mode
		width = 8; // Width for descriptor mode
		//[Descriptor_Width_A NumProjections] = size(A);
		throw	runtime_error("think again");
	}

// prepare result matrices
// Mat	AnnA2B = ones(hA,wA,d_mapping,k,'int32');
//  if (calcBnn)AnnB2A = ones(hB,wB,d_mapping,k,'int32');

// choose element type
	int classType=CV_16S;
	if (!descriptor_mode)
		if (width > 8)
			classType = CV_32S;
// - nBestMapping32uA: of size like A, holds the current best found mapping which is by a FLAT index into B, that runs column after column
// - bestErrorsNewA  : of size like A, holds the approximated errors (GCK errors, not SSD errors) of the current best mapping
	cout << "patch mode: " << patch_mode << endl;
//nBestMapping32uA = zeros(hA,wA,k,'uint32'); % current best mapping
//bestErrorsNewA = zeros(hA,wA,k,'uint32'); % current (GCK)-errors (approximation of SSD error)
	if(patch_mode) { // here [hA,wA] is the size of the (width-1)-BR-padded valid patches image and therefore, each currFiltImgs_A is of size [hA,wA]
		// Getting Walsh Hadamard GCK projections for patch mode...
		// The input A has size: size(A) = [width^2,(hA-[width-1])*(wA-[width-1])*dA]
		// [currFiltImgs_A,currFiltImgs_B,nSequencyOrder16u,nSequencyLevels16u ,WHK_with_Cb_Cr] = GetResultsOfKernelApplication(A,B,TLboundary,BRboundary,width,classType,maxKernels,hA,wA,dA,hB,wB);
		vector<Mat> v=GetResultsOfKernelApplication(A,B,TLboundary,BRboundary,width,classType,maxKernels,hA,wA,dA,hB,wB);
	} else if(descriptor_mode) {
		// Getting principle component analysis results...
		// [currFiltImgs_A,currFiltImgs_B,PCA_A,PCA_B,nSequencyOrder16u,nSequencyLevels16u,MaxDescriptorIntVal] = GetDescriptorPCA(A,B,hA,wA,hB,wB,TLboundary,BRboundary,classType,maxKernels);
		// WHK_with_Cb_Cr = []; % to maintain compatibility
	} else {
		// Getting Walsh Hadamard GCK projections
		// [currFiltImgs_A,currFiltImgs_B,nSequencyOrder16u,nSequencyLevels16u ,WHK_with_Cb_Cr] =GetResultsOfKernelApplication(A,B,TLboundary,BRboundary,width,classType,maxKernels);
		vector<Mat> v=GetResultsOfKernelApplication(A,B,TLboundary,BRboundary,width,classType,maxKernels);
	}
	return make_pair(Mat(),Mat());
}


}

namespace CSH
{

pair<Mat,Mat>	nn(Mat A,Mat B,short width,short iterations,short k,bool calcBnn,Mat bMask,short distFromIdentity,PatchParams* patch_params,bool fastKNN)
{
	bool	patch_mode=false;
	if(floor(log2(width)) != log2(width))	throw runtime_error("width must be a power of 2");
	// A] PREPARATIONS
	// 1) CSH - parameters preparation and packing
	uint		numHashs = 2;	// width of hash table
	uint		TLboundary = 2*width;
	uint		BRboundary = width;
	uint		br_boundary_to_ignore = width;
	uint		maxKernels = floor((log2(width))*(log2(width)))*3;	// 3 is the number of channels (Y, Cb, Cr)


	HashingSchemeParams	hashingSchemeParams;
	hashingSchemeParams.width=width;
	hashingSchemeParams.dist = distFromIdentity;
	hashingSchemeParams.maxKernels = maxKernels;
	hashingSchemeParams.TLboundary = TLboundary;
	hashingSchemeParams.BRboundary = BRboundary;
	hashingSchemeParams.br_boundary_to_ignore = br_boundary_to_ignore;
	hashingSchemeParams.numHashTables = iterations;
	hashingSchemeParams.numHashs = numHashs;

	hashingSchemeParams.descriptor_params.descriptor_mode = false;
	if(patch_params) {		// patch_mode
		patch_mode = true;
		hashingSchemeParams.descriptor_params.rotation_invariant = patch_params->rotation_invariant;
		hashingSchemeParams.descriptor_params.hA	= patch_params->hA + width -1; // adding width-1 to create an image that includes last patch's pixels
		hashingSchemeParams.descriptor_params.wA= patch_params->wA + width -1;
		hashingSchemeParams.descriptor_params.dA= patch_params->dA;
		hashingSchemeParams.descriptor_params.hB = patch_params->hB + width -1;
		hashingSchemeParams.descriptor_params.wB = patch_params->wB + width -1;
		hashingSchemeParams.descriptor_params.dB = patch_params->dB;
	} else {
		patch_mode = false;
		hashingSchemeParams.descriptor_params.rotation_invariant = false;
		hashingSchemeParams.descriptor_params.hA	= A.rows + width -1; // adding width-1 to create an image that includes last patch's pixels
		hashingSchemeParams.descriptor_params.wA= A.cols + width -1;
		hashingSchemeParams.descriptor_params.dA= A.depth();
		hashingSchemeParams.descriptor_params.hB = B.rows + width -1;
		hashingSchemeParams.descriptor_params.wB = B.cols + width -1;
		hashingSchemeParams.descriptor_params.dB = B.depth();
	}
	// B] MAIN CALL TO algorithm
	Mat	KNN_extraInfo;
	if(k==1) {
		return	HashingSchemeNew(A,B,k,calcBnn,hashingSchemeParams,bMask,patch_mode);
	} else {
		if(fastKNN && (k>15)) {
			// HashingSchemeNewKNN_LargeK
			throw runtime_error("HashingSchemeNewKNN_LargeK not implemented");
		} else {
			return	HashingSchemeNew(A,B,k,calcBnn,hashingSchemeParams,bMask,patch_mode);
		}
	}
	// not reached
	return make_pair(Mat(),Mat());
}

};
