#ifndef CSH_H
#define CSH_H
#include <utility>
#include <opencv2/opencv.hpp>

namespace CSH {
	
class PatchParams
{
public:
	bool		rotation_invariant;
	int		hA,wA,dA;
	int		hB,wB,dB;
};

/**
 * \fn [CSH_ann,CSH_bnn] = CSH_nn(A,B,... % basic params - can handle only these (rest can be default)
 * width,iterations,k,calcBnn,bMask,patch_mode,distFromIdentity,patch_params,fastKNN) % additional important params
 *
 * \brief CSH_NN - Coherence Sensitive Hashing algorithm.
 * to compute the (dense) nearest neighbor field between images A and B
 * usage: nnf = CSH_nn(A,B,[width=8],[iterations=5],[k=1],[width=8],[calcBnn=0],[bMask=[]])
 *
 * \param A - an RGB image, the source of the NN field.
 * \param B - an RGB image, the destination of the NN field.
 * \param width -  the dimension [in pixels] of a  patch. Supported values are [2,4,8,16,32]
 * \param iterations - iterations algorithm.
 * \param k - as in k-nearest-neighbors, default is 1
 * \param calcBnn - calculated bnn as well as ann
 * \param bMask - mask on the target image, where to take the candidates from: 0 = use patch and 1 = hole.
 *
 * \return 
 * 1] CSH_ann - the nearest neighbor field from A to B, which is of size (hA,wA,2) where
 *                       * [hA,wA] are the height and width of A
 *                       * ann(i,j,1) and ann(i,j,2) are the (top-left-corner) [x,y] indices in B that the patch
 *                          in A with top-left-corner at [j,i] is mapped to
 * 2] CSH_bnn - same, but in the other direction
 * this is for CSH that keeps 'distFromIdentity' pixels in x or y away from identity
*/
std::pair<cv::Mat,cv::Mat>	nn(cv::Mat A,cv::Mat B,short width=8,short iterations=5,short k=1,bool calcBnn=false,cv::Mat bMask=cv::Mat(),short distFromIdentity=0,PatchParams* patch_params=nullptr,bool fastKNN=false);

	
};

#endif
