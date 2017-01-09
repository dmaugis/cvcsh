#ifndef UTILS_H
#define UTILS_H

inline tuple<int,int,int> size(Mat& A)
{
	return make_tuple(A.rows,A.cols,A.depth());
}

#endif
