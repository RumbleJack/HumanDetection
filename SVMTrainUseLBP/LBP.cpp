#include "LBP.h"

//ԭʼLBP  
void LBP(Mat img, vector<float> &lbpDescriptor)
{
	Mat result;
	Mat src;
	copyMakeBorder(img, src, 1, 1, 1, 1, BORDER_REPLICATE);

	result.create(img.rows, img.cols, img.type());
	result.setTo(0);

	for (int i = 1; i<src.rows - 1; i++)
	{
		for (int j = 1; j<src.cols - 1; j++)
		{
			uchar center = src.at<uchar>(i, j);
			uchar code = 0;
			code |= (src.at<uchar>(i - 1, j - 1) >= center) << 7;
			code |= (src.at<uchar>(i - 1, j) >= center) << 6;
			code |= (src.at<uchar>(i - 1, j + 1) >= center) << 5;
			code |= (src.at<uchar>(i, j + 1) >= center) << 4;
			code |= (src.at<uchar>(i + 1, j + 1) >= center) << 3;
			code |= (src.at<uchar>(i + 1, j) >= center) << 2;
			code |= (src.at<uchar>(i + 1, j - 1) >= center) << 1;
			code |= (src.at<uchar>(i, j - 1) >= center) << 0;
			result.at<uchar>(i - 1, j - 1) = code;
		}
	}
	result = result(Rect(0, 0, 64, 128));
	for (int i = 0; i < result.rows / 8 - 1; i++)
	{
		for (int j = 0; j < result.cols / 8 - 1; j++)
		{
			Mat tempMat = result(Rect(8 * j, 8 * i, 16, 16));
			vector<float> des;
			des.resize(8);
			memset(des.data(), 0, 4 * 8);

			uchar* tdata;
			for (int k = 0; k < tempMat.rows; k++)
			{
				tdata = tempMat.ptr<uchar>(k);
				for (int l = 0; l < tempMat.cols; l++)
				{
					des[tdata[l] / 32] += 1;
				}
			}
			Mat desMat(1, 8, CV_32FC1);
			memcpy(desMat.ptr<float>(0), des.data(), 8 * 4);

			normalize(desMat, desMat, 1, 0, NORM_L2);

			memcpy(des.data(), desMat.ptr<float>(0), 8 * 4);

			lbpDescriptor.insert(lbpDescriptor.begin(), des.begin(), des.end());
		}
	}
}