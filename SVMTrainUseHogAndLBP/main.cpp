#include <iostream>
#include <fstream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;

#include "dataset.h" // 定义一些数据
#include "my_svm.h"  // MySVM继承自CvSVM的类
#include "LBP.h"

int main()
{
	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	int DescriptorDim = 0;
	//SVM分类器
	MySVM svm;

	// 训练分类器
	if (TRAIN)
	{
		// 图片名(绝对路径)
		string ImgName;
		// 正、负样本图片的文件名列表
		ifstream finPos(PosSamListFile);
		ifstream finNeg(NegSamListFile);

		// 所有训练样本的特征向量组成的矩阵(行数:样本数，列数:HOG描述子维数)
		Mat sampleFeatureMat;
		// 训练样本的类别向量(行数:样本数，列数为1；1表示有人，-1表示无人)
		Mat sampleLabelMat;

		// 依次读取正样本图片，生成HOG描述子
		for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			//加上正样本的路径名
			ImgName = "../dataset/pos/" + ImgName;
			//读取图片
			Mat src = imread(ImgName);
			if (CENTRAL_CROP){
				if (src.cols >= 96 && src.rows >= 160){
					//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
					src = src(Rect(16, 16, 64, 128));
				}
			}
			// HOG描述子向量
			vector<float> descriptors;
			// 计算HOG描述子，检测窗口移动步长(8,8)
			// 实际上由于检测窗口和图像尺寸相等，不需要移动
			hog.compute(src, descriptors, Size(8, 8));

			// LBP描述子向量
			vector<float> LBPdescriptors;
			// 计算LBP描述子，检测窗口移动步长(8,8)
			// 实际上由于检测窗口和图像尺寸相等，不需要移动
			LBP(src, LBPdescriptors);
			descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());


			//处理第一个样本时初始化特征向量矩阵和类别矩阵
			if (0 == num)
			{
				//HOG描述子的维数
				DescriptorDim = static_cast<int>(descriptors.size());
				//初始化所有训练样本的特征向量组成的矩阵(行数:样本数，列数:HOG描述子维数)
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//初始化训练样本的类别向量矩阵(行数:样本数，列数为1；1表示有人，-1表示无人)
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			// 将HOG描述子复制到样本特征矩阵
			for (int i = 0; i < DescriptorDim; i++){
				sampleFeatureMat.at<float>(num, i) = descriptors[i];
			}
			// 正样本类别为1，有人
			sampleLabelMat.at<float>(num, 0) = 1;
		}

		//依次读取负样本图片，生成HOG描述子
		for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "处理：" << ImgName << endl;
			// 加上负样本的路径名
			ImgName = "../dataset/neg/" + ImgName;
			// 读取图片
			Mat src = imread(ImgName);

			//HOG描述子向量
			vector<float> descriptors;
			//计算HOG描述子，检测窗口移动步长(8,8)
			hog.compute(src, descriptors, Size(8, 8));
			
			// LBP描述子向量
			vector<float> LBPdescriptors;
			// 计算LBP描述子，检测窗口移动步长(8,8)
			// 实际上由于检测窗口和图像尺寸相等，不需要移动
			LBP(src, LBPdescriptors);
			descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());

			// 将HOG描述子复制到样本特征矩阵
			for (int i = 0; i < DescriptorDim; i++){
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];
			}
			//负样本类别为-1，无人
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;
		}

		//处理HardExample负样本
		if (HardExampleNO > 0)
		{
			//HardExample负样本的文件名列表
			ifstream finHardExample(HardExampleListFile);
			//依次读取HardExample负样本图片，生成HOG描述子
			for (int num = 0; num < HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "处理：" << ImgName << endl;
				ImgName = "../dataset/HardExample/" + ImgName;
				Mat src = imread(ImgName);

				//HOG描述子向量
				vector<float> descriptors;
				//计算HOG描述子，检测窗口移动步长(8,8)
				hog.compute(src, descriptors, Size(8, 8));

				// LBP描述子向量
				vector<float> LBPdescriptors;
				// 计算LBP描述子，检测窗口移动步长(8,8)
				// 实际上由于检测窗口和图像尺寸相等，不需要移动
				LBP(src, LBPdescriptors);
				descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());

				//将HOG描述子复制到样本特征矩阵
				for (int i = 0; i < DescriptorDim; i++){
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];
				}
				//负样本类别为-1，无人
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;
			}
		}

		// 训练SVM分类器
		// 迭代终止条件，当迭代满TermCriteriaCount次或误差小于FLT_EPSILON时停止迭代
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		// SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR,0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "开始训练SVM分类器" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);
		cout << "训练完成" << endl;
		svm.save("SVM_HOG.xml");//将训练好的SVM模型保存为xml文件
	}
	else{ //若TRAIN为false，从XML文件读取训练好的分类器
		svm.load("SVM_HOG.xml");
	}

	//测试样本的特征向量矩阵
	
	ifstream finTestNeg(NegTestListFile, ios::in);
	string testImgName;
	int falseNegNum = 0;
	int falsePosNum = 0;

	//依次读取负样本图片，生成HOG描述子
	for (int num = 0; num < NegTestNO && getline(finTestNeg, testImgName); num++)
	{
		cout << "处理：" << testImgName << endl;
		// 加上负样本的路径名
		testImgName = "../dataset/test/neg/" + testImgName;
		// 读取图片
		Mat src = imread(testImgName);

		//HOG描述子向量
		vector<float> descriptors;
		//计算HOG描述子，检测窗口移动步长(8,8)
		hog.compute(src, descriptors, Size(8, 8));

		// LBP描述子向量
		vector<float> LBPdescriptors;
		// 计算LBP描述子，检测窗口移动步长(8,8)
		// 实际上由于检测窗口和图像尺寸相等，不需要移动
		LBP(src, LBPdescriptors);
		descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());

		Mat testFeatureMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
		//将计算好的HOG描述子复制到testFeatureMat矩阵中
		memcpy(testFeatureMat.ptr<float>(0), descriptors.data(), descriptors.size() * 4);
		// 用训练好的SVM分类器对测试图片的特征向量进行分类,返回类标
		int result = svm.predict(testFeatureMat);
		falsePosNum += (result == 1) ? 1 : 0;
	}
	ifstream finTestPos(PosTestListFile, ios::in);
	//依次读取正样本图片，生成HOG描述子
	for (int num = 0; num < PosTestNO && getline(finTestPos, testImgName); num++)
	{
		cout << "处理：" << testImgName << endl;
		// 加上负样本的路径名
		testImgName = "../dataset/test/pos/" + testImgName;
		// 读取图片
		Mat src = imread(testImgName);
		src = src(Rect(3, 3, 64, 128));
		//HOG描述子向量
		vector<float> descriptors;
		//计算HOG描述子，检测窗口移动步长(8,8)
		hog.compute(src, descriptors, Size(8, 8));

		// LBP描述子向量
		vector<float> LBPdescriptors;
		// 计算LBP描述子，检测窗口移动步长(8,8)
		// 实际上由于检测窗口和图像尺寸相等，不需要移动
		LBP(src, LBPdescriptors);
		descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());

		Mat testFeatureMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
		//将计算好的HOG描述子复制到testFeatureMat矩阵中
		memcpy(testFeatureMat.ptr<float>(0), descriptors.data(), descriptors.size() * 4);
		// 用训练好的SVM分类器对测试图片的特征向量进行分类,返回类标
		int result = svm.predict(testFeatureMat);
		falseNegNum += (result == -1) ? 1 : 0;
	}

	double falseNegRate = falseNegNum / static_cast<double>(PosTestNO);
	double falsePosRate = falsePosNum / static_cast<double>(NegTestNO);
	cout << "falseNegRate:" << falseNegRate << endl
		<< "falsePosRate:" << falsePosRate << endl;
	system("pause");
	return 0;
}
