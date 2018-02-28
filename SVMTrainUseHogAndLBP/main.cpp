#include <iostream>
#include <fstream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;

#include "dataset.h" // ����һЩ����
#include "my_svm.h"  // MySVM�̳���CvSVM����
#include "LBP.h"

int main()
{
	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	int DescriptorDim = 0;
	//SVM������
	MySVM svm;

	// ѵ��������
	if (TRAIN)
	{
		// ͼƬ��(����·��)
		string ImgName;
		// ����������ͼƬ���ļ����б�
		ifstream finPos(PosSamListFile);
		ifstream finNeg(NegSamListFile);

		// ����ѵ������������������ɵľ���(����:������������:HOG������ά��)
		Mat sampleFeatureMat;
		// ѵ���������������(����:������������Ϊ1��1��ʾ���ˣ�-1��ʾ����)
		Mat sampleLabelMat;

		// ���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			//������������·����
			ImgName = "../dataset/pos/" + ImgName;
			//��ȡͼƬ
			Mat src = imread(ImgName);
			if (CENTRAL_CROP){
				if (src.cols >= 96 && src.rows >= 160){
					//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
					src = src(Rect(16, 16, 64, 128));
				}
			}
			// HOG����������
			vector<float> descriptors;
			// ����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			// ʵ�������ڼ�ⴰ�ں�ͼ��ߴ���ȣ�����Ҫ�ƶ�
			hog.compute(src, descriptors, Size(8, 8));

			// LBP����������
			vector<float> LBPdescriptors;
			// ����LBP�����ӣ���ⴰ���ƶ�����(8,8)
			// ʵ�������ڼ�ⴰ�ں�ͼ��ߴ���ȣ�����Ҫ�ƶ�
			LBP(src, LBPdescriptors);
			descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());


			//�����һ������ʱ��ʼ���������������������
			if (0 == num)
			{
				//HOG�����ӵ�ά��
				DescriptorDim = static_cast<int>(descriptors.size());
				//��ʼ������ѵ������������������ɵľ���(����:������������:HOG������ά��)
				sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ�������������������(����:������������Ϊ1��1��ʾ���ˣ�-1��ʾ����)
				sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32FC1);
			}

			// ��HOG�����Ӹ��Ƶ�������������
			for (int i = 0; i < DescriptorDim; i++){
				sampleFeatureMat.at<float>(num, i) = descriptors[i];
			}
			// ���������Ϊ1������
			sampleLabelMat.at<float>(num, 0) = 1;
		}

		//���ζ�ȡ������ͼƬ������HOG������
		for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++)
		{
			cout << "����" << ImgName << endl;
			// ���ϸ�������·����
			ImgName = "../dataset/neg/" + ImgName;
			// ��ȡͼƬ
			Mat src = imread(ImgName);

			//HOG����������
			vector<float> descriptors;
			//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			hog.compute(src, descriptors, Size(8, 8));
			
			// LBP����������
			vector<float> LBPdescriptors;
			// ����LBP�����ӣ���ⴰ���ƶ�����(8,8)
			// ʵ�������ڼ�ⴰ�ں�ͼ��ߴ���ȣ�����Ҫ�ƶ�
			LBP(src, LBPdescriptors);
			descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());

			// ��HOG�����Ӹ��Ƶ�������������
			for (int i = 0; i < DescriptorDim; i++){
				sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];
			}
			//���������Ϊ-1������
			sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;
		}

		//����HardExample������
		if (HardExampleNO > 0)
		{
			//HardExample���������ļ����б�
			ifstream finHardExample(HardExampleListFile);
			//���ζ�ȡHardExample������ͼƬ������HOG������
			for (int num = 0; num < HardExampleNO && getline(finHardExample, ImgName); num++)
			{
				cout << "����" << ImgName << endl;
				ImgName = "../dataset/HardExample/" + ImgName;
				Mat src = imread(ImgName);

				//HOG����������
				vector<float> descriptors;
				//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
				hog.compute(src, descriptors, Size(8, 8));

				// LBP����������
				vector<float> LBPdescriptors;
				// ����LBP�����ӣ���ⴰ���ƶ�����(8,8)
				// ʵ�������ڼ�ⴰ�ں�ͼ��ߴ���ȣ�����Ҫ�ƶ�
				LBP(src, LBPdescriptors);
				descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());

				//��HOG�����Ӹ��Ƶ�������������
				for (int i = 0; i < DescriptorDim; i++){
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];
				}
				//���������Ϊ-1������
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;
			}
		}

		// ѵ��SVM������
		// ������ֹ��������������TermCriteriaCount�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, TermCriteriaCount, FLT_EPSILON);
		// SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR,0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout << "��ʼѵ��SVM������" << endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);
		cout << "ѵ�����" << endl;
		svm.save("SVM_HOG.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	}
	else{ //��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
		svm.load("SVM_HOG.xml");
	}

	//����������������������
	
	ifstream finTestNeg(NegTestListFile, ios::in);
	string testImgName;
	int falseNegNum = 0;
	int falsePosNum = 0;

	//���ζ�ȡ������ͼƬ������HOG������
	for (int num = 0; num < NegTestNO && getline(finTestNeg, testImgName); num++)
	{
		cout << "����" << testImgName << endl;
		// ���ϸ�������·����
		testImgName = "../dataset/test/neg/" + testImgName;
		// ��ȡͼƬ
		Mat src = imread(testImgName);

		//HOG����������
		vector<float> descriptors;
		//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		hog.compute(src, descriptors, Size(8, 8));

		// LBP����������
		vector<float> LBPdescriptors;
		// ����LBP�����ӣ���ⴰ���ƶ�����(8,8)
		// ʵ�������ڼ�ⴰ�ں�ͼ��ߴ���ȣ�����Ҫ�ƶ�
		LBP(src, LBPdescriptors);
		descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());

		Mat testFeatureMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
		//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
		memcpy(testFeatureMat.ptr<float>(0), descriptors.data(), descriptors.size() * 4);
		// ��ѵ���õ�SVM�������Բ���ͼƬ�������������з���,�������
		int result = svm.predict(testFeatureMat);
		falsePosNum += (result == 1) ? 1 : 0;
	}
	ifstream finTestPos(PosTestListFile, ios::in);
	//���ζ�ȡ������ͼƬ������HOG������
	for (int num = 0; num < PosTestNO && getline(finTestPos, testImgName); num++)
	{
		cout << "����" << testImgName << endl;
		// ���ϸ�������·����
		testImgName = "../dataset/test/pos/" + testImgName;
		// ��ȡͼƬ
		Mat src = imread(testImgName);
		src = src(Rect(3, 3, 64, 128));
		//HOG����������
		vector<float> descriptors;
		//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		hog.compute(src, descriptors, Size(8, 8));

		// LBP����������
		vector<float> LBPdescriptors;
		// ����LBP�����ӣ���ⴰ���ƶ�����(8,8)
		// ʵ�������ڼ�ⴰ�ں�ͼ��ߴ���ȣ�����Ҫ�ƶ�
		LBP(src, LBPdescriptors);
		descriptors.insert(descriptors.begin(), LBPdescriptors.begin(), LBPdescriptors.end());

		Mat testFeatureMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
		//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
		memcpy(testFeatureMat.ptr<float>(0), descriptors.data(), descriptors.size() * 4);
		// ��ѵ���õ�SVM�������Բ���ͼƬ�������������з���,�������
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
