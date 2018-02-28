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

int main()
{
	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	int DescriptorDim = 0;
	//SVM������
	MySVM svm;

	//��TRAINΪtrue������ѵ��������
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

			//�����һ������ʱ��ʼ���������������������
			if (0 == num)
			{
				//HOG�����ӵ�ά��
				DescriptorDim = descriptors.size();
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

				//��HOG�����Ӹ��Ƶ�������������
				for (int i = 0; i < DescriptorDim; i++){
					sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];
				}
				//���������Ϊ-1������
				sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;
			}
		}

		// ���������HOG�������������ļ�
		/*	ofstream fout("SampleFeatureMat.txt");
			for(int i=0; i<PosSamNO+NegSamNO; i++)
			{
			fout<<i<<endl;
			for(int j=0; j<DescriptorDim; j++)
			{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";

			}
			fout<<endl;
			}*/

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

	/*************************************************************************************************
	  ����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	  ��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
	  ��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	  �Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	  ***************************************************************************************************/
	
	// ����������ά������HOG�����ӵ�ά��
	DescriptorDim = svm.get_var_count();
	// ֧�������ĸ���
	int supportVectorNum = svm.get_support_vector_count();
	cout << "֧������������" << supportVectorNum << endl;
	// alpha���������ȵ���֧����������
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
	// ֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);
	// alpha��������֧����������Ľ��
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);

	// ��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i < supportVectorNum; i++)
	{
		// ���ص�i��֧������������ָ��
		const float * pSVData = svm.get_support_vector(i);
		for (int j = 0; j < DescriptorDim; j++)
		{
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	// ��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();
	for (int i = 0; i < supportVectorNum; i++){
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i < DescriptorDim; i++){
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	ofstream fout("HOGDetectorForOpenCV.txt");
	for (int i = 0; i < myDetector.size(); i++){
		fout << myDetector[i] << endl;
	}

	/**************����ͼƬ����HOG���˼��******************/
	Mat src = imread(TestImageFileName);
	
	//���ο�����
	vector<Rect> found, found_filtered;
	cout << "���ж�߶�HOG������" << endl;

	//��ͼƬ���ж�߶����˼��
	myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	//srcΪ���������ͼƬ��foundΪ��⵽Ŀ�������б�����3Ϊ�����ڲ�����Ϊ����Ŀ�����ֵ��Ҳ���Ǽ�⵽��������SVM���೬ƽ��ľ���;
	//����4Ϊ��������ÿ���ƶ��ľ��롣�������ǿ��ƶ���������������5Ϊͼ������Ĵ�С������6Ϊ����ϵ����������ͼƬÿ�γߴ��������ӵı�����
	//����7Ϊ����ֵ����У��ϵ������һ��Ŀ�걻������ڼ�����ʱ���ò�����ʱ�����˵������ã�Ϊ0ʱ��ʾ����������á�

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
	for (int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}
	cout << "�ҵ��ľ��ο������" << found_filtered.size() << endl;

	// �����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
	for (int i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
	}

	imwrite("ImgProcessed.jpg", src);
	imshow("src", src);
	waitKey(0);


	/******************���뵥��64*128�Ĳ���ͼ������HOG�����ӽ��з���*********************/
	////��ȡ����ͼƬ(64*128��С)����������HOG������
	//Mat testImg = imread("person014142.jpg");
	//Mat testImg = imread("noperson000026.jpg");
	//vector<float> descriptor;
	//hog.compute(testImg,descriptor,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
	//Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//����������������������
	//������õ�HOG�����Ӹ��Ƶ�testFeatureMat������
	//for(int i=0; i<descriptor.size(); i++)
	//	testFeatureMat.at<float>(0,i) = descriptor[i];

	//��ѵ���õ�SVM�������Բ���ͼƬ�������������з���
	//int result = svm.predict(testFeatureMat);//�������
	//cout<<"��������"<<result<<endl;

	return 0;
}
