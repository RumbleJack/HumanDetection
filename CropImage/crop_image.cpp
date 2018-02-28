#include <iostream>
#include <iostream>
#include <fstream>
#include <stdlib.h> //srand()��rand()����
#include <time.h> //time()����
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

// ԭʼ������ͼƬ�ļ��б�
#define INRIANegativeImageList "TestExample_FromINRIA_NegList.txt" 

int CropImageCount = 0; //�ü������ĸ�����ͼƬ����

int main()
{
	char saveName[256];//�ü������ĸ�����ͼƬ�ļ���
	ifstream fin(INRIANegativeImageList);//��ԭʼ������ͼƬ�ļ��б�
	//ifstream fin("subset.txt");

	//һ��һ�ж�ȡ�ļ��б�
	Mat src;
	string ImgName;
	while (getline(fin, ImgName))
	{
		cout << "����:"<< ImgName << endl;
		ImgName = "../INRIAPerson/" + ImgName;

		src = imread(ImgName, 1);//��ȡͼƬ

		//ͼƬ��СӦ���������ٰ���һ��64*128�Ĵ���
		if (src.cols >= 64 && src.rows >= 128)
		{
			srand(time(NULL));//�������������

			//��ÿ��ͼƬ������ü�10��64*128��С�Ĳ������˵ĸ�����
			for (int i = 0; i<10; i++)
			{
				int x = (rand() % (src.cols - 64)); //���Ͻ�x����
				int y = (rand() % (src.rows - 128)); //���Ͻ�y����
													 //cout<<x<<","<<y<<endl;
				Mat imgROI = src(Rect(x, y, 64, 128));
				sprintf(saveName, "../dataset/test/neg/noperson%06d.jpg", ++CropImageCount);//���ɲü����ĸ�����ͼƬ���ļ���
				imwrite(saveName, imgROI);//�����ļ�
			}
		}
	}
	cout << "�ܹ��ü���" << CropImageCount << "��ͼƬ" << endl;
	system("pause");
}
