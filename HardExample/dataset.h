#pragma once 

// ������������
#define PosSamNO 2416  
#define NegSamNO 12180  

// ��������ͼƬ���ļ����б�
#define PosSamListFile "INRIAPerson96X160PosList.txt" 
#define NegSamListFile "NoPersonFromINRIAList.txt" 

//�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
#define TRAIN true   
//true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����
#define CENTRAL_CROP true  

#define HardExampleListFile "HardExample_FromINRIA_NegList.txt"
//HardExample�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������
//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ
#define HardExampleNO 0

//������ֹ��������������50000�λ����С��FLT_EPSILONʱֹͣ����
#define TermCriteriaCount 50000  

//ѵ����ɺ����һ��ͼƬ������Ч��
#define TestImageFileName "Test.jpg"  

