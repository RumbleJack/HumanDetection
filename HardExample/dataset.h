#pragma once 

// 正负样本个数
#define PosSamNO 2416  
#define NegSamNO 12180  

// 正负样本图片的文件名列表
#define PosSamListFile "INRIAPerson96X160PosList.txt" 
#define NegSamListFile "NoPersonFromINRIAList.txt" 

//是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define TRAIN true   
//true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体
#define CENTRAL_CROP true  

#define HardExampleListFile "HardExample_FromINRIA_NegList.txt"
//HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。
//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值
#define HardExampleNO 0

//迭代终止条件，当迭代满50000次或误差小于FLT_EPSILON时停止迭代
#define TermCriteriaCount 50000  

//训练完成后读入一张图片来测试效果
#define TestImageFileName "Test.jpg"  

