#include "Header.h"




/***********************************
** �^�C���X�^���v�擾 **
************************************/
std::string get_date(){
	struct tm date;
	time_t now;
	int year, month, day;
	int hour, minute, second;

	time(&now);

	localtime_s(&date, &now);
	
	//date = localtime(&now);
	month = date.tm_mon + 1;
	day = date.tm_mday;
	hour = date.tm_hour;
	minute = date.tm_min;
	
	std::stringstream stream2;
	stream2  << month << day << hour << minute;
	return stream2.str();
}


/***********************************
** �Ή��_�ۑ� **
************************************/
void saveFile(cv::Mat K1, cv::Mat K2, cv::Mat R2, cv::Mat t2,  std::vector<cv::Point3d> worldPoints, std::vector<cv::Point2d> imagePoints1, std::vector<cv::Point2d> imagePoints2)
{
	//�^�C���X�^���v�擾

	//���ݎ����擾
	std::string filename ="../groundtruth_" + get_date() + ".xml";
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);

	write(fs, "worldPoints", worldPoints);
	write(fs, "imagePoints1", imagePoints1);
	write(fs, "imagePoints2", imagePoints2);
	write(fs, "K1", K1);
	write(fs, "K2", K2);
	write(fs, "R2", R2);
	write(fs, "t2", t2);

	std::cout << "points saved." << std::endl;
}

void loadFile(const std::string& filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode node(fs.fs, NULL);

	//read(node["points"], loadresult);

	std::cout << "file loaded." << std::endl;
}
	

int main()
{
	//************************************************
	//�J����1
	cv::Mat K1 = cv::Mat::eye(3,3,CV_64F);
	cv::Mat R1 = cv::Mat::eye(3,3,CV_64F);
	cv::Mat t1 = cv::Mat::zeros(3,1,CV_64F);
	//***�p�����[�^�ݒ�***//
	int width1 = 1600;
	int height1 = 1400;
	double fx = 1368.4;
	double fy = 1365.8;
	double cx = (double)(width1 / 2);
	double cy = (double)(height1 / 2);

	K1.at<double>(0,0) = fx;
	K1.at<double>(1,1) = fy;
	K1.at<double>(0,2) = cx;
	K1.at<double>(1,2) = cy;

	//�J����2
	cv::Mat K2 = cv::Mat::eye(3,3,CV_64F);
	cv::Mat R2 = cv::Mat::eye(3,3,CV_64F);
	cv::Mat t2;
	//***�p�����[�^�ݒ�***//
	int width2 = 1440;
	int height2 = 900;
	fx = 1994.9;
	fy = 1995.3;
	cx = (double)(width2 / 2);
	cy = (double)(height2 / 2);

	double rx = 0.0;
	double ry = 0.0;
	double rz = 0.0;
	double tx = 0.0;
	double ty = 0.0;
	double tz = 0.0;

	cv::Mat rod = (cv::Mat_<double>(3, 1) << rx, ry, rz);
	cv::Rodrigues(rod, R2); //���h���Q�X
	t2 = (cv::Mat_<double>(3, 1) << tx, ty, tz);

	K2.at<double>(0,0) = fx;
	K2.at<double>(1,1) = fy;
	K2.at<double>(0,2) = cx;
	K2.at<double>(1,2) = cy;

	//************************************************

	int npoints = 100; //�T���v���_�̐�
	std::vector<cv::Point3d> worldPoints; //�Ή��_��3�������W
	std::vector<cv::Point2d> imagePoints1; //�J����1�摜���W�ւ̎ˉe�_
	std::vector<cv::Point2d> imagePoints2; //�J����2�摜���W�ւ̎ˉe�_
	std::vector<cv::Point2d> imagePoints1_cv; //�J����1�摜�ւ̎ˉe�_
	std::vector<cv::Point2d> imagePoints2_cv; //�J����2�摜�ւ̎ˉe�_

	double thresh = 2.0; //�����_�̂������l

	//��������
    std::random_device rnd;     // �񌈒�I�ȗ���������𐶐�  
	std::mt19937_64 mt(rnd());     //  �����Z���k�E�c�C�X�^��32�r�b�g�ŁA�����͏����V�[�h�l
    std::uniform_real_distribution<> rand_x(-thresh, thresh);        // [0, 99] �͈͂̈�l����
	std::uniform_real_distribution<> rand_y(-thresh, thresh);        // [0, 99] �͈͂̈�l����
	std::uniform_real_distribution<> rand_z(0, thresh);        // [0, 99] �͈͂̈�l����

	//CV�摜��ւ̕ϊ��s��
	cv::Mat M1 = (cv::Mat_<double>(3, 3) << 1, 0, width1/2, 0, -1, height1/2, 0, 0, 1);
	cv::Mat M2 = (cv::Mat_<double>(3, 3) << 1, 0, width2/2, 0, -1, height2/2, 0, 0, 1);

	//3�����_�̐���
	while(worldPoints.size() < npoints)
	{
		cv::Point3d wp((double)rand_x(mt), (double)rand_y(mt), (double)rand_z(mt)); 
		cv::Point2d imagept1, imagept2, imagept1_cv, imagept2_cv;

		//�ˉe
		cv::Mat _wp =  (cv::Mat_<double>(4, 1) << wp.x, wp.y, wp.z, 1.0);
		cv::Mat Rt1 =  (cv::Mat_<double>(3, 4) << R1.at<double>(0,0), R1.at<double>(0,1), R1.at<double>(0,2), t1.at<double>(0,0),
																		  R1.at<double>(1,0), R1.at<double>(1,1), R1.at<double>(1,2), t1.at<double>(1,0),
																		  R1.at<double>(2,0), R1.at<double>(2,1), R1.at<double>(2,2), t1.at<double>(2,0));
		cv::Mat Rt2 =  (cv::Mat_<double>(3, 4) << R2.at<double>(0,0), R2.at<double>(0,1), R2.at<double>(0,2), t2.at<double>(0,0),
																		  R2.at<double>(1,0), R2.at<double>(1,1), R2.at<double>(1,2), t2.at<double>(1,0),
																		  R2.at<double>(2,0), R2.at<double>(2,1), R2.at<double>(2,2), t2.at<double>(2,0));
		cv::Mat pt1 = K1 * Rt1 * _wp;
		cv::Mat pt2 = K2 * Rt2 * _wp;

		imagept1 = cv::Point2d(pt1.at<double>(0,0) / pt1.at<double>(2,0), pt1.at<double>(1,0) / pt1.at<double>(2,0)); 
		imagept2 = cv::Point2d(pt2.at<double>(0,0) / pt2.at<double>(2,0), pt2.at<double>(1,0) / pt2.at<double>(2,0)); 

		cv::Mat ip1 = (cv::Mat_<double>(3, 1) << imagept1.x, imagept1.y, 1);
		cv::Mat ip2 = (cv::Mat_<double>(3, 1) << imagept2.x, imagept2.y, 1);
		cv::Mat ip1_ = M1 * ip1;
		cv::Mat ip2_  = M2 * ip2;
		imagept1_cv = cv::Point2d(ip1_.at<double>(0,0), ip1_.at<double>(1,0));
		imagept2_cv = cv::Point2d(ip2_.at<double>(0,0), ip2_.at<double>(1,0));

		//if(-width1/2 <= imagept1.x && imagept1.x <= width1/2 && -height1/2 <= imagept1.y && imagept1.y <= height1/2 &&
		//	-width2/2 <= imagept2.x && imagept2.x <= width2/2 && -height2/2 <= imagept2.y && imagept2.y <= height2/2)	
		if(0 <= imagept1_cv.x && imagept1_cv.x <= width1 && 0 <= imagept1_cv.y && imagept1_cv.y <= height1 &&
			0 <= imagept2_cv.x && imagept2_cv.x <= width2 && 0 <= imagept2_cv.y && imagept2_cv.y <= height2)	
		{
			worldPoints.push_back(wp);
			imagePoints1.push_back(imagept1);
			imagePoints2.push_back(imagept2);
			imagePoints1_cv.push_back(imagept1_cv);
			imagePoints2_cv.push_back(imagept2_cv);
		}
	}
	//************************************************

	//�t�@�C���ɏo��
	saveFile(K1, K2, R2, t2, worldPoints, imagePoints1, imagePoints2);

	//����
	cv::Mat image1(cv::Size(width1, height1), CV_8UC3, cv::Scalar::all(255));
	cv::Mat image2(cv::Size(width2, height2), CV_8UC3, cv::Scalar::all(255));

	for(int i = 0; i < worldPoints.size(); i++)
	{
		cv::circle(image1, imagePoints1_cv[i], 1.0, cv::Scalar(0, 0, 255), 2);
		cv::circle(image2, imagePoints2_cv[i], 1.0, cv::Scalar(255, 0, 0), 2);
	}

	cv::Mat resize1(cv::Size(width1/2, height1/2), CV_8UC3);
	cv::Mat resize2(cv::Size(width2/2, height2/2), CV_8UC3);
	cv::resize(image1, resize1, cv::Size(width1/2, height1/2));
	cv::resize(image2, resize2, cv::Size(width2/2, height2/2));

	cv::imshow("image1", resize1);
	cv::imshow("image2", resize2);


	std::cout << "�I�����܂���. �����L�[�������Ă�������..." << std::endl;

	cv::waitKey(0);

	return 0;
}