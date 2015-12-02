#include "Header.h"




/***********************************
** タイムスタンプ取得 **
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
** 対応点保存 **
************************************/
void saveFile(cv::Mat K1, cv::Mat K2, cv::Mat R2, cv::Mat t2,  std::vector<cv::Point3d> worldPoints, std::vector<cv::Point2d> imagePoints1, std::vector<cv::Point2d> imagePoints2)
{
	//タイムスタンプ取得

	//現在時刻取得
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
	//カメラ1
	cv::Mat K1 = cv::Mat::eye(3,3,CV_64F);
	cv::Mat R1 = cv::Mat::eye(3,3,CV_64F);
	cv::Mat t1 = cv::Mat::zeros(3,1,CV_64F);
	//***パラメータ設定***//
	int width1 = 640;
	int height1 = 480;
	double fx = 310.0f;
	double fy = 310.0f;
	double cx = (double)(width1 / 2);
	double cy = (double)(height1 / 2);

	K1.at<double>(0,0) = fx;
	K1.at<double>(1,1) = fy;
	K1.at<double>(0,2) = cx;
	K1.at<double>(1,2) = cy;

	//カメラ2
	cv::Mat K2 = cv::Mat::eye(3,3,CV_64F);
	cv::Mat R2 = cv::Mat::eye(3,3,CV_64F);
	cv::Mat t2;
	//***パラメータ設定***//
	int width2 = 1440;
	int height2 = 900;
	fx = 1995.0f;
	fy = 1995.0f;
	cx = (double)(width2 / 2);
	cy = (double)(height2 / 2);

	double rx = 0.0;
	double ry = 0.0;
	double rz = 0.0;
	double tx = 0.4;
	double ty = 0.5;
	double tz = 0.6;

	cv::Mat rod = (cv::Mat_<double>(3, 1) << rx, ry, rz);
	cv::Rodrigues(rod, R2); //ロドリゲス
	t2 = (cv::Mat_<double>(3, 1) << tx, ty, tz);

	K2.at<double>(0,0) = fx;
	K2.at<double>(1,1) = fy;
	K2.at<double>(0,2) = cx;
	K2.at<double>(1,2) = cy;

	//************************************************

	int npoints = 50; //サンプル点の数
	std::vector<cv::Point3d> worldPoints; //対応点の3次元座標
	std::vector<cv::Point2d> imagePoints1; //カメラ1画像への射影点
	std::vector<cv::Point2d> imagePoints2; //カメラ2画像への射影点

	double thresh = 2.0; //生成点のしきい値

	//乱数生成
    std::random_device rnd;     // 非決定的な乱数生成器を生成  
	std::mt19937_64 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
    std::uniform_real_distribution<> rand_x(-thresh, thresh);        // [0, 99] 範囲の一様乱数
    std::uniform_real_distribution<> rand_y(-thresh, thresh);        // [0, 99] 範囲の一様乱数
    std::uniform_real_distribution<> rand_z(0, thresh);        // [0, 99] 範囲の一様乱数

	//3次元点の生成
	while(worldPoints.size() < npoints)
	{
		cv::Point3d wp((double)rand_x(mt), (double)rand_y(mt), (double)rand_z(mt)); 
		cv::Point2d imagept1, imagept2;

		//射影
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

		if(0 <= imagept1.x && imagept1.x <= width1 && 0 <= imagept1.y && imagept1.y <= height1 &&
			0 <= imagept2.x && imagept2.x <= width2 && 0 <= imagept2.y && imagept2.y <= height2)	
		{
			worldPoints.push_back(wp);
			imagePoints1.push_back(imagept1);
			imagePoints2.push_back(imagept2);
		}
	}
	//************************************************

	//ファイルに出力
	saveFile(K1, K2, R2, t2, worldPoints, imagePoints1, imagePoints2);

	std::cout << "終了しました. 何かキーを押してください..." << std::endl;

	cv::waitKey(0);

	return 0;
}