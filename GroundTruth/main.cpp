#include "Header.h"

#include <Eigen/Dense>
#include "unsupported/Eigen/NonLinearOptimization"
#include "unsupported/Eigen/NumericalDiff"

using namespace Eigen;

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
	

/***********************************
** Eigenを用いたLM法 **
************************************/
// Generic functor
template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct Functor
{
	typedef _Scalar Scalar;
	enum {
	InputsAtCompileTime = NX,
	ValuesAtCompileTime = NY
	};
	typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
	typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
	typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;
};

struct misra1a_functor : Functor<double>
{
	// 目的関数
	misra1a_functor(int inputs, int values, std::vector<cv::Point2d>& proj_p, std::vector<cv::Point2d>& cam_p, cv::Mat& cam_K, cv::Mat& proj_K) 
		: inputs_(inputs), values_(values), proj_p_(proj_p), cam_p_(cam_p), cam_K_(cam_K), proj_K_(proj_K), projK_inv_t(proj_K_.inv().t()), camK_inv(cam_K.inv()) {}
    
	std::vector<cv::Point2d> proj_p_;
	std::vector<cv::Point2d> cam_p_;
	const cv::Mat cam_K_;
	const cv::Mat proj_K_;
	const cv::Mat projK_inv_t;
	const cv::Mat camK_inv;

	int operator()(const VectorXd& _Rt, VectorXd& fvec) const
	{
		//回転ベクトルから回転行列にする
		cv::Mat rotateVec = (cv::Mat_<double>(3, 1) << _Rt[0], _Rt[1], _Rt[2]);
		cv::Mat R(3, 3, CV_64F, cv::Scalar::all(0));
		Rodrigues(rotateVec, R);
		//cv::Mat R = (cv::Mat_<double>(3, 3) << _Rt[0], _Rt[1], _Rt[2], _Rt[3], _Rt[4], _Rt[5], _Rt[6], _Rt[7], _Rt[8]);
		//[t]x
		cv::Mat tx = (cv::Mat_<double>(3, 3) << 0, -_Rt[5], _Rt[4], _Rt[5], 0, -_Rt[3], -_Rt[4], _Rt[3], 0);
		//cv::Mat tx = (cv::Mat_<double>(3, 3) << 0, -_Rt[11], _Rt[10], _Rt[11], 0, -_Rt[9], -_Rt[10], _Rt[9], 0);
		for (int i = 0; i < values_; ++i) {
			cv::Mat cp = (cv::Mat_<double>(3, 1) << (double)cam_p_.at(i).x,  (double)cam_p_.at(i).y,  1);
			cv::Mat pp = (cv::Mat_<double>(3, 1) << (double)proj_p_.at(i).x,  (double)proj_p_.at(i).y,  1);
			cv::Mat error = pp.t() * projK_inv_t * tx * R * camK_inv * cp;
			fvec[i] = error.at<double>(0, 0);
			//std::cout << "fvec[" << i << "] = " << fvec[i] << std::endl; 
		}
		return 0;
	}

	//Rの自由度を9にする
	//int operator()(const VectorXd& _Rt, VectorXd& fvec) const
	//{
	//	//cv::Mat R = (cv::Mat_<double>(3, 3) << _Rt[0], _Rt[1], _Rt[2], _Rt[3], _Rt[4], _Rt[5], _Rt[6], _Rt[7], _Rt[8]);
	//	////[t]x
	//	//cv::Mat tx = (cv::Mat_<double>(3, 3) << 0, -_Rt[5], _Rt[4], _Rt[5], 0, -_Rt[3], -_Rt[4], _Rt[3], 0);
	//	for (int i = 0; i < values_; ++i) {
	//		//cv::Mat cp = (cv::Mat_<double>(3, 1) << (double)cam_p_.at(i).x,  (double)cam_p_.at(i).y,  1);
	//		//cv::Mat pp = (cv::Mat_<double>(3, 1) << (double)proj_p_.at(i).x,  (double)proj_p_.at(i).y,  1);
	//		//cv::Mat error = pp.t() * projK_inv_t * tx * R * camK_inv * cp;
	//		//fvec[i] = error.at<double>(0, 0);
	//		//直に計算
	//		fvec[i] = (double)proj_p_.at(i).x * (projK_inv_t.at<double>(0, 0) * (-_Rt[11] * (_Rt[3] * (camK_inv.at<double>(0, 0) * cam_p_.at(i).x + camK_inv.at<double>(0, 1) * cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[4] * (camK_inv.at<double>(1, 0) * cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[5] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[7] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[8] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2)))) + projK_inv_t.at<double>(0, 1) * (_Rt[11] * (_Rt[0] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[1] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[2] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[7] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[8] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2)))) + projK_inv_t.at<double>(0, 2) * (-_Rt[10] * (_Rt[0] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[1] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[2] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[4] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[5] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))))) +
	//				  (double)proj_p_.at(i).y * (projK_inv_t.at<double>(1, 0) * (-_Rt[11] * (_Rt[3] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[4] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[5] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[7] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[8] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2)))) + projK_inv_t.at<double>(1, 1) * (_Rt[11] * (_Rt[0] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[1] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[2] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[7] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[8] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2)))) + projK_inv_t.at<double>(1, 2) * (-_Rt[10] * (_Rt[0] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[1] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[2] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[4] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[5] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))))) +
	//				                    projK_inv_t.at<double>(2, 0) * (-_Rt[11] * (_Rt[3] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[4] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[5] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) + _Rt[10] * (_Rt[6] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[7] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[8] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2)))) + projK_inv_t.at<double>(2, 1) * (_Rt[11] * (_Rt[0] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[1] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[2] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) - _Rt[9] * (_Rt[6] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[7] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[8] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2)))) + projK_inv_t.at<double>(2, 2) * (-_Rt[10] * (_Rt[0] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[1] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[2] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))) + _Rt[9] * (_Rt[3] * (camK_inv.at<double>(0, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(0, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(0, 2)) + _Rt[4] * (camK_inv.at<double>(1, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(1, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(1, 2)) + _Rt[5] * (camK_inv.at<double>(2, 0) * (double)cam_p_.at(i).x + camK_inv.at<double>(2, 1) * (double)cam_p_.at(i).y + camK_inv.at<double>(2, 2))));
	//		//std::cout << "fvec[" << i << "] = " << fvec[i] << std::endl; 
	//	}
	//	return 0;
	//}

	const int inputs_;
	const int values_;
	int inputs() const { return inputs_; }
	int values() const { return values_; }
};

//計算部分
void calcProjectorPose(std::vector<cv::Point2d> imagePoints, std::vector<cv::Point2d> projPoints, cv::Mat K1, cv::Mat K2, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
{
	//回転行列から回転ベクトルにする
	cv::Mat rotateVec(3, 1,  CV_64F, cv::Scalar::all(0));
	Rodrigues(initialR, rotateVec);

	int n = 6; //変数の数
	int info;
	VectorXd initial(n);
	initial <<
		rotateVec.at<double>(0, 0),
		rotateVec.at<double>(1, 0),
		rotateVec.at<double>(2, 0),
		initialT.at<double>(0, 0),
		initialT.at<double>(1, 0),
		initialT.at<double>(2, 0);

	misra1a_functor functor(n, imagePoints.size(), projPoints, imagePoints, K1, K2);
    
	NumericalDiff<misra1a_functor> numDiff(functor);
	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
	info = lm.minimize(initial);
    
	std::cout << "学習結果: " << std::endl;
	std::cout <<
		initial[0] << " " <<
		initial[1] << " " <<
		initial[2] << " " <<
		initial[3] << " " <<
		initial[4] << " " <<
		initial[5]	 << std::endl;

	//出力
	cv::Mat dstRVec = (cv::Mat_<double>(3, 1) << initial[0], initial[1], initial[2]);
	Rodrigues(dstRVec, dstR);
	dstT = (cv::Mat_<double>(3, 1) << initial[3], initial[4], initial[5]);
}

//計算部分(Rの自由度9)
void calcProjectorPose2(std::vector<cv::Point2d> imagePoints, std::vector<cv::Point2d> projPoints, cv::Mat K1, cv::Mat K2, cv::Mat initialR, cv::Mat initialT, cv::Mat& dstR, cv::Mat& dstT)
{

	int n = 12; //変数の数
	int info;
		
	VectorXd initial(n);
	initial <<
		initialR.at<double>(0, 0),
		initialR.at<double>(0, 1),
		initialR.at<double>(0, 2),
		initialR.at<double>(1, 0),
		initialR.at<double>(1, 1),
		initialR.at<double>(1, 2),
		initialR.at<double>(2, 0),
		initialR.at<double>(2, 1),
		initialR.at<double>(2, 2),
		initialT.at<double>(0, 0),
		initialT.at<double>(1, 0),
		initialT.at<double>(2, 0);


	misra1a_functor functor(n, imagePoints.size(), projPoints, imagePoints, K1, K2);
    
	NumericalDiff<misra1a_functor> numDiff(functor);
	LevenbergMarquardt<NumericalDiff<misra1a_functor> > lm(numDiff);
	info = lm.minimize(initial);
    
	std::cout << "学習結果: " << std::endl;
	std::cout <<
		initial[0] << " " <<
		initial[1] << " " <<
		initial[2] << " " <<
		initial[3] << " " <<
		initial[4] << " " <<
		initial[5] << " " <<
		initial[6] << " " <<
		initial[7] << " " <<
		initial[8] << " " <<
		initial[9] << " " <<
		initial[10] << " " <<
		initial[11] << " " << std::endl;

	//出力
	dstR = (cv::Mat_<double>(3, 3) << initial[0], initial[1], initial[2], initial[3], initial[4], initial[5], initial[6], initial[7], initial[8]);
	dstT = (cv::Mat_<double>(3, 1) << initial[9], initial[10], initial[11]);
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
	double fx = 294.3;
	double fy = 298.1;
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
	int width2 = 1280;
	int height2 = 800;
	fx = 923.7;
	fy = 924.5;
	cx = (double)(width2 / 2);
	cy = (double)(height2 / 2);

	double rx = 30.0 * CV_PI / 180;
	double ry = 0.0;
	double rz = 30.0 * CV_PI / 180; //30°のラジアン
	double tx = 0.1;
	double ty = 0.2;
	double tz = 0.3;

	cv::Mat rod = (cv::Mat_<double>(3, 1) << rx, ry, rz);
	cv::Rodrigues(rod, R2); //ロドリゲス
	t2 = (cv::Mat_<double>(3, 1) << tx, ty, tz);

	K2.at<double>(0,0) = fx;
	K2.at<double>(1,1) = fy;
	K2.at<double>(0,2) = cx;
	K2.at<double>(1,2) = cy;

	//************************************************
	//R,tにノイズ付加
	cv::Mat R2_noise = cv::Mat::eye(3,3,CV_64F);
	cv::Mat t2_noise;

	cv::Mat rod_noise = (cv::Mat_<double>(3, 1) << rx + 0.0 * CV_PI / 180, ry + 0.0 * CV_PI / 180, rz + 0.0 * CV_PI / 180);
	cv::Rodrigues(rod_noise, R2_noise); //ロドリゲス
	t2_noise = (cv::Mat_<double>(3, 1) << tx + 0.35, ty + 0.2, tz - 0.5);

	//************************************************

	//************************************************

	int npoints = 50; //サンプル点の数
	std::vector<cv::Point3d> worldPoints; //対応点の3次元座標
	std::vector<cv::Point2d> imagePoints1; //カメラ1画像座標への射影点
	std::vector<cv::Point2d> imagePoints2; //カメラ2画像座標への射影点
	std::vector<cv::Point2d> imagePoints1_cv; //カメラ1画像への射影点
	std::vector<cv::Point2d> imagePoints2_cv; //カメラ2画像への射影点

	double thresh = 2.0; //生成点のしきい値

	//乱数生成
    std::random_device rnd;     // 非決定的な乱数生成器を生成  
	std::mt19937_64 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
    std::uniform_real_distribution<> rand_x(-thresh, thresh);        // [0, 99] 範囲の一様乱数
	std::uniform_real_distribution<> rand_y(-thresh, thresh);        // [0, 99] 範囲の一様乱数
	std::uniform_real_distribution<> rand_z(0, thresh);        // [0, 99] 範囲の一様乱数

	//CV画像上への変換行列
	cv::Mat M1 = (cv::Mat_<double>(3, 3) << 1, 0, width1/2, 0, -1, height1/2, 0, 0, 1);
	cv::Mat M2 = (cv::Mat_<double>(3, 3) << 1, 0, width2/2, 0, -1, height2/2, 0, 0, 1);

	//3次元点の生成
	while(worldPoints.size() < npoints)
	{
		cv::Point3d wp((double)rand_x(mt), (double)rand_y(mt), (double)rand_z(mt)); 
		cv::Point2d imagept1, imagept2, imagept1_cv, imagept2_cv;

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

		cv::Mat ip1 = (cv::Mat_<double>(3, 1) << imagept1.x, imagept1.y, 1);
		cv::Mat ip2 = (cv::Mat_<double>(3, 1) << imagept2.x, imagept2.y, 1);
		cv::Mat ip1_ = M1 * ip1;
		cv::Mat ip2_  = M2 * ip2;
		imagept1_cv = cv::Point2d(ip1_.at<double>(0,0), ip1_.at<double>(1,0));
		imagept2_cv = cv::Point2d(ip2_.at<double>(0,0), ip2_.at<double>(1,0));

		if(-width1/2 <= imagept1.x && imagept1.x <= width1/2 && -height1/2 <= imagept1.y && imagept1.y <= height1/2 &&
			-width2/2 <= imagept2.x && imagept2.x <= width2/2 && -height2/2 <= imagept2.y && imagept2.y <= height2/2)	
		//if(0 <= imagept1_cv.x && imagept1_cv.x <= width1 && 0 <= imagept1_cv.y && imagept1_cv.y <= height1 &&
		//	0 <= imagept2_cv.x && imagept2_cv.x <= width2 && 0 <= imagept2_cv.y && imagept2_cv.y <= height2)	
		{
			worldPoints.push_back(wp);
			imagePoints1.push_back(imagept1);
			imagePoints2.push_back(imagept2);
			imagePoints1_cv.push_back(imagept1_cv);
			imagePoints2_cv.push_back(imagept2_cv);
		}
	}
	//************************************************

	//ファイルに出力
	saveFile(K1, K2, R2, t2, worldPoints, imagePoints1, imagePoints2);

	//可視化
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

	//エピポーラ方程式が成り立つか確認
	for(int i =0; i < imagePoints1.size(); i++)
	{
		cv::Mat cp = (cv::Mat_<double>(3, 1) << (double)imagePoints1.at(i).x,  (double)imagePoints1.at(i).y,  1);
		cv::Mat pp = (cv::Mat_<double>(3, 1) << (double)imagePoints2.at(i).x,  (double)imagePoints2.at(i).y,  1);
		cv::Mat projK_inv_t = K2.inv().t();
		cv::Mat tx = (cv::Mat_<double>(3, 3) << 0, -t2.at<double>(2,0), t2.at<double>(1,0), t2.at<double>(2,0), 0, -t2.at<double>(0,0), -t2.at<double>(1,0), t2.at<double>(0,0), 0);
		cv::Mat e = pp.t() * projK_inv_t * tx * R2 * K1.inv() * cp;

		double error = e.at<double>(0, 0);

		std::cout << "e[" << i << "] = " << error << std::endl; 
	}

	//ノイズ入りから非線形最適化(LM法)で正解が求まるか
	cv::Mat dstR, dstT;
	calcProjectorPose(imagePoints1, imagePoints2, K1, K2,  R2_noise, t2_noise, dstR, dstT);

	std::cout << "初期値：" << std::endl; 
	std::cout << "R: \n" << R2_noise << "\nt: \n" << t2_noise << std::endl;
	std::cout << "収束値：" << std::endl; 
	std::cout << "R: \n" << dstR << "\nt: \n" << dstT << std::endl;
	std::cout << "正解：" << std::endl; 
	std::cout << "R: \n" << R2 << "\nt: \n" << t2 << std::endl;



	std::cout << "終了しました. 何かキーを押してください..." << std::endl;

	cv::waitKey(0);

	return 0;
}