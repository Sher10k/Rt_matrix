#ifndef HANDLER_RT_H
#define HANDLER_RT_H

// ZCM
#include <zcm/zcm-cpp.hpp>
#include "../zcm_types/service/cpp_types/ZcmService.hpp"
#include "../zcm_types/camera_basler/cpp_types/ZcmCameraBaslerJpegFrame.hpp"
#include "../zcm_types/saugl/cpp_types/ZcmSauglState.hpp"
#include "../zcm_types/neural_rail_detector/cpp_types/ZcmRailDetectorMask.hpp"
#include "../zcm_types/lidar_scala/cpp_types/ZcmLidarScalaDataScan.hpp"

// Computer vision
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
// #include <opencv2/stereo.hpp>
#include <opencv2/imgproc.hpp>

// Eigen
#include <Eigen/Eigen>

// OPEN3D
#include <Open3D/Open3D.h>
#include <Open3D/Geometry/Geometry.h>
#include <Open3D/Geometry/Geometry3D.h>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Geometry/Octree.h>
#include <Open3D/Visualization/Visualizer/Visualizer.h>
#include <Open3D/Visualization/Visualizer/ViewControlWithEditing.h>

// STD
#include <string>
#include <iostream>
#include <ctime>

#define DEBUG

class Handler
{
public:
    struct StereoParams
        {
            int minDisp;
            int maxDisp;
    
            int blockSize;
            
            int preFilterCap;
    
            int P1;
            int P2;
    
            int speckleRange;
            int speckleWindowSize;
        };
    Handler(
        std::string& left_channel,
        std::string& right_channel,
        std::string& railway_channel,
        std::string& stereo_points_channel,
        Handler::StereoParams params,
        zcm::ZCM* zcm_out,
        int hardware_binning=1,
        int software_binning=1)
    :
        hardware_binning(hardware_binning),
        software_binning(software_binning),
        left_channel(left_channel),
        right_channel(right_channel),
        railway_channel(railway_channel),
        stereo_points_channel(stereo_points_channel),
        calibration_type(CALIBRATION_TYPE::CLASSIC),
        zcm_out(zcm_out)
    {
        bm = cv::StereoSGBM::create(params.minDisp, params.maxDisp, params.blockSize);
        bm->setPreFilterCap(params.preFilterCap);
        bm->setP1(params.P1);
        bm->setP2(params.P2);
        bm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
        bm->setSpeckleRange(params.speckleRange);
        bm->setSpeckleWindowSize(params.speckleWindowSize);
        
//        bm = cv::StereoSGBM::create(0, 96, 21);
//        bm->setPreFilterCap(1);
//        bm->setP1(96);
//        bm->setP2(2048);
//        bm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
//        bm->setSpeckleRange(20);
//        bm->setSpeckleWindowSize(2048);
    }
    
    void handleMessage(
        const zcm::ReceiveBuffer*,
        const std::string& channel,
        const ZcmCameraBaslerJpegFrame *msg)
    {
        std::cout << "Get data from channel "
                  << channel
                  << std::endl;

        std::vector< char > jpeg_buf;
        jpeg_buf.assign(
            msg->jpeg.data(),
            msg->jpeg.data() + msg->jpeg.size());
        
        ZcmSauglState zcm_rev;
        std::cout << "Revers: " << int(zcm_rev.reversor) << std::endl;

        if (channel == left_channel)
        {
            last_left_img = cv::imdecode( jpeg_buf, cv::IMREAD_COLOR );
            last_left_ts = msg->service.u_timestamp;
            std::cout << "Read left image: " << last_left_img.size() << " | " << last_left_ts << "\n";

            std::string SHIT((char*)msg->jpeg.data(), (size_t)100);
            std::string SHIT_from_vector((char*)jpeg_buf.data(), (size_t)100);
        }
        else if (channel == right_channel)
        {
            last_right_img = cv::imdecode( jpeg_buf, cv::IMREAD_COLOR );
            last_right_ts = msg->service.u_timestamp;
            std::cout << "Read right image: " << last_right_img.size() << " | " << last_right_ts << "\n";
        }

        if (abs(last_left_ts - last_right_ts) > 100 )
        {
            std::cout << "Not sync:\n"
                      << last_left_ts << "\n"
                      << last_right_ts << "\n";
            return;
        }

        std::cout << "Synchronized!" << std::endl;
        std::cout << "Image size: " << last_left_img.size() << std::endl;

        cv::Size binning_size = {
                int(last_left_img.size().width / software_binning),
                int(last_left_img.size().height / software_binning),
            };

        cv::resize(
            last_left_img, last_left_img, binning_size);
        cv::resize(
            last_right_img, last_right_img, binning_size);

        if (calibration_type == CALIBRATION_TYPE::CLASSIC)
        {
            std::cout << "Left size after binning but before rectification: " << last_left_img.size() << "\n";
            cv::remap(
                last_left_img, last_left_img,
                leftMapX, leftMapY, cv::INTER_LINEAR);
            cv::remap(
                last_right_img, last_right_img,
                rightMapX, rightMapY, cv::INTER_LINEAR);
            std::cout << "Left size after binning and rectification: " << last_left_img.size() << "\n";
        }

        cv::Mat last_left_img_gray, last_right_img_gray;
        cv::cvtColor(last_left_img, last_left_img_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(last_right_img, last_right_img_gray, cv::COLOR_BGR2GRAY);
        
        std::cout << "Disp calcul !" << std::endl;
//        cv::imshow ("LRIG", last_left_img_gray);
//        cv::imshow ("RRIG", last_right_img_gray);
//        cv::waitKey(200);
        cv::Mat disparity;
        bm->compute(last_left_img_gray, last_right_img_gray, disparity);

        float coef = last_left_img.size().width/disparity.size().width;
        // coef = 1;
        std::cout << "Disparity coefficient: " << coef << "\n";
        
            // 3D points with mask
        cv::Mat disparity_masking;
        disparity.convertTo( disparity, CV_32F );
        if ( !last_railway_mask.empty() ) disparity.copyTo( disparity_masking, last_railway_mask);
        cv::reprojectImageTo3D( disparity_masking, XYZ, Q, 3 );
        std::cout << "XYZ size: " << XYZ.size() << std::endl;

        

        #ifdef DEBUG
            cv::Mat disparity_viz;
            disparity.convertTo( disparity_viz, CV_8U, 255/(96*16.0) );
            if ( !disparity_masking.empty() ) disparity_masking.convertTo( disparity_viz, CV_8U, 255/(96*16.0) );
            cv::applyColorMap( disparity_viz, disparity_viz, cv::COLORMAP_HSV );
            cv::imshow("disp", disparity_viz);
            cv::waitKey(10);
            cv::imshow("left", last_left_img);
            cv::waitKey(10);
        #endif
    }
    
    void handleRailway(
        const zcm::ReceiveBuffer*,
        const std::string& channel,
        const ZcmRailDetectorMask *msg)
    {
        std::cout << "Get data from channel "
                  << channel
                  << std::endl;

        std::vector< char > jpeg_buf;
        jpeg_buf.assign(
            msg->mask.data(),
            msg->mask.data() + msg->mask.size());

        if (channel == railway_channel)
        {
            last_railway_mask = cv::imdecode(
                jpeg_buf, cv::IMREAD_GRAYSCALE);
        }
        last_railway_ts = msg->service.u_timestamp;
            // Resize and blur for smooth noize 
        if (!last_left_img.empty())
        {
            cv::resize( last_railway_mask, last_railway_mask, last_left_img.size() );
            cv::blur(last_railway_mask, last_railway_mask, {5, 5});
            cv::inRange(last_railway_mask, 15, 255, last_railway_mask);
        }        
        std::cout << "Read railway image: " << last_railway_mask.size() << " | " << last_railway_ts << "\n";
        std::cout << "Jpeg buf info: " << jpeg_buf.size() << " | " << msg->mask.size() << "\n";
        #ifdef DEBUG
            cv::imshow("railway_channel", last_railway_mask);
            cv::waitKey(10);
        #endif
    }
    
    void handle3Dpoints(
            const zcm::ReceiveBuffer*,
            const std::string& channel,
            const ZcmLidarScalaDataScan *msg )
    {
        std::cout << "Get data from channel "
                  << channel
                  << std::endl;
        
        if ( channel == stereo_points_channel )
        {
            Cloud->Clear();
            
            for ( size_t i = 0; i < XYZ.total(); i ++ )
            {
                Eigen::Vector3d temPoint, tempColor;
                temPoint << double( XYZ.at< cv::Vec3f >( int(i) )[0] ),
                            double( XYZ.at< cv::Vec3f >( int(i) )[1] ),
                            double( XYZ.at< cv::Vec3f >( int(i) )[2] );
                tempColor.x() = double( last_left_img.at< cv::Vec3b >(int(i)).val[2] / 255.0 ); // imgRemap[0].at< cv::Vec3b >(int(i)).val[2] / 255.0
                tempColor.y() = double( last_left_img.at< cv::Vec3b >(int(i)).val[1] / 255.0 );
                tempColor.z() = double( last_left_img.at< cv::Vec3b >(int(i)).val[0] / 255.0 );
                Cloud->points_.push_back( temPoint );
                Cloud->colors_.push_back( tempColor );
            }
            
            vis->UpdateGeometry();
        }
        //vis->PollEvents();
        vis->Run();
    }
    
    
    void set_calib(
        cv::Mat& left_H, cv::Mat& right_H, cv::Mat& Q)
    {
        this->Q = Q;
        calibration_type = CALIBRATION_TYPE::UNCALIB;
    }

    void set_calib(
        cv::Mat& leftMapX, cv::Mat& leftMapY,
        cv::Mat& rightMapX, cv::Mat& rightMapY,
        cv::Mat& Q)
    {
        cv::resize(leftMapX, leftMapX, cv::Size(0, 0), 1.0/software_binning, 1.0/software_binning);
        cv::resize(rightMapX, rightMapX, cv::Size(0, 0), 1.0/software_binning, 1.0/software_binning);
        cv::resize(leftMapY, leftMapY, cv::Size(0, 0), 1.0/software_binning, 1.0/software_binning);
        cv::resize(rightMapY, rightMapY, cv::Size(0, 0), 1.0/software_binning, 1.0/software_binning);

        this->leftMapX = leftMapX/software_binning;
        this->rightMapX = rightMapX/software_binning;
        this->leftMapY = leftMapY/software_binning;
        this->rightMapY = rightMapY/software_binning;
        this->Q = Q;

        Q.at< double >(0, 3) /= software_binning;
        Q.at< double >(1, 3) /= software_binning;
        // Q.at< double >(2, 3) /= software_binning;
        // x, y *= software_binning

        calibration_type = CALIBRATION_TYPE::CLASSIC;
    }
    
    enum CALIBRATION_TYPE
    {
        NONE,
        CLASSIC,
        UNCALIB
    };
    
private:
    int64_t last_left_ts;
    int64_t last_right_ts;
    int64_t last_railway_ts;

    cv::Mat last_left_img, last_right_img, last_railway_mask;
    int hardware_binning, software_binning;


    std::string left_channel;
    std::string right_channel;
    std::string railway_channel;
    std::string stereo_points_channel;

    cv::Ptr<cv::StereoSGBM> bm;
    cv::Mat Q;
    cv::Mat XYZ, disparity32f;

    cv::Mat leftMapX, leftMapY;
    cv::Mat rightMapX, rightMapY;
    cv::Mat left_H, right_H;
    CALIBRATION_TYPE calibration_type;

    zcm::ZCM* zcm_out;
public:
    std::shared_ptr< open3d::geometry::PointCloud > Cloud;
    open3d::visualization::Visualizer *vis;
    
};


#endif // HANDLER_RT_H
