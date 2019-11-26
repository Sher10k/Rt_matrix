#include "Header/handler_Rt.h"
#include <QCoreApplication>

#include <iostream>
#include <vector>
#include <set>
#include <iterator>
#include <math.h>
#include <thread>
//#include <chrono>
#include <mutex>
#include <atomic>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

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

#include <zcm/zcm-cpp.hpp>
#include "../zcm_types/service/cpp_types/ZcmService.hpp"
//#include "ZcmCameraBaslerJpegFrame.hpp"
//#include "ZcmSauglState.hpp"
//#include "ZcmLidarScalaPoint.hpp"
//#include "ZcmLidarScalaDataScan.hpp"

// UNIX & Linux
#include <sys/types.h>
#include <unistd.h>


using namespace std;
using namespace cv;
using namespace Eigen;
using namespace zcm;
using namespace open3d;

#define DEBUG

void PaintGrid( visualization::Visualizer &vis, double Gsize, double Gstep )
{
    double R = 0.005, H = Gsize;
    int resolution = 3, split = 2;
    for ( double i = 0; i <= Gsize; i += Gstep )
    {
            // XoZ || oX
        auto plane1 = geometry::TriangleMesh::CreateCylinder( R, H, resolution, split );
        plane1->PaintUniformColor( Vector3d( 0.0, 0.0, 0.0 ) );
        Matrix4d Rt;
        Rt << 0.0, 0.0, 1.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              1.0, 0.0, 0.0,  i,
              0.0, 0.0, 0.0, 1.0;
        plane1->Transform( Rt );
        plane1->ComputeVertexNormals();
        vis.AddGeometry( plane1 );
        
            // XoZ || oZ
        auto plane2 = geometry::TriangleMesh::CreateCylinder( R, H, resolution, split );
        plane2->PaintUniformColor( Vector3d( 0.0, 0.0, 0.0 ) );
        Rt << 1.0, 0.0, 0.0, i-Gsize/2,
              0.0, 1.0, 0.0,    0.0,
              0.0, 0.0, 1.0,    H/2,
              0.0, 0.0, 0.0,    1.0;
        plane2->Transform( Rt );
        plane2->ComputeVertexNormals();
        vis.AddGeometry( plane2 );
        
            // YoZ || oZ
        auto plane3 = geometry::TriangleMesh::CreateCylinder( R, H, resolution, split );
        plane3->PaintUniformColor( Vector3d( 0.0, 0.0, 0.0 ) );
        Rt << 1.0, 0.0, 0.0,    0.0,
              0.0, 1.0, 0.0, i-Gsize/2,
              0.0, 0.0, 1.0,    H/2,
              0.0, 0.0, 0.0,    1.0;
        plane3->Transform( Rt );
        plane3->ComputeVertexNormals();
        vis.AddGeometry( plane3 );
        
            // YoZ || oY
        auto plane4 = geometry::TriangleMesh::CreateCylinder( R, H, resolution, split );
        plane4->PaintUniformColor( Vector3d( 0.0, 0.0, 0.0 ) );
        Rt << 1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, 1.0, 0.0,  i,
              0.0, 0.0, 0.0, 1.0;
        plane4->Transform( Rt );
        plane4->ComputeVertexNormals();
        vis.AddGeometry( plane4 );
    }
}

int main(int argc, const char* const argv[])  // int argc, char *argv[]
{
    // --config=cfg/config_luz_front.yaml
    /*** READ CONFGURATION ***/
    const cv::String keys =
        "{help h usage ? |             | print this message   }"
        "{config         |config.yaml  | path to config       }"
        "{p              |/tmp/pid.pid | path to pid          }"
        "{fps            | -1.0        | fps for output video }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Disparity calculator");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    std::string filename = parser.get<cv::String>("config");
    std::cout << "Configuration file is " << filename << "\n";

        // READ CONFIG FILE
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs.open(filename, cv::FileStorage::READ);

    std::cout << "Configuration is "
              << (fs.isOpened() ? "opened" : "not opened") << "\n";
    
    std::string input_addres, output_addres;
    std::string left_channel, right_channel, railway_channel;
    std::string stereo_points_channel;
    
    fs["input_addres"] >> input_addres;
    std::cout << "Input addres: " << input_addres << "\n";

    fs["output_addres"] >> output_addres;
    std::cout << "Output addres: " << output_addres << "\n";

    fs["left_channel"] >> left_channel;
    std::cout << "Left channel: " << left_channel << "\n";

    fs["right_channel"] >> right_channel;
    std::cout << "Right channel: " << right_channel << "\n";
    
    fs["railway_channel"] >> railway_channel;
    std::cout << "Railway channel: " << railway_channel << "\n";
    
    fs["stereo_points_channel"] >> stereo_points_channel;
    std::cout << "Stereo points out channel: " << stereo_points_channel << "\n";
    
    bool is_calibration_classic = bool(int(fs["is_calibration_classic"]) != 0);
    std::cout << "Calibration is "
              << (is_calibration_classic ? "" : "not ")
              << "classic\n";
    
        // Init calibration parameters
    int height, width, software_binning, hardware_binning;
    cv::Mat Q, mtxL, mtxR, distL, distR, rectifyL, rectifyR, projectL, projectR;
    fs["Q"] >> Q;
    fs["mtxL"] >> mtxL;
    fs["mtxR"] >> mtxR;
    fs["distL"] >> distL;
    fs["distR"] >> distR;
    fs["rectifyL"] >> rectifyL;
    fs["rectifyR"] >> rectifyR;
    fs["projectL"] >> projectL;
    fs["projectR"] >> projectR;
    fs["height"] >> height;
    fs["width"] >> width;
    fs["hardware_binning"] >> hardware_binning;
    fs["software_binning"] >> software_binning;
    std::cout << "Q matrix:\n" << Q << "\n";

        // PARAMETERS FOR STEREO (DISP IMG)
    Handler::StereoParams stereo_params;
    fs["minDisp"] >> stereo_params.minDisp;
    fs["maxDisp"] >> stereo_params.maxDisp;
    fs["blockSize"] >> stereo_params.blockSize;
    fs["preFilterCap"] >> stereo_params.preFilterCap;
    fs["P1"] >> stereo_params.P1;
    fs["P2"] >> stereo_params.P2;
    fs["speckleRange"] >> stereo_params.speckleRange;
    fs["speckleWindowSize"] >> stereo_params.speckleWindowSize;
    
    cv::Ptr<cv::StereoSGBM> bm;
    bm = cv::StereoSGBM::create(stereo_params.minDisp, stereo_params.maxDisp, stereo_params.blockSize);
    bm->setPreFilterCap(stereo_params.preFilterCap);
    bm->setP1(stereo_params.P1);
    bm->setP2(stereo_params.P2);
    bm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    bm->setSpeckleRange(stereo_params.speckleRange);
    bm->setSpeckleWindowSize(stereo_params.speckleWindowSize);
    
    
    // Read zcm file
    string zcm_file = "../zcm_logs/534_train_1911210018.zcm.1"; 
                      //"../zcm_logs/534_train_1911210018.zcm.0"; 
                      //"/home/roman/videos/data/534_loco/zcm_files/20191121/534_train_1911210018.zcm.1";
    // Open zcm log
    LogFile *zcm_log;
    zcm_log = new LogFile( zcm_file, "r" );
    if ( !zcm_log->good() )
    {
    cout << "Bad zcm log: " << zcm_file << endl;
    exit(0);
    }
    
    // --- START ---------- //
    Mat imgL, imgR, railway_mask;
    std::set < std::string > zcm_list;
    //int num_point = 1;
    bool view = false;
    RNG rng(12345);
    cv::Size binning_size = { int(width / software_binning), 
                              int(height / software_binning) };
    cv::Size origin_size = { int(width), 
                             int(height) };
    int64_t last_left_ts = 0;
    int64_t last_right_ts = 0;
    int64_t last_railway_ts = 0;
    
    long time_samp = 0;                                                     // Global time stemp
    bool Lflag = false, Rflag = false, Railflag = false; 
    
    cv::Mat leftMapX, leftMapY;
    cv::Mat rightMapX, rightMapY;
    cv::initUndistortRectifyMap(
         mtxL, distL, rectifyL, projectL,
        {width, height}, CV_32FC1, leftMapX, leftMapY
    );
    cv::initUndistortRectifyMap(
         mtxR, distR, rectifyR, projectR,
        {width, height}, CV_32FC1, rightMapX, rightMapY
    );
//    resize( leftMapX, leftMapX, binning_size, 0, 0, cv::INTER_LINEAR );
//    resize( leftMapY, leftMapY, binning_size, 0, 0, cv::INTER_LINEAR );
//    resize( rightMapX, rightMapX, binning_size, 0, 0, cv::INTER_LINEAR );
//    resize( rightMapY, rightMapY, binning_size, 0, 0, cv::INTER_LINEAR );
//    leftMapX /= software_binning;
//    leftMapY /= software_binning;
//    rightMapX /= software_binning;
//    rightMapY /= software_binning;
    
    shared_ptr< geometry::PointCloud > Cloud ( new geometry::PointCloud );  // Global cloud 3D points
    
        // Visualization
    visualization::Visualizer vis;
    vis.CreateVisualizerWindow( "Open3D_odometry", 1600, 900, 50, 50 );
        // Add Coordinate
    auto coord = geometry::TriangleMesh::CreateCoordinateFrame( 1.0, Vector3d( 0.0, 0.0, 0.0 ) );
    coord->ComputeVertexNormals();
    vis.AddGeometry( coord );
        // Grid
    PaintGrid( vis, 150.0, 1.0 );
        // Add Lidar
    vis.AddGeometry( Cloud );
    
    while(1)
    {
        int click = 0;
        const zcm::LogEvent* event = zcm_log->readNextEvent();
        if ( !event ) break;
            zcm_list.insert( event->channel );
        
            // Frame from camera
        long tts = 0;                                           // Local time stemp
        if ( event->channel == "LZcmCameraBaslerJpegFrame" )
        {
//            ZcmSauglState zcm_rev;
//            cout << "Revers: " << zcm_rev.reversor << endl;
            ZcmCameraBaslerJpegFrame zcm_msg; 
            zcm_msg.decode( event->data, 0, static_cast< unsigned >( event->datalen ) );
            imgL = imdecode( zcm_msg.jpeg, IMREAD_COLOR );
            
            Mat imgTemp;
            resize( imgL, imgL, origin_size, 0, 0, cv::INTER_LINEAR );
            resize( imgL, imgTemp, binning_size, 0, 0, cv::INTER_LINEAR );
            
            tts = zcm_msg.service.u_timestamp;
            last_left_ts = tts;
            cout << " --- L:\t" << tts << endl;
            imshow( "imgL", imgTemp );
            click = waitKey(50);
            
            Lflag = true;
            if ( abs(last_left_ts - last_right_ts) > 100 ) 
            {
                time_samp = tts;
                Rflag = false;
            }
        }
        else if ( event->channel == "RZcmCameraBaslerJpegFrame" )
        {
            ZcmCameraBaslerJpegFrame zcm_msg; 
            zcm_msg.decode( event->data, 0, static_cast< unsigned >( event->datalen ) );
            imgR = imdecode( zcm_msg.jpeg, IMREAD_COLOR );
            
            Mat imgTemp;
            resize( imgR, imgR, origin_size, 0, 0, cv::INTER_LINEAR );
            resize( imgR, imgTemp, binning_size, 0, 0, cv::INTER_LINEAR );
            
            tts = zcm_msg.service.u_timestamp;
            last_right_ts = tts;
            cout << " --- R:\t" << tts << endl;
            imshow( "imgR", imgTemp );
            click = waitKey(30);
            
            Rflag = true;
            if ( abs(last_left_ts - last_right_ts) > 100 ) // time_samp != tts
            {
                time_samp = tts;
                Lflag = false;
            }
        }
        else if ( event->channel == "LRailDetection" )
        {
            ZcmRailDetectorMask zcm_msg;
            zcm_msg.decode( event->data, 0, static_cast< unsigned >( event->datalen  ) );
//            std::vector< char > jpeg_buf;
//            jpeg_buf.assign( zcm_msg.mask.data(), 
//                             zcm_msg.mask.data() + zcm_msg.mask.size() );
//            railway_mask = cv::imdecode( jpeg_buf, cv::IMREAD_GRAYSCALE );
            railway_mask = cv::imdecode( zcm_msg.mask, cv::IMREAD_GRAYSCALE );
            
            Mat imgTemp;
            resize( railway_mask, railway_mask, origin_size, 0, 0, cv::INTER_LINEAR );
            resize( railway_mask, imgTemp, binning_size, 0, 0, cv::INTER_LINEAR );
            
            cv::blur(railway_mask, railway_mask, {5, 5});
            cv::inRange(railway_mask, 15, 255, railway_mask);
            
            last_railway_ts = zcm_msg.service.u_timestamp;
            //cout << " --- LRailDetection:\t\t" << last_railway_ts << endl;
            imshow( "railway_channel", imgTemp );
            click = waitKey(30);
        }
        
        
        
            // If finded two same frame-time, make depth map
        if( Lflag && Rflag )
        {
            cout << "Pair same frames found \n" << endl;
            Lflag = false;
            Rflag = false;
            
            
                // Undistort & rectify
            cv::Mat imgRemap[2];
//            for ( int i = 0; i < 2; i++ )
//                remap( img[i], imgRemap[i], 
//                       rmap[i][0], rmap[i][1], 
//                       INTER_LINEAR );
            cv::remap(
                imgL, imgRemap[0],
                leftMapX, leftMapY, cv::INTER_LINEAR);
            cv::remap(
                imgR, imgRemap[1],
                rightMapX, rightMapY, cv::INTER_LINEAR);

            
            Mat imgGrey[2]; 
            cvtColor( imgRemap[0], imgGrey[0], COLOR_BGR2GRAY);
            cvtColor( imgRemap[1], imgGrey[1], COLOR_BGR2GRAY);
//            imshow( "iRL", imgRemap[0] );
//            imshow( "iRR", imgRemap[1] );
//            imshow( "GL", imgGrey[0] );
//            imshow( "GR", imgGrey[1] );
//            click = waitKey(0);
            
            Mat frameLR = Mat::zeros(Size(2 * width, height), CV_8UC3);
            Rect r1(0, 0, width, height);
            Rect r2(width, 0, width, height);
            putText( imgRemap[0], "L", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
            putText( imgRemap[1], "R", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
            imgRemap[0].copyTo(frameLR( r1 ));
            imgRemap[1].copyTo(frameLR( r2 ));
            for( int i = 0; i < frameLR.rows; i += 100 )
                for( int j = 0; j < frameLR.cols; j++ )
                    frameLR.at< Vec3b >(i, j)[2] = 255;
            resize( frameLR, frameLR, Size(2 * binning_size.width, binning_size.height), 0, 0, cv::INTER_LINEAR );
            imshow( "LR", frameLR );
            //waitKey(0);
            
            
                // Calculate disparity
            cv::Mat disparity;
            bm->compute( imgGrey[0], imgGrey[1], disparity );
    
            float coef = imgL.size().width/disparity.size().width;
            std::cout << "Disparity coefficient: " << coef << "\n";
            
            cv::Mat disparity_viz;
            disparity.convertTo( disparity_viz, CV_8U, 255/(96*16.0) );
            //if ( !disparity_masking.empty() ) disparity_masking.convertTo( disparity_viz, CV_8U, 255/(96*16.0) );
            cv::applyColorMap( disparity_viz, disparity_viz, cv::COLORMAP_HSV );
            resize( disparity_viz, disparity_viz, binning_size, 0, 0, cv::INTER_LINEAR );
            cv::imshow("disp", disparity_viz);
            
//                // Nomalization
//            double minVal; double maxVal;
//            minMaxLoc( disparity, &minVal, &maxVal );
//            Mat imgDispNorm;
//            disparity.convertTo( imgDispNorm, CV_8UC1, 255/(maxVal - minVal) );
//            Mat imgDisp_color;
//            applyColorMap( imgDispNorm, imgDisp_color, COLORMAP_RAINBOW );   // COLORMAP_HOT
//            resize( imgDisp_color, imgDisp_color, binning_size, 0, 0, cv::INTER_LINEAR );
//            imshow( "Disparity", imgDisp_color );
            
                // 3D points with mask
            Mat points3D, disparity_masking;
            disparity.convertTo( disparity, CV_32F );
            disparity.copyTo( disparity_masking, railway_mask );
            reprojectImageTo3D( disparity_masking, points3D, Q, 3 ); 
            
                // 3D point
            Cloud->Clear();
            for ( size_t i = 0; i < points3D.total(); i ++ )
            {
                Vector3d temPoint, tempColor;
                temPoint << double( points3D.at< Vec3f >( static_cast<int>(i) )[0] ),
                            double( points3D.at< Vec3f >( static_cast<int>(i) )[1] ),
                            double( points3D.at< Vec3f >( static_cast<int>(i) )[2] );
                tempColor.x() = double( imgRemap[0].at< Vec3b >(int(i)).val[2] / 255.0 );
                tempColor.y() = double( imgRemap[0].at< Vec3b >(int(i)).val[1] / 255.0 );
                tempColor.z() = double( imgRemap[0].at< Vec3b >(int(i)).val[0] / 255.0 );
                Cloud->points_.push_back( temPoint );
                Cloud->colors_.push_back( tempColor );
            }
            waitKey(50);
            
            vis.UpdateGeometry();
            //vis.PollEvents();
            vis.Run();
            click = waitKey(0);
        }
        if ( click == 'v' || click == 'V' ) view = true;
        else if ( click == 27 || click == 'q' || click == 'Q' ) break;                                // Interrupt the cycle, press "ESC"
        
    }
    
        // Output list zcm parameters
    std::cout << "zcm_list: " << std::endl;
    for(auto i : zcm_list)
        std::cout << "\t" << i << std::endl;
    
    fs.release();
    
    return 0;    
    
    
    
    
    // Start zcm logplayer
    
/*    zcm::ZCM zcm_in { input_addres.c_str() };
    zcm::ZCM zcm_out { output_addres.c_str() };

    if (!(zcm_in.good() and zcm_out.good()))
    {
        std::cout << "Zcm.good return false";
        return 1;
    }
//        // PARAMETERS FOR STEREO (DISP IMG)
//    Handler::StereoParams stereo_params;
//    fs["minDisp"] >> stereo_params.minDisp;
//    fs["maxDisp"] >> stereo_params.maxDisp;
//    fs["blockSize"] >> stereo_params.blockSize;
//    fs["preFilterCap"] >> stereo_params.preFilterCap;
//    fs["P1"] >> stereo_params.P1;
//    fs["P2"] >> stereo_params.P2;
//    fs["speckleRange"] >> stereo_params.speckleRange;
//    fs["speckleWindowSize"] >> stereo_params.speckleWindowSize;
    
    Handler handlerObject(
        left_channel, right_channel, railway_channel,
        stereo_points_channel, stereo_params,
        &zcm_out, hardware_binning, software_binning);
    
        // CALIBRATION
    if (is_calibration_classic)
    {
        cv::Mat leftMapX, leftMapY;
        cv::Mat rightMapX, rightMapY;

        cv::initUndistortRectifyMap(
            mtxL, distL, rectifyL, projectL,
            {width, height}, CV_32FC1, leftMapX, leftMapY
        );
        cv::initUndistortRectifyMap(
            mtxR, distR, rectifyR, projectR,
            {width, height}, CV_32FC1, rightMapX, rightMapY
        );

        handlerObject.set_calib(
            leftMapX, leftMapY, rightMapX, rightMapY, Q);
        std::cout << "Classic calibration\n";
    }
    else
    {
        cv::Mat left_H(4, 4, CV_32F);
        cv::Mat right_H(4, 4, CV_32F);

        handlerObject.set_calib(
            left_H, right_H, Q);
        std::cout << "Unclassic calibration\n";

        std::cout << "left_H matrix:\n" << left_H << "\n";
        std::cout << "right_H matrix:\n" << right_H << "\n";
    }

    fs.release();
    
    
        // Visualization
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow( "3D_points", 1600, 900, 50, 50 );
        // Add Coordinate
    auto coord = open3d::geometry::TriangleMesh::CreateCoordinateFrame( 1.0, Eigen::Vector3d( 0.0, 0.0, 0.0 ) );
    coord->ComputeVertexNormals();
    vis.AddGeometry( coord );
    
    std::shared_ptr< open3d::geometry::PointCloud > Cloud ( new open3d::geometry::PointCloud ); // Global cloud 3D points
    vis.AddGeometry( Cloud );
        // Grid
    PaintGrid( vis, 150.0, 1.0 );
    
    handlerObject.Cloud = Cloud;
    handlerObject.vis = &vis;
    vis.PollEvents();
    
    
    
    zcm_in.subscribe( left_channel, &Handler::handleMessage, &handlerObject );
    zcm_in.subscribe( right_channel, &Handler::handleMessage, &handlerObject );
    zcm_in.subscribe( railway_channel, &Handler::handleRailway, &handlerObject );
    zcm_in.subscribe( stereo_points_channel, &Handler::handle3Dpoints, &handlerObject );
    
    
    cv::waitKey(100);
    zcm_in.run();
    
    
    return 0;*/
}




















/*
// Read zcm file
string zcm_file = "/home/roman/videos/data/534_loco/zcm_files/20191122/534_train_1911221125.zcm.1";
// Open zcm log
LogFile *zcm_log;
zcm_log = new LogFile( zcm_file, "r" );
if ( !zcm_log->good() )
{
cout << "Bad zcm log: " << zcm_file << endl;
exit(0);
}

// --- START ---------- //
Mat img;
std::set < std::string > zcm_list;
//int num_point = 1;
bool view = false;
RNG rng(12345);

while(1)
{
int click = 0;
const zcm::LogEvent* event = zcm_log->readNextEvent();
if ( !event ) break;
zcm_list.insert( event->channel );

    // Frame from camera
if ( event->channel == "LZcmCameraBaslerJpegFrame" )  //FLZcmCameraBaslerJpegFrame	SLZcmCameraBaslerJpegFrame
{
    ZcmSauglState zcm_rev;
    cout << "Revers: " << zcm_rev.reversor << endl;
    ZcmCameraBaslerJpegFrame zcm_msg;
    zcm_msg.decode( event->data, 0, static_cast< unsigned >( event->datalen  ) );
    img = cv::imdecode( zcm_msg.jpeg, cv::IMREAD_COLOR );
    resize( img, img, Size(1280, 720), 0, 0, cv::INTER_LINEAR );
    imshow( "flow", img );
    click = waitKey(0);    
}
if ( click == 'v' || click == 'V' ) view = true;

if( click == 27 || click == 'q' || click == 'Q' ) break;                                // Interrupt the cycle, press "ESC"
}

// Output list zcm parameters
std::cout << "zcm_list: " << std::endl;
for(auto i : zcm_list)
std::cout << "\t" << i << std::endl;
*/
