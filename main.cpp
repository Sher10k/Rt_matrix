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

    zcm::ZCM zcm_in { input_addres.c_str() };
    zcm::ZCM zcm_out { output_addres.c_str() };

    if (!(zcm_in.good() and zcm_out.good()))
    {
        std::cout << "Zcm.good return false";
        return 1;
    }
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
    
    
    return 0;
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
