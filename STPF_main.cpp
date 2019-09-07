#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <ctime>
#include <sstream>
#include <algorithm>

#include <sys/stat.h> //create directory
#include <errno.h>    // errno, ENOENT, EEXIST
#if defined(_WIN32)
#include <direct.h>   // _mkdir
#endif
#include <iterator>

#include "Timer.h"
#include "LS.h"
#include "EdgeMap.h"
#include "EDLib.h"

#define DivideErrThre_e 1.4//edge breaking threshold
#define DivideErrThre_m 5
#define BufferThre 30
#define edgeLenThre 12
#define angularThre 90

#define PI 3.1415926

using namespace Eigen;
using namespace cv;
using namespace std;

double edge_detction_time = 0.0;
double edge_splitting_time = 0.0;
double sm_time = 0.0;
double total_time = 0.0;

struct EdgeFragment{//divided edge fragments from detected edge segments
    int seg_id;//edge fragements belong to the seg_id th detected segment
    int start_idx;// its start index in the seg_id th detected segment
    int end_idx;// its end index in the seg_id th detected segment
};

//int label_flag = 1;//finish this whole frame labeling (press key 's')
int mbd_gt = 0;//mouse button down for finish current shape labeling
int lbd_gt = 0;//mouse middle button dow-L lib -Wl,-rpath,'$$ORIGIN/lib'n for selecting edge fragments
Point pt_lbd_gt, pt_mv_gt;//left mouse button down point and mouse move points
//vector<Point>gtLabelPts;//store left mouse button clicked points

void createColorPallet(vector<vector<int> > &color_pallet);

void CallBackFunc(int event, int x, int y, int flags, void* userdata);//mouse event response function
float distance(float x1, float y1, float x2, float y2);//compute distance between two points (x1,y1) and (x2,y2)
int EdgeBreakFit(float xyh[3][4]);//edge segment breaking measure computing
vector<vector<int> > randColor(int num_color);//generate different colors for each fragment
Mat drawESs(EdgeMap *map, Mat frame_label_tmp);//draw edge segments
vector<vector<Point> > EdgeFilterBreak(EdgeMap *map);//edge segment breaking algorithm
int findEF(Point pt, vector<vector<Point> > edgeFragments, int &EF_id, int &Pixel_id);//find the closest fragment to the mouse pointer
int drawEF(vector<Point> &edgeFragment, Mat &frame_label_tmp, int rc, int gc, int bc);//draw one edge fragment
int drawEFs( vector<vector<Point> > &edgeFragments, Mat &frame_label_tmp, vector<vector<int> > rand_color);//draw edge fragments
int linesOrder(Point pt1, Point pt2, Point pt3, Point pt4);//find the closed endpoints of two segments
int linesToPtOrder(Point pt0, Point pt1, Point pt2);//find the closed endpoints of a segments to a certain point
int pointsOnLine(Mat &frame, Point pt1, Point pt2, vector<Point> &tmp_points);//find points on a straight line
int addLabelFragments(Mat &frame, int EF_LS_flg, Point pt_lbd_gt,vector<vector<Point> > edgeFragments,
                      vector<vector<Point> > &labeledFragments, vector<int> &labeledEFID, vector<int> &labeledEFtype);//add new fragment or segmet to current shape
int finishCurrentLabel(Mat &frame,vector<vector<Point> > &labeledFragments, vector<int> &labeledEFID, vector<int> &labeledEFtype);
//finish current shape labeling and results a closed boundary
int drawLFs(vector<vector<Point> > &labeledFragments, vector<int> &labeledEFtype, Mat &show_label_tmp);//draw label fragments and segments
void drawLabelResults(int biMultiClass_flg, vector<vector<int> > &color_pallet, Mat &edge_map_classes, Mat &edge_map_instances, Mat &region_map_classes, Mat &region_map_instances,
                      vector<vector<vector<Point> > > &labeledShapes, vector<int> &labeledShapeIdx, vector<int> &labeledShapeClassName);

void drawGrayEdgeFragments(Mat &tmp_frame, EdgeMap *map);
//void drawGrayBoundaries(Mat &tmp_frame, vector<vector<vector<Point> > > &shapeTemplates);//generate distance transform map
void drawEdgeMap(Mat &tmp_frame, EdgeMap *map);
void drawBoundaries(Mat &tmp_frame,vector<vector<vector<Point> > > &shapeTemplates, vector<vector<Mat> > &Hs, vector<vector<int> > &color_pallet);//draw tracked boundaries

int inputNewShapeYN(int mbd_gt);//if start labeling a new shape on the current image

void drawShapeTemplate(Mat &edge_map, vector<vector<Point> > &shapeTemplate);
void generateDistMapTemplate(Mat &frame, vector<vector<Point> > &shapeTemplate, Mat &dist_map, Mat &label_map);
void generateDistMapTemplates(Mat &frame, vector<vector<vector<Point> > >&shapeTemplates, vector<Mat> &dist_maps,vector<Mat> &label_maps);
vector<vector<Point> > edgePixelFilter(EdgeMap *map, Mat &Dist, Mat &h);

void searchMethod(Mat &frame, vector<vector<Point> > edgeFragments, vector<vector<Point> > &shapeTemplate, Mat &dist_map, Mat &label_map, vector<Mat> &Hi);
void shapeTracker(Mat &frame, EdgeMap *map, vector<vector<vector<Point> > > &shapeTemplates, vector<Mat> &dist_maps, vector<Mat> &label_maps, vector<vector<Mat> > &Hs);
//===========================MAIN FUNCTION START===========================
//===========================MAIN FUNCTION START===========================
//===========================MAIN FUNCTION START===========================
//===========================MAIN FUNCTION START===========================


int video_proc = 0;//video_proc: 0 read from webcam, 1 read from video file

int k, pk=1;
int pki = 1;
bool tracking_flg = false;

vector<Point> points; //contour of pre-shape
Point pt_lbd, pt_mv;
int mbd = 0;//flag of right button click

double ave_fps = 0;//average fps

int frame_width = 0;
int frame_height = 0;

//int start_frame_id = 0;

int main(int argc, char* argv[]){

    int key;
    //int video_proc = atoi(para_lines[1][1].c_str());//0;//video_proc: 0 read from webcam, 1 read from video file
    int biMultiClass_flg = 0;//atoi(argv[1]);//0: bi-class, 1: multi-class
    int simple_shape = 0;//0: objects are defined by multiple boundaries, 1: objects are defined by single boundary
    int start_frame_id = 0;//BookStand.avi
    int ini_flg=0;

    VideoCapture cap;

    Mat frame;
    Mat grayscaleFrame;
    //int width, height;
    //int frame_width;
    //int frame_height;

    int frame_whole_id=0;//indicates tracked frame numbers
    int frame_id = 0;//indicates recorded frame numbers

    if(argc==3){
        cout<<"Read video from recorded camera..."<<endl;
    }
    else if(argc==2){
        cout<<"Read video from webcam..."<<endl;
    }

    video_proc = atoi(argv[1]);

    //VideoCapture cap;
    if(video_proc==0){//read video from webcam
        cap = VideoCapture(0);
    }
    else if(video_proc==1){//read video from recorded video

        ini_flg = 1;

        const char* video_path = argv[2];
        cap = VideoCapture(video_path);
        //skip "start_frame_id-1" frames
        for(int i = 0; i<start_frame_id; i++){
            cout<<" ignore frame: "<<i<<endl;
            if(!cap.read(frame)){
                break;
            }
        }
        frame_id = frame_id + start_frame_id;
        //frame_whole_id = frame_whole_id + start_frame_id;
    }

    if(!cap.isOpened())  // check if video reading succeeded
        return -1;

    //Write captured video to Video_frame
/*    VideoWriter video;
    if(video_proc==0){
        video = VideoWriter("RECORDED_VIDEO.avi",CV_FOURCC('M','J','P','G'),30,Size(frame_width,frame_height),true);
    }
    //Write tracking results Video_results
    VideoWriter videor;
    videor = VideoWriter("RECORDED_RESULTS.avi",CV_FOURCC('M','J','P','G'),30,Size(frame_width,frame_height),true);

    remove("TRACKING_RESULTS.txt");//remove prevoius results
*/
    //==========================Initialization variables=======================
    vector<vector<vector<Point> > > shapeTemplates;
    vector<vector<vector<Point> > > shapeOfFramei;
    vector<vector<Mat> > Hs;//homography transformation matrix of the current frame
    int init_H = 1;
    vector<Mat> dist_maps;
    vector<Mat> label_maps;

    //obtain by manually initializaton
    //each shape comprised of a set of points

    vector<vector<vector<Point> > > labeledShapes;//number equals to the number of closed boundaries
    vector<int> labeledShapeClassName;//store the class names
    vector<int> labeledShapeIdx;//number equals to the number of closed boundaries

    //number equals to the number selected and drawn segments (EF corresponding to that of the detected EF's index, otherwise -1)
    vector<vector<int> > labeledShapeEFID;
    //number equals to the number selected or drawn segments(EF 1, LS 2, GAP 3)
    vector<vector<int> > labeledShapeEFtypes;

    vector<vector<Point> > labeledFragments;
    vector<int> labeledEFID;//indicates the ID of labeled edge fragment EF
    vector<int> labeledEFtype;//1 indicates EF, 2 means draw line segment, and 3 means gap
    vector<string> classNameList;//store class name of each classes

    //generate color pallet
    vector<vector<int> > color_pallet;
    createColorPallet(color_pallet);

    //labeling control keys and flags
    int label_key = 0;
    int EF_LS_flg = 0;//indicate labeling by selecting edge fragments "0" or drawing lines "1"
    int EF_show_flg = 1;//show edge fragments or not

    //Write Video_frame
    int frame_width_w=   cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height_w=   cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    VideoWriter video;
    if(!video_proc){
        video = VideoWriter("../ShpTkr_data/RECORDED_VIDEO.avi",CV_FOURCC('M','J','P','G'),30,Size(frame_width_w,frame_height_w),true);
    }

    for(;;){
        //cap >> frame; // get a new frame from camera
        if(video_proc){//read recorded video
            if(frame_whole_id==0||tracking_flg){
                if(!cap.read(frame)){
                    break;
                }
            }
        }else{
            if(!cap.read(frame)){
                break;
            }
        }

        frame_width = frame.size().width;
        frame_height = frame.size().height;

        //===============Edge/Line segments detection================START
        Timer time_ed;
        time_ed.Start();

        cvtColor(frame,grayscaleFrame, CV_BGR2GRAY);//convert rgb image to gray image
        unsigned char *srcImg;
        srcImg = grayscaleFrame.data;
        EdgeMap *map = DetectEdgesByED(srcImg, frame.size().width, frame.size().height, SOBEL_OPERATOR, 25, 8, 1.0);//36 8 1.0//Edge Drawing for edge detection

        time_ed.Stop();
        edge_detction_time +=time_ed.ElapsedTime();

        //Mat tmp_frame = frame.clone();

        /*vector<vector<Point> > edgeFragments;
        edgeFragments = EdgeFilterBreak(map);//divide edge segments into fragments
        vector<vector<int> > rand_color_EFs = randColor(edgeFragments.size());*/
        //---------------Edge/Line segments detection-----------------END

//================================================INITIALIZATION=============================================


        if(!tracking_flg){
            //drawEdgeMap(frame, map);

            Mat edge_map_classes, edge_map_instances, region_map_classes, region_map_instances;

            edge_map_classes = Mat::zeros(frame_height,frame_width,CV_8UC3);
            region_map_classes = Mat::zeros(frame_height,frame_width,CV_8UC3);
            edge_map_instances = Mat::zeros(frame_height,frame_width,CV_8UC3);
            region_map_instances = Mat::zeros(frame_height,frame_width,CV_8UC3);

            vector<vector<Point> > edgeFragments;
            edgeFragments = EdgeFilterBreak(map);//divide edge segments into fragments
            vector<vector<int> > rand_color_EFs = randColor(edgeFragments.size());

            int bdy = 0;//0 produce closed boundary by pressing "c", 1 produce open boundary by pressing "o"

            while(1&&ini_flg){

                namedWindow("Initialization", CV_WINDOW_NORMAL);

                Mat show_label_tmp = frame.clone();
                if(EF_show_flg){//show edge fragments or not
                    drawEFs(edgeFragments, show_label_tmp, rand_color_EFs);//draw edge fragments
                }

                if(labeledShapes.size()>0){//draw labeled shapes
                    for(int i = 0; i < labeledShapes.size(); i++){
                        drawLFs(labeledShapes[i],labeledShapeEFtypes[i],show_label_tmp);
                    }
                }

                setMouseCallback("Initialization", CallBackFunc, &pt_lbd_gt);//get mouse events

                int foundIdx = -1;
                int pixelIdx = -1;

                if(!mbd_gt){//--continue current shape labeling
                    if(lbd_gt){//EVENT-mouse left button down
                        addLabelFragments(frame, EF_LS_flg, pt_lbd_gt, edgeFragments, labeledFragments, labeledEFID, labeledEFtype);
                        lbd_gt = 0;
                    }else{//EVENT-mouse move
                        if(!EF_LS_flg){//show the closet EF when at EF selecting mode
                            //int foundIdx = -1;
                            //int pixelIdx = -1;
                            findEF(pt_mv_gt, edgeFragments,foundIdx,pixelIdx);
                            if(foundIdx>-1 && foundIdx<edgeFragments.size()){
                                drawEF(edgeFragments[foundIdx], show_label_tmp, 255,0,0);//r g b

                                if(pixelIdx > 0 && pixelIdx < edgeFragments[foundIdx].size()-2){
                                    //draw current breaking two pixels
                                    Vec3b & color1 = show_label_tmp.at<Vec3b>(edgeFragments[foundIdx][pixelIdx].y,edgeFragments[foundIdx][pixelIdx].x);
                                    color1[0] = 0;
                                    color1[1] = 255;
                                    color1[2] = 0;

                                    //draw current breaking two pixels
                                    Vec3b & color2 = show_label_tmp.at<Vec3b>(edgeFragments[foundIdx][pixelIdx+1].y,edgeFragments[foundIdx][pixelIdx+1].x);
                                    color2[0] = 255;
                                    color2[1] = 0;
                                    color2[2] = 0;
                                }
                            }
                        }else{//show the connectivity from mouse pointer to start and end points
                            if(labeledFragments.size() == 1){
                                line(show_label_tmp,pt_mv_gt,labeledFragments[0][0],Scalar(0,255,0),1,8,0);
                                line(show_label_tmp,pt_mv_gt,labeledFragments[labeledFragments.size()-1][labeledFragments[labeledFragments.size()-1].size()-1],Scalar(0,255,0),1,8,0);
                            }
                            else if(labeledFragments.size() > 1){
                                line(show_label_tmp,pt_mv_gt,labeledFragments[0][0],Scalar(0,255,0),1,8,0);
                                line(show_label_tmp,pt_mv_gt,labeledFragments[labeledFragments.size()-1][labeledFragments[labeledFragments.size()-1].size()-1],Scalar(0,0,255),1,8,0);
                            }
                        }
                    }//if(lbd_gt)
                }
                else{//finish labeling
                    if(labeledFragments.size()>0 && labeledFragments[0].size()>1){
                        //get closed shape contour
                        if(!bdy){
                            finishCurrentLabel(frame,labeledFragments,labeledEFID, labeledEFtype);
                        }

                        labeledShapes.push_back(labeledFragments);
                        labeledShapeEFID.push_back(labeledEFID);
                        labeledShapeEFtypes.push_back(labeledEFtype);

                        //generate binary edge and region maps of labels
                        //drawLabelEdges(bw_im_edge, bw_im_region, labeledFragments);
                        //draw current labeling shape
                        Mat show_label_tmp_tmp = show_label_tmp.clone();
                        drawLFs(labeledFragments, labeledEFtype, show_label_tmp_tmp);
                        imshow("Initialization",show_label_tmp_tmp);
                        //imshow("labeling2",show_label_tmp);


                        //follow codes handle objects with holes
                        if(labeledShapeIdx.size() == 0){
                            labeledShapeIdx.push_back(0);
                        }
                        int YorN = 1;
                        if(0 == simple_shape){
                            YorN = inputNewShapeYN(mbd_gt);
                        }else if(1 == simple_shape){
                            YorN = 1;
                        }



                        if(1 == YorN){//this is the last boundary of the current object

                            //drawLFs(labeledFragments, labeledEFtype, show_label_tmp_tmp);
                            //imshow("Labeling",show_label_tmp_tmp);

                            labeledShapeIdx.push_back(labeledShapeIdx[labeledShapeIdx.size()-1]+1);
                            //generate binary edge and region maps of labels

                            if(0 == classNameList.size()){
                                classNameList.push_back("1");
                            }
                            labeledShapeClassName.push_back(0);

                            //drawLabelResults(biMultiClass_flg, color_pallet, edge_map_classes, edge_map_instances, region_map_classes, region_map_instances, labeledShapes, labeledShapeIdx, labeledShapeClassName);

                            drawLFs(labeledFragments, labeledEFtype, show_label_tmp);

                            cout<<"--Labeled Object ID: "<<labeledShapeClassName.size()-1<<endl;
                            //cout<<"--Object Class Name: "<<classNameList[labeledShapeClassName[labeledShapeClassName.size()-1]]<<endl;
                            cout<<"-----------------------------------------------"<<endl;

                        }else if(0 == YorN){//this is not the last boundary of the current object
                             labeledShapeIdx.push_back(labeledShapeIdx[labeledShapeIdx.size()-1]);
                        }else if(-1 == YorN){//undo the last selection

                            labeledShapes.pop_back();
                            labeledShapeEFID.pop_back();
                            labeledShapeEFtypes.pop_back();

                            if(labeledFragments.size()>0){
                               labeledFragments.pop_back();//undo the last selection
                               labeledEFID.pop_back();
                               labeledEFtype.pop_back();

                               labeledShapeIdx.pop_back();
                            }
                             mbd_gt = 0;
                             continue;
                        }

                        labeledFragments.clear();
                        vector<vector<Point> >().swap(labeledFragments);
                        labeledEFID.clear();
                        vector<int>().swap(labeledEFID);
                        labeledEFtype.clear();
                        vector<int>().swap(labeledEFtype);
                    }
                    mbd_gt = 0;
                }//if(!mbd_gt)

                //draw current labeling shape
                drawLFs(labeledFragments, labeledEFtype, show_label_tmp);

                //------------------------------------LABEL CONTROL---------------------------------------
                imshow("Initialization",show_label_tmp);
                label_key = cvWaitKey(1) & 255;

               if(97 == label_key){//"a"
                    EF_LS_flg = !EF_LS_flg;
                }
                else if(98 == label_key){//"b" split the current closet edge fragment into two edge fragments
                   if(foundIdx>-1 && foundIdx<edgeFragments.size() && pixelIdx > 0 && pixelIdx < edgeFragments[foundIdx].size()-2){
                        vector<Point> temp;

                        int szEFi = edgeFragments[foundIdx].size();

                        for(int i = szEFi-1; i > pixelIdx; i--){

                            temp.push_back(edgeFragments[foundIdx][edgeFragments[foundIdx].size()-1]);
                            edgeFragments[foundIdx].pop_back();
                        }

                        edgeFragments.push_back(temp);
                        temp.clear();
                        vector<Point>().swap(temp);

                        vector<int> temp_clr;

                        temp_clr.push_back(100+int(rand()%156));
                        temp_clr.push_back(100+int(rand()%156));
                        temp_clr.push_back(100+int(rand()%156));

                        rand_color_EFs.push_back(temp_clr);

                        temp_clr.clear();
                        vector<int>().swap(temp_clr);


                   }//
                }
                else if(101 == label_key){//"e"
                    EF_show_flg = !EF_show_flg;
                }else if(102 == label_key){//"f" undo the last selection
                    if(labeledFragments.size()>0){
                       labeledFragments.pop_back();//undo the last selection
                       labeledEFID.pop_back();
                       labeledEFtype.pop_back();
                    }
                }else if(99 == label_key){//"c"
                   cout<<bdy<<" output closed boundary"<<endl;
                   bdy = 0;
               }else if(111 == label_key){//"o"
                   cout<<bdy<<" output open boundary"<<endl;
                   bdy = 1;
               }else if((10 == label_key || 13 == label_key)&& 0 == labeledFragments.size()){//"Enter" finish current frame labeling

                   pki = 0;

                   destroyWindow("Initialization");

                   vector<vector<Point> >tmp_Shape;
                   vector<Point> tmp_boundary;

                   //vector<int> labeledShapeIdx;

                   for(int i = 0; i < labeledShapes.size(); i++){

                       if(0==i){
                           for(int j = 0; j < labeledShapes[i].size(); j++){
                               for(int k = 0; k < labeledShapes[i][j].size(); k++){
                                    tmp_boundary.push_back(labeledShapes[i][j][k]);
                               }
                           }
                           tmp_Shape.push_back(tmp_boundary);
                           tmp_boundary.clear();
                           vector<Point>().swap(tmp_boundary);

                       }else{//i>0

                           if(labeledShapeIdx[i-1]==labeledShapeIdx[i]){

                               for(int j = 0; j < labeledShapes[i].size(); j++){
                                   for(int k = 0; k < labeledShapes[i][j].size(); k++){
                                        tmp_boundary.push_back(labeledShapes[i][j][k]);
                                   }
                               }
                               tmp_Shape.push_back(tmp_boundary);
                               tmp_boundary.clear();
                               vector<Point>().swap(tmp_boundary);
                           }else{

                               shapeTemplates.push_back(tmp_Shape);
                               tmp_Shape.clear();
                               vector<vector<Point> >().swap(tmp_Shape);

                               for(int j = 0; j < labeledShapes[i].size(); j++){
                                   for(int k = 0; k < labeledShapes[i][j].size(); k++){
                                        tmp_boundary.push_back(labeledShapes[i][j][k]);
                                   }
                               }
                               tmp_Shape.push_back(tmp_boundary);
                               tmp_boundary.clear();
                               vector<Point>().swap(tmp_boundary);

                           }//!=
                       }//i>0

                      if(labeledShapes.size()-1==i){
                          shapeTemplates.push_back(tmp_Shape);
                          tmp_Shape.clear();
                          vector<vector<Point> >().swap(tmp_Shape);
                      }

                   }//for i = 0

                   //generate distance maps and closest points correspondences
                   Timer timerDTM;
                   timerDTM.Start();

                   generateDistMapTemplates(frame,shapeTemplates, dist_maps, label_maps);

                   timerDTM.Stop();
                   cout<<"Distance Map Time: "<<timerDTM.ElapsedTime()<<endl;

                   cout<<"Shape num: "<<shapeTemplates.size()<<endl;
                   cout<<">>>>>Start Tracking..."<<endl;
                   tracking_flg = 1;
                   break;

                }else if(27  == label_key){//"Esc"
                   return 0;
               }
            }//end while(1)

        }//if(!tracking_flg)
//-----------------------------------------End of Initialization--------------------------------------------

//=============================================start tracking===============================================

        if(tracking_flg){

            if(!video_proc){//record video when read video stream from webcam
                video.write(frame);
            }

            cout<<"--- fame_id: "<<frame_id<<"---"<<endl;

            if(init_H){
                vector<Mat> Hi;
                for(int i = 0; i < shapeTemplates.size(); i++){
                    Mat h = Mat::eye(3,3, CV_64F);
                    Hi.push_back(h);
                }

                init_H = 0;
                Hs.push_back(Hi);
                Hi.clear();
                vector<Mat>().swap(Hi);

            }//initialize homography matrics

            shapeTracker(frame, map, shapeTemplates, dist_maps, label_maps, Hs);

//            Mat rslt = Mat::zeros(frame.size().height,frame.size().width,CV_8U);

            for(int p = 0; p < shapeTemplates.size(); p++){
//                for(int t = 0; t < shapeTemplates[p].size(); t++){
//                    for(int k = 0; k < shapeTemplates[p][t].size(); k++){
//                        double x = double(shapeTemplates[p][t][k].x);
//                        double y = double(shapeTemplates[p][t][k].y);

//                        double hx = Hs[Hs.size()-1][p].at<double>(0,0)*x + Hs[Hs.size()-1][p].at<double>(0,1)*y + Hs[Hs.size()-1][p].at<double>(0,2);
//                        double hy = Hs[Hs.size()-1][p].at<double>(1,0)*x + Hs[Hs.size()-1][p].at<double>(1,1)*y + Hs[Hs.size()-1][p].at<double>(1,2);
//                        double hz = Hs[Hs.size()-1][p].at<double>(2,0)*x + Hs[Hs.size()-1][p].at<double>(2,1)*y + Hs[Hs.size()-1][p].at<double>(2,2);

//                        int r = hy/hz;
//                        int c = hx/hz;

//                        if(r>=0 && r < frame_height && c >= 0 && c < frame_width){
//                            Vec3b & color = frame.at<Vec3b>(r,c);
//                            color[0] = 255;
//                            color[1] = 255;
//                            color[2] = 0;
//                        }
//                    }

//                }

                int rp=0;
                int cp=0;

                int ri = 0;
                int ci = 0;

                for(int t = 0; t < shapeTemplates[p].size(); t++){
                    for(int k = 0; k < shapeTemplates[p][t].size(); k++){
                        double x = double(shapeTemplates[p][t][k].x);
                        double y = double(shapeTemplates[p][t][k].y);

                        double hx = Hs[Hs.size()-1][p].at<double>(0,0)*x + Hs[Hs.size()-1][p].at<double>(0,1)*y + Hs[Hs.size()-1][p].at<double>(0,2);
                        double hy = Hs[Hs.size()-1][p].at<double>(1,0)*x + Hs[Hs.size()-1][p].at<double>(1,1)*y + Hs[Hs.size()-1][p].at<double>(1,2);
                        double hz = Hs[Hs.size()-1][p].at<double>(2,0)*x + Hs[Hs.size()-1][p].at<double>(2,1)*y + Hs[Hs.size()-1][p].at<double>(2,2);

                        int r = hy/hz;
                        int c = hx/hz;

                        if(t == shapeTemplates[p].size()-1 && k == shapeTemplates[p][t].size()-1){

//                            line(rslt,Point(cp, rp), Point(ci,ri), Scalar(255), 1, 8);
                            line(frame,Point(cp, rp), Point(ci,ri), Scalar(0,0,255), 1, 8);

                        }

                        if(t==0&&k==0){
                            ri = r;
                            ci = c;
                        }else if(t!=0 || k!=0){
//                            line(rslt,Point(cp, rp), Point(c,r), Scalar(255), 1, 8);
                            line(frame,Point(cp, rp), Point(c,r), Scalar(0,0,255), 1, 8);
                        }

                        rp = r;
                        cp = c;
                    }

                }
            }

            frame_id++;

            //save the tracking results into a folder
//            char tmp_name[20];
//            sprintf(tmp_name,"%04d.png",frame_id);
//            imwrite("../ShpTkr_data/ETTracker_png/disc_390/"+string(tmp_name),rslt);

        }
//---------------------------------------------End of tracking-----------------------------------------------

        //Mat tmp_frame = frame.clone();
        //drawEdgeMap(tmp_frame,map);
        //drawBoundaries(tmp_frame, shapeTemplates, Hs, color_pallet);
        namedWindow("Display",CV_WINDOW_NORMAL);
        imshow("Display",frame);

        k = cvWaitKey(pki) & 255;

        //tmp_frame.release();

        frame_whole_id++;

        if(k == 27){//"Esc" key
            break;
            //return 0;
        }
        else if(k == 105){//"i" key
            ini_flg = 1;
        }else if(k == 115){//"s"
            pki = 0;
        }else if(k == 32){//114:"r", 32: "Space"
            pki = 1;
        }

        delete map;
    }//end for(;;)

    total_time = edge_detction_time + edge_splitting_time + sm_time;

    cout<<"ave edge detection time: "<<edge_detction_time/frame_id<<" ms"<<endl;
    cout<<"ave edge splitting time: "<<edge_splitting_time/frame_id<<" ms"<<endl;
    cout<<"ave search method time: "<<sm_time/frame_id<<" ms"<<endl;
    cout<<"ave total time: "<<total_time/frame_id<<" ms"<<endl;
//    cout<<"average FPS: "<<1000*frame_id/total_time<<endl;

    return 0;
}

void createColorPallet(vector<vector<int> > &color_pallet){
    int low = 115;
    int high = 255;

    vector<int> Ri, Gi, Bi;

    for(int i = low; i < high; i=i+10){
        Ri.push_back(i);
        Gi.push_back(i);
        Bi.push_back(i);
    }

    random_shuffle (Ri.begin(), Ri.end());
    random_shuffle (Gi.begin(), Gi.end());
    random_shuffle (Bi.begin(), Bi.end());

    for(int ri = 0; ri < Ri.size(); ri++){
        for(int gi = 0; gi < Gi.size(); gi++){
            for(int bi = 0; bi < Bi.size(); bi++){
                vector<int> tmp;
                tmp.push_back(Ri[ri]);
                tmp.push_back(Gi[gi]);
                tmp.push_back(Bi[bi]);

                color_pallet.push_back(tmp);

                tmp.clear();
                vector<int>().swap(tmp);

            }//for ri
        }//for gi
    }//for bi

}


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        //Point* pt = (Point*) userdata;
        //if(mbd_gt==0){
        pt_lbd_gt.x = x;
        pt_lbd_gt.y = y;
            //gtLabelPts.push_back(pt_lbd_gt);
        lbd_gt=1;
        //}
        //cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if  ( event == EVENT_RBUTTONDOWN )
    {
        //cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if  ( event == EVENT_MBUTTONDOWN )
    {
        mbd_gt = 1;//finish one object lableing
        //cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if ( event == EVENT_MOUSEMOVE )
    {
        //if(mbd_gt==0){
            pt_mv_gt.x = x;
            pt_mv_gt.y = y;
        //}
    }

    return;
}

float distance(float x1, float y1, float x2, float y2){
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

int EdgeBreakFit(float xyh[3][4]){

    float abcXe = 0;//end point test
    float abcXm = 0;//middle point test

    float a=0, b=0, c=0;
    a = xyh[1][0]-xyh[1][2];
    b = xyh[0][2]-xyh[0][0];
    c = xyh[0][0]*xyh[1][2]-xyh[1][0]*xyh[0][2];

    abcXe = a*xyh[0][3]+b*xyh[1][3]+c;
    abcXm = a*xyh[0][1]+b*xyh[1][1]+c;

    float erre = abs(abcXe)/sqrt(pow(a,2)+pow(b,2));
    float errm = abs(abcXm)/sqrt(pow(a,2)+pow(b,2));

    if(erre<DivideErrThre_e && errm<DivideErrThre_m){
        return 1;
    }else{
        return 0;
    }

    //return err;
}

vector<vector<int> > randColor(int num_color){
    vector<vector<int> > rand_color;

    int lowest=100, highest=255;
    int range=(highest-lowest)+1;
    vector<int> color_tmp;

    for(int i = 0; i < num_color; i++){
       color_tmp.push_back(lowest+int(rand()%range));
       color_tmp.push_back(lowest+int(rand()%range));
       color_tmp.push_back(lowest+int(rand()%range));

       rand_color.push_back(color_tmp);
       color_tmp.clear();
       vector<int>().swap(color_tmp);
    }

    return rand_color;
}

Mat drawESs(EdgeMap *map, Mat frame_label_tmp){//draw edge segments

    int lowest=100, highest=255;
    int range=(highest-lowest)+1;
    int rc, gc, bc;//the color of edge fragments

    for(int i = 0; i < map->noSegments; i++){

        rc = lowest+int(rand()%range);
        gc = lowest+int(rand()%range);
        bc = lowest+int(rand()%range);

        for(int j = 0; j < map->segments[i].noPixels; j++){
            int r = map->segments[i].pixels[j].r;
            int c = map->segments[i].pixels[j].c;

            Vec3b & color = frame_label_tmp.at<Vec3b>(r,c);
            color[0] = bc;
            color[1] = gc;
            color[2] = rc;
        }
    }

    return frame_label_tmp;
}

vector<vector<Point> > EdgeFilterBreak(EdgeMap *map){

    vector<EdgeFragment> EFs;//divided Edge Fragments

    for(int i = 0; i < map->noSegments; i++){

        EdgeFragment EF_tmp;

        if(map->segments[i].noPixels<5){
            EF_tmp.seg_id = i;
            EF_tmp.start_idx = 0;
            EF_tmp.end_idx = map->segments[i].noPixels-1;
            EFs.push_back(EF_tmp);
        }else{

            int newEdge = 1;//start a new fragment searching
            int iEgStrt = 0;

            float PT[3][4];

            for (int iEgEnd = iEgStrt+4; iEgEnd < map->segments[i].noPixels;){

                if(newEdge){//start a new small edge segment searching

                    EF_tmp.seg_id = i;
                    EF_tmp.start_idx = iEgStrt;

                    PT[0][0] = map->segments[i].pixels[iEgStrt].r;
                    PT[1][0] = map->segments[i].pixels[iEgStrt].c;
                    PT[2][0] = 1;
                }

                PT[0][1] = map->segments[i].pixels[(iEgEnd+iEgStrt)/2].r;
                PT[1][1] = map->segments[i].pixels[(iEgEnd+iEgStrt)/2].c;
                PT[2][1] = 1;

                PT[0][2] = map->segments[i].pixels[iEgEnd-2].r;
                PT[1][2] = map->segments[i].pixels[iEgEnd-2].c;
                PT[2][2] = 1;

                PT[0][3] = map->segments[i].pixels[iEgEnd].r;
                PT[1][3] = map->segments[i].pixels[iEgEnd].c;
                PT[2][3] = 1;

                if(EdgeBreakFit(PT)==1){//<DivideErrThre_e

                    if(iEgEnd < map->segments[i].noPixels - 2 ){
                        newEdge = 0;
                        iEgEnd +=2;
                    }else{
                        newEdge = 1;
                        EF_tmp.end_idx = map->segments[i].noPixels-1;
                        //if((EF_tmp.end_idx-EF_tmp.start_idx+1)>edgeLenThre){//EF_tmp.end_idx!=EF_tmp.start_idx
                            EFs.push_back(EF_tmp);// find the last small edge segment
                        //}
                        break;
                    }

                }else{
                    newEdge = 1;//start a new small edge
                    EF_tmp.end_idx = iEgEnd - 2;
                    //if((EF_tmp.end_idx-EF_tmp.start_idx+1)>edgeLenThre){//
                        EFs.push_back(EF_tmp);// find a new small edge segment
                    //}//

                    iEgStrt = iEgEnd-1;
                    EF_tmp.start_idx =  iEgStrt;
                    if(map->segments[i].noPixels-iEgStrt < 5){// there are less than 5 remaining pixels in this edge segment
                        EF_tmp.end_idx = map->segments[i].noPixels-1;
                        //if((EF_tmp.end_idx-EF_tmp.start_idx+1)>edgeLenThre){//EF_tmp.end_idx!=EF_tmp.start_idx
                            EFs.push_back(EF_tmp);
                        //}
                        break;
                    }else{// there are less than 3 remaining pixels in this edge segment and start a new small edge segment searching
                        iEgEnd = iEgStrt + 4;
                    }
                }//end if(LeastSquaresLineFit < DividedErrThre){} else{}

            }//for(int j)
        }//if(map...
    }//for(int i)

    vector<vector<Point> > points;
    for(int i = 0; i < EFs.size(); i++){
        int tmp_seg_id = EFs[i].seg_id;
        vector<Point> tmp_pts;

        for(int j = EFs[i].start_idx; j < EFs[i].end_idx + 1; j++){
            Point pt;
            pt.y = map->segments[tmp_seg_id].pixels[j].r;
            pt.x = map->segments[tmp_seg_id].pixels[j].c;
            tmp_pts.push_back(pt);

        }
        points.push_back(tmp_pts);
        tmp_pts.clear();
        vector<Point>().swap(tmp_pts);
    }

    return points;
}

int findEF(Point pt, vector<vector<Point> > edgeFragments,int &EF_id, int &Pixel_id){
    int indx_sel = 0;
    int pixel_sel = 0;
    float dist_sel = distance(float(pt.x),float(pt.y),
                              float(edgeFragments[0][0].x),
                              float(edgeFragments[0][0].y));
    for(int i = 0; i < edgeFragments.size(); i++){
        for(int j = 0; j < edgeFragments[i].size(); j++){
            float tmp_dist = distance(float(pt.x),float(pt.y),
                                      float(edgeFragments[i][j].x),
                                      float(edgeFragments[i][j].y));
            if(dist_sel > tmp_dist){
                indx_sel = i;
                pixel_sel = j;
                dist_sel = tmp_dist;
            }
        }//for j
    }//fori

    if(dist_sel<3){
        Pixel_id = pixel_sel;
        EF_id = indx_sel;
        return 1;
    }else{
        return 0;
    }
}

int drawEF(vector<Point> &edgeFragment, Mat &frame_label_tmp, int rc, int gc, int bc){//draw one edge fragment
    if(edgeFragment.size() > 0){

        for(int i = 0; i < edgeFragment.size(); i++){
            int r = edgeFragment[i].y;
            int c = edgeFragment[i].x;

            Vec3b & color = frame_label_tmp.at<Vec3b>(r,c);
            color[0] = bc;
            color[1] = gc;
            color[2] = rc;
        }
        return 1;
    }
    else{
        return 0;
    }
    //return frame_label_tmp;
}

int drawEFs(vector<vector<Point> > &edgeFragments, Mat &frame_label_tmp, vector<vector<int> > rand_color){
    if(edgeFragments.size()>0){

        for(int i = 0; i < edgeFragments.size(); i++){

            drawEF(edgeFragments[i], frame_label_tmp, rand_color[i][0], rand_color[i][1],rand_color[i][2]);

        }

        return 1;
    }
    else{
        return 0;
    }

}

int linesOrder(Point pt1, Point pt2, Point pt3, Point pt4){
//pt1---------------pt2  pt4---------------pt3
//to find the closest two endpoints which determines the storage order of each edge pixel
    float dists[4];

    dists[0] = distance(float(pt1.x),float(pt1.y),float(pt3.x),float(pt3.y));
    dists[1] = distance(float(pt1.x),float(pt1.y),float(pt4.x),float(pt4.y));
    dists[2] = distance(float(pt2.x),float(pt2.y),float(pt3.x),float(pt3.y));
    dists[3] = distance(float(pt2.x),float(pt2.y),float(pt4.x),float(pt4.y));

    int lorder = 0;
    float temp_dist = dists[0];
    for(int i = 1; i < 4; i++){
        if(dists[i] < temp_dist){
            lorder = i;
            temp_dist = dists[i];
        }
    }

    return lorder;//0, 1, 2, 3
}

int linesToPtOrder(Point pt0, Point pt1, Point pt2){

    float dists01 = distance(float(pt0.x),float(pt0.y),float(pt1.x),float(pt1.y));
    float dists02 = distance(float(pt0.x),float(pt0.y),float(pt2.x),float(pt2.y));

    if(dists01<dists02){
        return 1;
    }else{
        return 2;
    }

}

int pointsOnLine(Mat &frame, Point pt1, Point pt2, vector<Point> &tmp_points){

    LineIterator it(frame, pt1, pt2, 8);

    for(int i =0; i < it.count; i++,++it){
        tmp_points.push_back(it.pos());
    }

    return tmp_points.size();
}

int addLabelFragments(Mat &frame, int EF_LS_flg, Point pt_lbd_gt,vector<vector<Point> > edgeFragments,
                                        vector<vector<Point> > &labeledFragments, vector<int> &labeledEFID, vector<int> &labeledEFtype){
    vector<Point> tmp_add_pt;
    if(!EF_LS_flg){//add selected edgefragment
        int selEFIdx = -1;
        int selPixelIdx = -1;
        findEF(pt_lbd_gt,edgeFragments, selEFIdx, selPixelIdx);

        if(selEFIdx>-1 && selEFIdx < edgeFragments.size()){
            if(labeledFragments.size() == 0){//zero fragment
                labeledFragments.push_back(edgeFragments[selEFIdx]);//add new fragment
                labeledEFID.push_back(selEFIdx);
                labeledEFtype.push_back(1);//EF
            }
            else if(labeledFragments.size() == 1){//one fragment
                if(labeledFragments[0].size() == 1){//the first segments only have one pixel(it is added by clicking)
                    int ltopt = linesToPtOrder(labeledFragments[0][0],edgeFragments[selEFIdx][0], edgeFragments[selEFIdx][edgeFragments[selEFIdx].size()-1]);

                    if(ltopt==2){
                        reverse(edgeFragments[selEFIdx].begin(),edgeFragments[selEFIdx].end());
                    }


                    pointsOnLine(frame, labeledFragments[0][0], edgeFragments[selEFIdx][0], tmp_add_pt);

                    if(tmp_add_pt.size()>1){
                        labeledFragments.pop_back();
                        labeledEFID.pop_back();
                        labeledEFtype.pop_back();

                        tmp_add_pt.pop_back();
                        labeledFragments.push_back(tmp_add_pt);
                        tmp_add_pt.clear();
                        vector<Point>().swap(tmp_add_pt);
                        labeledEFID.push_back(-1);//LS no EF ID
                        labeledEFtype.push_back(2);//LS

                        labeledFragments.push_back(edgeFragments[selEFIdx]);
                        labeledEFID.push_back(selEFIdx);
                        labeledEFtype.push_back(1);//EF
                    }

                }
                else{

                    int lorder = linesOrder(labeledFragments[0][0], labeledFragments[0][labeledFragments[0].size()-1],
                            edgeFragments[selEFIdx][0], edgeFragments[selEFIdx][edgeFragments[selEFIdx].size()-1]);

                    switch(lorder){
                    case 0:{//pt1-pt3
                        reverse(labeledFragments[0].begin(),labeledFragments[0].end());
                        break;
                    }
                    case 1:{//pt1-pt4
                        reverse(labeledFragments[0].begin(),labeledFragments[0].end());
                        reverse(edgeFragments[selEFIdx].begin(),edgeFragments[selEFIdx].end());
                        break;
                    }
                    case 2:{//pt2-pt3
                        break;
                    }
                    case 3:{//pt2-pt4
                        reverse(edgeFragments[selEFIdx].begin(),edgeFragments[selEFIdx].end());
                        break;
                    }
                    }//switch

                    pointsOnLine(frame, labeledFragments[labeledFragments.size()-1][labeledFragments[labeledFragments.size()-1].size()-1],
                            edgeFragments[selEFIdx][0], tmp_add_pt);
                    if(tmp_add_pt.size()>2){
                        tmp_add_pt.pop_back();
                        tmp_add_pt.erase(tmp_add_pt.begin());
                        labeledFragments.push_back(tmp_add_pt);
                        tmp_add_pt.clear();
                        vector<Point>().swap(tmp_add_pt);
                        labeledEFID.push_back(-1);//GAP no EF ID
                        labeledEFtype.push_back(3);//GAP
                    }

                    labeledFragments.push_back(edgeFragments[selEFIdx]);//add new selected edge fragment into labeling shape
                    labeledEFID.push_back(selEFIdx);//EF ID
                    labeledEFtype.push_back(1);//EF
                }

            }
            else{//two and more fragments
                int ltopt = linesToPtOrder(labeledFragments[labeledFragments.size()-1][labeledFragments[labeledFragments.size()-1].size()-1],
                        edgeFragments[selEFIdx][0], edgeFragments[selEFIdx][edgeFragments[selEFIdx].size()-1]);
                if(ltopt==2){
                    reverse(edgeFragments[selEFIdx].begin(),edgeFragments[selEFIdx].end());
                }

                pointsOnLine(frame, labeledFragments[labeledFragments.size()-1][labeledFragments[labeledFragments.size()-1].size()-1],
                        edgeFragments[selEFIdx][0], tmp_add_pt);

                if(tmp_add_pt.size()>2){
                    tmp_add_pt.pop_back();
                    tmp_add_pt.erase(tmp_add_pt.begin());
                    labeledFragments.push_back(tmp_add_pt);
                    tmp_add_pt.clear();
                    vector<Point>().swap(tmp_add_pt);
                    labeledEFID.push_back(-1);//GAP no EF ID
                    labeledEFtype.push_back(3);//GAP
                }

                labeledFragments.push_back(edgeFragments[selEFIdx]);//
                labeledEFID.push_back(selEFIdx);//EF ID
                labeledEFtype.push_back(1);//EF

            }//if(labeledFragments.size()
        }//if(selEFIdx
    }//if(!EF_LS_flg)
    else{//add a new point
        //vector<Point> tmp_add_pt;
        if(labeledFragments.size() == 0){
            tmp_add_pt.push_back(pt_lbd_gt);

            labeledFragments.push_back(tmp_add_pt);
            tmp_add_pt.clear();
            vector<Point>().swap(tmp_add_pt);
            labeledEFID.push_back(-1);//LS no EF ID
            labeledEFtype.push_back(2);//LS
        }
        else if(labeledFragments.size() == 1){
            if(labeledFragments[0].size()==1){
                pointsOnLine(frame, labeledFragments[0][0], pt_lbd_gt, tmp_add_pt);
                labeledFragments.pop_back();
                labeledEFID.pop_back();
                labeledEFtype.pop_back();

                labeledFragments.push_back(tmp_add_pt);
                tmp_add_pt.clear();
                vector<Point>().swap(tmp_add_pt);
                labeledEFID.push_back(-1);//LS no EF ID
                labeledEFtype.push_back(2);//LS
            }
            else{
                int ltopt = linesToPtOrder(pt_lbd_gt,
                        labeledFragments[labeledFragments.size()-1][0],
                        labeledFragments[labeledFragments.size()-1][labeledFragments[labeledFragments.size()-1].size()-1]);
                if(1 == ltopt){
                    reverse(labeledFragments[labeledFragments.size()-1].begin(),labeledFragments[labeledFragments.size()-1].end());
                }

                pointsOnLine(frame, labeledFragments[0][labeledFragments[0].size()-1], pt_lbd_gt, tmp_add_pt);

                if(tmp_add_pt.size()>1){
                    tmp_add_pt.erase(tmp_add_pt.begin());
                    labeledFragments.push_back(tmp_add_pt);
                    tmp_add_pt.clear();
                    vector<Point>().swap(tmp_add_pt);
                    labeledEFID.push_back(-1);//LS no EF ID
                    labeledEFtype.push_back(2);//LS
                }

            }

        }
        else{
            pointsOnLine(frame, labeledFragments[labeledFragments.size()-1][labeledFragments[labeledFragments.size()-1].size()-1], pt_lbd_gt, tmp_add_pt);


            if(tmp_add_pt.size()>1){

                tmp_add_pt.erase(tmp_add_pt.begin());
                labeledFragments.push_back(tmp_add_pt);
                tmp_add_pt.clear();
                vector<Point>().swap(tmp_add_pt);
                labeledEFID.push_back(-1);//LS no EF ID
                labeledEFtype.push_back(2);//LS
            }

        }
    }

    return labeledFragments.size();
}

int finishCurrentLabel(Mat &frame,vector<vector<Point> > &labeledFragments, vector<int> &labeledEFID, vector<int> &labeledEFtype){

    if(labeledFragments.size()>0 && labeledFragments[0].size()>1){
        vector<Point> tmp_gap_close;
        pointsOnLine(frame, labeledFragments[labeledFragments.size()-1][labeledFragments[labeledFragments.size()-1].size()-1], labeledFragments[0][0],tmp_gap_close);

        if(tmp_gap_close.size()>2){
            tmp_gap_close.pop_back();
            tmp_gap_close.erase(tmp_gap_close.begin());
            labeledFragments.push_back(tmp_gap_close);
            tmp_gap_close.clear();
            vector<Point>().swap(tmp_gap_close);
            labeledEFID.push_back(-1);//GAP no EF ID
            labeledEFtype.push_back(3);
        }

        return labeledFragments.size();
    }
    else{
        return 0;
    }

}

int drawLFs(vector<vector<Point> > &labeledFragments, vector<int> &labeledEFtype, Mat &show_label_tmp){

    if(labeledFragments.size()>0){

        for(int i = 0; i < labeledFragments.size(); i++){

            for(int j = 0; j < labeledFragments[i].size(); j++){
                int r = labeledFragments[i][j].y;
                int c = labeledFragments[i][j].x;

                Vec3b & color = show_label_tmp.at<Vec3b>(r,c);
                //int pixelColor = labeledEFtype[i];
                switch(labeledEFtype[i]){
                case 1:{
                    color[0] = 0;
                    color[1] = 0;
                    color[2] = 255;
                    break;
                }
                case 2:{
                    color[0] = 255;
                    color[1] = 0;
                    color[2] = 255;
                    break;
                }
                case 3:{
                    color[0] = 0;
                    color[1] = 255;
                    color[2] = 0;
                    break;
                }
                }//switch

            }
        }
        return 1;
    }
    else{
        return 0;
    }
}

void drawLabelResults(int biMultiClass_flg, vector<vector<int> > &color_pallet, Mat &edge_map_classes, Mat &edge_map_instances, Mat &region_map_classes,
                      Mat &region_map_instances, vector<vector<vector<Point> > > &labeledShapes, vector<int> &labeledShapeIdx, vector<int> &labeledShapeClassName){

    if(labeledShapeIdx.size()>0){

        vector<vector<Point> > shapePoints;
        vector<Point> tmp_shp_points;
        int shapNum = 0;

        for(int j = 0; j < labeledShapes[labeledShapeIdx.size()-2].size(); j++){
            for(int t = 0; t < labeledShapes[labeledShapeIdx.size()-2][j].size(); t++){
                tmp_shp_points.push_back(labeledShapes[labeledShapeIdx.size()-2][j][t]);
            }//for(int t
        }//for(int j
        shapNum += 1;
        shapePoints.push_back(tmp_shp_points);
        tmp_shp_points.clear();
        vector<Point>().swap(tmp_shp_points);

        int tmp_idx = labeledShapeIdx[labeledShapeIdx.size()-2];
        for(int i = labeledShapeIdx.size()-3; i > -1; i--){
            if(tmp_idx == labeledShapeIdx[i]){

                for(int j = 0; j < labeledShapes[i].size(); j++){
                    for(int t = 0; t < labeledShapes[i][j].size(); t++){
                        tmp_shp_points.push_back(labeledShapes[i][j][t]);
                    }//for(int t

                }//for(int j

                shapNum += 1;
                shapePoints.push_back(tmp_shp_points);
                tmp_shp_points.clear();
                vector<Point>().swap(tmp_shp_points);
            }
            else{
                break;
            }

        }//for(int i

        if(!shapePoints.empty()){
            int shp_num = shapePoints.size();
            const Point** pts = new const Point*[shp_num]();
            for(int i = 0; i < shp_num; i++){
                pts[i] = new Point[shapePoints[i].size()];
                pts[i] = &shapePoints[i][0];
            }

            int *npt = new int[shp_num];
            for(int i = 0; i < shp_num; i++){
            //    *pt[i] = &vec_pts[i][0];
                npt[i] = shapePoints[i].size();
            }


            polylines(edge_map_instances,pts,npt,shp_num,1,Scalar(color_pallet[labeledShapes.size()-1][0],color_pallet[labeledShapes.size()-1][1],color_pallet[labeledShapes.size()-1][2]),1,8,0);
            fillPoly(region_map_instances, pts, npt, shp_num, Scalar(color_pallet[labeledShapes.size()-1][0],color_pallet[labeledShapes.size()-1][1],color_pallet[labeledShapes.size()-1][2]), 8);

            if(biMultiClass_flg){//multi-classes

                int tmp_r = color_pallet[labeledShapeClassName[labeledShapeClassName.size()-1]][0];
                int tmp_g = color_pallet[labeledShapeClassName[labeledShapeClassName.size()-1]][1];
                int tmp_b = color_pallet[labeledShapeClassName[labeledShapeClassName.size()-1]][2];

                polylines( edge_map_classes, pts, npt, shp_num, 1, Scalar(tmp_r,tmp_g,tmp_b), 1, 8, 0);
                fillPoly( region_map_classes, pts, npt, shp_num, Scalar(tmp_r,tmp_g,tmp_b), 8);

            }
            else{//bi-classes
                polylines( edge_map_classes, pts, npt, shp_num, 1, Scalar(255,255,255), 1, 8, 0);
                fillPoly( region_map_classes, pts, npt, shp_num, Scalar(255,255,255), 8);
            }

            //polylines( bw_im_edge, pts, npt, shp_num, 1, Scalar(250), 1, 8, 0) ;
            //fillPoly( bw_im_region, pts, npt, shp_num, Scalar(250), 8);
        }

    }//if(labeledShapeIdx.size()>0)
}

int inputNewShapeYN(int mbd_gt){

    int input_YN = 1;
    int txt_key = 255;
    namedWindow("New_Shape(y/n)",CV_WINDOW_AUTOSIZE);//CV_WINDOW_NORMAL
    Mat inputBoxImg = imread("./icon/start_new_shape_yn.png",1);

    string tmp_str = "";
    while(mbd_gt){

        Mat txt_im = inputBoxImg.clone();

        putText(txt_im, tmp_str, Point(620,80), CV_FONT_HERSHEY_COMPLEX, 3,
                Scalar(255,0,0), 3, 8, false);
        imshow("New_Shape(y/n)",txt_im);
        txt_key = cvWaitKey(1) & 255;

        if(89==txt_key || 121==txt_key){
            input_YN = 1;
            tmp_str = "y";

        }else if(78 == txt_key || 110 == txt_key){
            input_YN = 0;
            tmp_str = "n";

        }else if(8 == txt_key){//backspace---------------------------
            input_YN = 1;
            tmp_str = "";

        }else if(102 == txt_key){//"f" undo the last selection

            destroyWindow("New_Shape(y/n)");
            return -1;
        }

        if(10 == txt_key || 13 == txt_key){//enter--------------------------------

            destroyWindow("New_Shape(y/n)");
            return input_YN;
        }


    }
    return 1;
}


//generate distance transform map
void drawGrayEdgeFragments(Mat &tmp_frame, EdgeMap *map){
    if(map->noSegments>0){

        for(int i = 0; i < map->noSegments; i++){
            for(int j = 0; j < map->segments[i].noPixels; j++){

                int r = map->segments[i].pixels[j].r;
                int c = map->segments[i].pixels[j].c;

                tmp_frame.at<uchar>(r,c) = 100;
            }
        }
    }
}//End of function

void drawEdgeMap(Mat &tmp_frame, EdgeMap *map){
    if(map->noSegments>0){

        for(int i = 0; i < map->noSegments; i++){
            for(int j = 0; j < map->segments[i].noPixels; j++){

                int r = map->segments[i].pixels[j].r;
                int c = map->segments[i].pixels[j].c;

                Vec3b & color = tmp_frame.at<Vec3b>(r,c);
                color[0] = 0;
                color[1] = 255;
                color[2] = 0;
            }
        }
    }
}

//draw tracked boundaries
void drawBoundaries(Mat &tmp_frame,vector<vector<vector<Point> > > &shapeTemplates, vector<vector<Mat> > &Hs, vector<vector<int> > &color_pallet){

    cout<<"Debuggging--------drawBoundaries---------"<<endl;
    if(shapeTemplates.size()>0){

        Mat xy = Mat::ones(3,1,CV_64F);
        Mat hxy = Mat::ones(3,1,CV_64F);

        for(int i = 0; i < shapeTemplates.size();i++){

            for(int j = 0; j < shapeTemplates[i].size(); j++){
                for(int k = 0; k < shapeTemplates[i][j].size(); k++){

                    xy.at<double>(0,0) = double(shapeTemplates[i][j][k].x);
                    xy.at<double>(1,0) = double(shapeTemplates[i][j][k].y);

                    hxy = Hs[Hs.size()-1][i]*xy;

                    int r = int(hxy.at<double>(1,0)/hxy.at<double>(2,0));
                    int c = int(hxy.at<double>(0,0)/hxy.at<double>(2,0));

                    Vec3b & color = tmp_frame.at<Vec3b>(r,c);
                    if(i==0){//R
                        color[0] = 0;
                        color[1] = 0;
                        color[2] = 255;
                    }else if(1 == i){//G
                        color[0] = 0;
                        color[1] = 255;
                        color[2] = 0;
                    }else if(2 == i){//B
                        color[0] = 255;
                        color[1] = 0;
                        color[2] = 0;
                    }else{
                        color[0] = color_pallet[i][0];
                        color[1] = color_pallet[i][1];
                        color[2] = color_pallet[i][2];
                    }

                }//end of i
            }//end of j
        }//end of k
    }
}//End of function


void drawShapeTemplate(Mat &edge_map, vector<vector<Point> > &shapeTemplate){
    for(int i = 0; i < shapeTemplate.size(); i++){
        for(int j = 0; j < shapeTemplate[i].size(); j++){

            int r = shapeTemplate[i][j].y;
            int c = shapeTemplate[i][j].x;

            edge_map.at<uchar>(r,c) = 0;
        }
    }
}

void generateDistMapTemplate(Mat &frame, vector<vector<Point> > &shapeTemplate, Mat &dist_map, Mat &label_map){

     Mat edge_map = Mat::ones(frame.size().height, frame.size().width, CV_8UC1);

     //drawGrayEdgeFragments(edge_map, map);
     drawShapeTemplate(edge_map, shapeTemplate);

     //Mat tmp_bw, tmp_bw_revse;
     //tmp_bw_revse = 255-edge_map;
     //threshold(tmp_bw_revse, tmp_bw, 160, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

     distanceTransform(edge_map, dist_map, label_map, CV_DIST_L2, 3, CV_DIST_LABEL_PIXEL);//distance transform

     //label_map.convertTo(label_map,CV_32F);
     //normalize(label_map, label_map, 0, 1., NORM_MINMAX);
     //namedWindow("Distance Map",CV_WINDOW_NORMAL);
     //imshow("Distance Map",label_map);
     //cvWaitKey(0);
}

void generateDistMapTemplates(Mat &frame, vector<vector<vector<Point> > > &shapeTemplates, vector<Mat> &dist_maps,vector<Mat> &label_maps){
    for(int i = 0 ; i < shapeTemplates.size(); i++){
        Mat dist_map,label_map;
        generateDistMapTemplate(frame,shapeTemplates[i], dist_map, label_map);
        dist_maps.push_back(dist_map);
        label_maps.push_back(label_map);
    }
}

vector<vector<Point> > edgePixelFilter(EdgeMap *map, Mat &Dist, Mat &h){
    //Timer timer2;
    //timer2.Start();

    vector<vector<Point> > filteredEdgePoints;

    vector<EdgeFragment> EdgeFilters;//divided Edge Fragments
    //edge segments filtering according to buffer region of the prior shape
    for(int i = 0; i < map->noSegments; i++){
        int iEgStrt = -1;
        EdgeFragment EdgeFilter;
        for (int j = 0; j < map->segments[i].noPixels; j++){
            int r = map->segments[i].pixels[j].r;
            int c = map->segments[i].pixels[j].c;
            if((r >= 0) && (r < frame_height) && (c >= 0) && (c < frame_width)/* && (Dist.at<float>(r,c) < BufferThre)*/){

                if(iEgStrt==-1){
                    iEgStrt = j;
                }else{
                    if(j == (map->segments[i].noPixels-1)){
                        EdgeFilter.seg_id = i;
                        EdgeFilter.start_idx = iEgStrt;
                        EdgeFilter.end_idx = j;

                        EdgeFilters.push_back(EdgeFilter);
                        iEgStrt = -1;
                    }
                }
            }else{
                if(iEgStrt!=-1){
                    EdgeFilter.seg_id = i;
                    EdgeFilter.start_idx = iEgStrt;
                    EdgeFilter.end_idx = j - 1;

                    EdgeFilters.push_back(EdgeFilter);
                    iEgStrt = -1;
                }
            }
        }//for(int j)mapGEDT
    }//for(int i)

    vector<EdgeFragment> dividedEF;//divided Edge Fragments
    for(int i = 0; i < EdgeFilters.size(); i++){

        int newEdge = 1;// start a new small edge segment searching

        int iEgStrt = EdgeFilters[i].start_idx;
        EdgeFragment EF_tmp;
        //EF_tmp.seg_id = EdgeFilters[i].seg_id;

        float PT[3][4];

        int noPixels = EdgeFilters[i].end_idx-EdgeFilters[i].start_idx+1;

        if(noPixels < 5 && noPixels > 1){
            /*EF_tmp.seg_id = EdgeFilters[i].seg_id;
            EF_tmp.start_idx = iEgStrt;
            EF_tmp.end_idx = EdgeFilters[i].end_idx;noSegments
            dividedEF.push_back(EF_tmp);*/
        }else{

            //every two new pixels will be tested by line fitting
            for (int iEgEnd = iEgStrt + 4; iEgEnd < EdgeFilters[i].end_idx + 1;){

                if(newEdge){//start a new small edge segment searching

                    EF_tmp.seg_id = EdgeFilters[i].seg_id;
                    EF_tmp.start_idx = iEgStrt;

                    PT[0][0] = map->segments[EdgeFilters[i].seg_id].pixels[iEgStrt].r;
                    PT[1][0] = map->segments[EdgeFilters[i].seg_id].pixels[iEgStrt].c;
                    PT[2][0] = 1;
                }

                PT[0][1] = map->segments[EdgeFilters[i].seg_id].pixels[(iEgEnd+iEgStrt)/2].r;
                PT[1][1] = map->segments[EdgeFilters[i].seg_id].pixels[(iEgEnd+iEgStrt)/2].c;
                PT[2][1] = 1;

                PT[0][2] = map->segments[EdgeFilters[i].seg_id].pixels[iEgEnd-2].r;
                PT[1][2] = map->segments[EdgeFilters[i].seg_id].pixels[iEgEnd-2].c;
                PT[2][2] = 1;

                PT[0][3] = map->segments[EdgeFilters[i].seg_id].pixels[iEgEnd].r;
                PT[1][3] = map->segments[EdgeFilters[i].seg_id].pixels[iEgEnd].c;
                PT[2][3] = 1;

                if(EdgeBreakFit(PT)==1/*<DivideErrThre_e*/){

                    if(iEgEnd < EdgeFilters[i].end_idx - 1 ){
                        newEdge = 0;
                        iEgEnd +=2;
                    }else{
                        newEdge = 1;
                        EF_tmp.end_idx = EdgeFilters[i].end_idx;
                        if((EF_tmp.end_idx-EF_tmp.start_idx+1)>edgeLenThre/*EF_tmp.end_idx!=EF_tmp.start_idx*/){//
                            dividedEF.push_back(EF_tmp);// find the last small edge segment
                        }
                        break;
                    }

                }else{
                    newEdge = 1;//start a new small edge
                    EF_tmp.end_idx = iEgEnd - 2;
                    if((EF_tmp.end_idx-EF_tmp.start_idx+1)>edgeLenThre){//
                        dividedEF.push_back(EF_tmp);// find a new small edge segment
                    }//


                    iEgStrt = iEgEnd-1;
                    EF_tmp.start_idx =  iEgStrt;
                    if(EdgeFilters[i].end_idx-iEgStrt < 4){// there are less than 5 remaining pixels in this edge segment
                        EF_tmp.end_idx = EdgeFilters[i].end_idx;
                        if((EF_tmp.end_idx-EF_tmp.start_idx+1)>edgeLenThre/*EF_tmp.end_idx!=EF_tmp.start_idx*/){
                            dividedEF.push_back(EF_tmp);
                        }
                        break;
                    }else{// there are less than 3 remaining pixels in this edge segment and start a new small edge segment searching
                        iEgEnd = iEgStrt + 4;
                    }
                }//end if(LeastSquaresLineFit < DividedErrThre){} else{}

            }//for(int iEgEnd)
        }//if(noPixels < 5)
    }//for(int i)


    for(int i = 0; i < dividedEF.size(); i++){
        vector<Point> tmp_pt;
        //float distDiff = 0;
        for(int j = dividedEF[i].start_idx; j < dividedEF[i].end_idx + 1; j++){
            tmp_pt.push_back(Point(map->segments[dividedEF[i].seg_id].pixels[j].c,map->segments[dividedEF[i].seg_id].pixels[j].r));
            //if(j < dividedEF[i].end_idx){
            //    distDiff += abs(Dist.at<float>(map->segments[dividedEF[i].seg_id].pixels[j].r,map->segments[dividedEF[i].seg_id].pixels[j].c)-
            //            Dist.at<float>(map->segments[dividedEF[i].seg_id].pixels[j+1].r,map->segments[dividedEF[i].seg_id].pixels[j+1].c));
            //}

        }

        //double ratio = double(distDiff)/double(dividedEF[i].end_idx-dividedEF[i].start_idx+1);
        //double angle = asin(ratio)*180/PI;

        //cout<<"distDiff: "<<distDiff<<endl;
        //cout<<"edgeLength: "<<dividedEF[i].end_idx-dividedEF[i].start_idx+1<<endl;
        //cout<<"angle: "<<angle<<endl;
        //if(angle < angularThre){
            filteredEdgePoints.push_back(tmp_pt);
        //}
        tmp_pt.clear();
        vector<Point>().swap(tmp_pt);
    }

    return filteredEdgePoints;
}


void searchMethod(Mat &frame, vector<vector<Point> > edgeFragments, vector<vector<Point> > &shapeTemplate, Mat &dist_map, Mat &label_map, vector<Mat> &Hi){

//    Mat frame_SM = frame.clone();

    //get the distance map and label map of the template tracked from the last frame

    //Timer sm5;
    //sm5.Start();

    Mat distMap_TT, labelMap_TT;
    Mat edge_tt = Mat::ones(frame.size().height,frame.size().width,CV_8UC1);

    for(int i = 0; i < shapeTemplate.size(); i++){
        for(int j = 0; j < shapeTemplate[i].size()-1; j++){

            double x1 = double(shapeTemplate[i][j].x);
            double y1 = double(shapeTemplate[i][j].y);
            double hx1 = Hi[Hi.size()-1].at<double>(0,0)*x1 + Hi[Hi.size()-1].at<double>(0,1)*y1 + Hi[Hi.size()-1].at<double>(0,2);
            double hy1 = Hi[Hi.size()-1].at<double>(1,0)*x1 + Hi[Hi.size()-1].at<double>(1,1)*y1 + Hi[Hi.size()-1].at<double>(1,2);
            double hz1 = Hi[Hi.size()-1].at<double>(2,0)*x1 + Hi[Hi.size()-1].at<double>(2,1)*y1 + Hi[Hi.size()-1].at<double>(2,2);

            double x2 = double(shapeTemplate[i][j+1].x);
            double y2 = double(shapeTemplate[i][j+1].y);
            double hx2 = Hi[Hi.size()-1].at<double>(0,0)*x2 + Hi[Hi.size()-1].at<double>(0,1)*y2 + Hi[Hi.size()-1].at<double>(0,2);
            double hy2 = Hi[Hi.size()-1].at<double>(1,0)*x2 + Hi[Hi.size()-1].at<double>(1,1)*y2 + Hi[Hi.size()-1].at<double>(1,2);
            double hz2 = Hi[Hi.size()-1].at<double>(2,0)*x2 + Hi[Hi.size()-1].at<double>(2,1)*y2 + Hi[Hi.size()-1].at<double>(2,2);

            line(edge_tt,Point(round(hx1/hz1),round(hy1/hz1)),Point(round(hx2/hz2),round(hy2/hz2)),Scalar(0),1,8);

//            line(frame_SM,Point(round(hx1/hz1),round(hy1/hz1)),Point(round(hx2/hz2),round(hy2/hz2)),Scalar(0,255,0),1,8);
        }
    }
    distanceTransform(edge_tt, distMap_TT, labelMap_TT, CV_DIST_L2, 3, CV_DIST_LABEL_PIXEL);

    //filter and order the detected edge fragments
    vector<int> idxEF;
    for(int i = 0; i < edgeFragments.size(); i++){
        float aveOfdist = 0;
        float aveOfDD = 0;
        float last_dist = distMap_TT.at<float>(edgeFragments[i][0].y,edgeFragments[i][0].x);
        for(int j = 0; j < edgeFragments[i].size(); j++){
            aveOfdist = aveOfdist + distMap_TT.at<float>(edgeFragments[i][j].y,edgeFragments[i][j].x);
            aveOfDD = aveOfDD + abs(distMap_TT.at<float>(edgeFragments[i][j].y,edgeFragments[i][j].x) -last_dist);
            last_dist = distMap_TT.at<float>(edgeFragments[i][j].y,edgeFragments[i][j].x);
        }
        aveOfdist = aveOfdist/float(edgeFragments[i].size());
        aveOfDD = aveOfDD/(float(edgeFragments[i].size()));

        if(aveOfdist < 10 && aveOfDD < 0.8){//BufferThre
            idxEF.push_back(i);
        }
    }
    //cout<<" filtered edge fragments number: "<<idxEF.size()<<endl;

    sqrt(distMap_TT, distMap_TT);//get square root of distance map
    sqrt(distMap_TT, distMap_TT);//get square root of distance map
    //sqrt(distMap_TT, distMap_TT);//get square root of distance map

    int pixNum = 0;
    for(int i = 0; i < idxEF.size(); i++){
        pixNum += edgeFragments[idxEF[i]].size();
    }

    //============================Optimization================================
    //compute the gradient of dist_map (grad_x, grad_y)
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;

    Mat grad_x, grad_y;
    Sobel( distMap_TT, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    Sobel( distMap_TT, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

    int iteration = 0;
    float lastError = 0;

    Mat A;
    Mat b;
    Mat Iwpt;
    Mat IwptT;//IwptT = Iwpt.t();
    Mat deltap;

    Mat Iwptl;
    Mat IwptlT;//IwptlT = Iwptl.t();

    Mat IwptmlT;//

    float Distl;
    float lambda = 100;
    int splIdx = 1;

    if(pixNum>100){
        splIdx = pixNum/100;//sample edge pixels with interval splIdx
    }

    //cout<<"pixNum: "<<pixNum<<endl;
    //cout<<"splIdx: "<<splIdx<<endl;

    Mat tmpH = Mat::eye(3,3,CV_64F);

    for(;;){
        //Timer ite_time;
        //ite_time.Start();

 //       Mat frame_opt = frame_SM.clone();

        A = Mat::zeros(8,8,CV_64F);
        b = Mat::zeros(8,1,CV_64F);
        Iwpt = Mat::zeros(8,1,CV_64F);
        IwptT = Mat::zeros(1,8,CV_64F);//
        IwptlT = Mat::zeros(1,8,CV_64F);//
        IwptmlT = Mat::zeros(8,1,CV_64F);//
        deltap = Mat::zeros(8,1,CV_64F);

        Iwptl = Mat::zeros(8,1,CV_64F);
        Distl = 0;

        for(int i = 0; i < idxEF.size(); i++){

            for(int j = 0; j < edgeFragments[idxEF[i]].size(); j+=splIdx){

                double x = double(edgeFragments[idxEF[i]][j].x);
                double y = double(edgeFragments[idxEF[i]][j].y);

                double hx = tmpH.at<double>(0,0)*x + tmpH.at<double>(0,1)*y + tmpH.at<double>(0,2);
                double hy = tmpH.at<double>(1,0)*x + tmpH.at<double>(1,1)*y + tmpH.at<double>(1,2);
                double hz = tmpH.at<double>(2,0)*x + tmpH.at<double>(2,1)*y + tmpH.at<double>(2,2);

                double xi = double(hx/hz);
                double yi = double(hy/hz);


                double p78 = 1 + tmpH.at<double>(2,0)*xi + tmpH.at<double>(2,1)*yi;

                Iwpt.at<double>(0,0) = xi*grad_x.at<float>(int(yi), int(xi));
                Iwpt.at<double>(1,0) = xi*grad_y.at<float>(int(yi), int(xi));
                Iwpt.at<double>(2,0) = yi*grad_x.at<float>(int(yi), int(xi));
                Iwpt.at<double>(3,0) = yi*grad_y.at<float>(int(yi), int(xi));
                Iwpt.at<double>(4,0) = grad_x.at<float>(int(yi), int(xi));
                Iwpt.at<double>(5,0) = grad_y.at<float>(int(yi), int(xi));

                Iwpt.at<double>(6,0) = (-1*xi*(tmpH.at<double>(0,0)*xi + tmpH.at<double>(0,1)*yi + tmpH.at<double>(0,2))*grad_x.at<float>(round(yi), round(xi))
                                        -xi*(tmpH.at<double>(1,0)*xi + tmpH.at<double>(1,1)*yi + tmpH.at<double>(1,2))*grad_y.at<float>(round(yi), round(xi)))/p78;
                Iwpt.at<double>(7,0) = (-1*yi*(tmpH.at<double>(0,0)*xi + tmpH.at<double>(0,1)*yi + tmpH.at<double>(0,2))*grad_x.at<float>(round(yi), round(xi))
                                        -yi*(tmpH.at<double>(1,0)*xi + tmpH.at<double>(1,1)*yi + tmpH.at<double>(1,2))*grad_y.at<float>(round(yi), round(xi)))/p78;

                Iwpt = Iwpt/p78;
                IwptT = Iwpt.t();

                //A = A + Iwpt*Iwpt.t();
                A += Iwpt*IwptT;
                b += Iwpt*distMap_TT.at<float>(int(yi+0.5), int(xi+0.5));

                if(j>0){


                    A += lambda*(Iwpt-Iwptl)*(IwptT-IwptlT);
                    b += lambda*(Iwpt-Iwptl)*(distMap_TT.at<float>(int(yi+0.5), int(xi+0.5)) - Distl);


                }

                Iwptl = Iwpt;
                IwptlT = IwptT;
                Distl = distMap_TT.at<float>(int(yi+0.5), int(xi+0.5));


                //}
            }//j
        }//i


        A = -A;
        //SVD::solveZ(A,deltap);
        solve(A,b,deltap,DECOMP_QR);

        tmpH.at<double>(0,0) += deltap.at<double>(0,0);
        tmpH.at<double>(0,1) += deltap.at<double>(2,0);
        tmpH.at<double>(0,2) += deltap.at<double>(4,0);
        tmpH.at<double>(1,0) += deltap.at<double>(1,0);
        tmpH.at<double>(1,1) += deltap.at<double>(3,0);
        tmpH.at<double>(1,2) += deltap.at<double>(5,0);
        tmpH.at<double>(2,0) += deltap.at<double>(6,0);
        tmpH.at<double>(2,1) += deltap.at<double>(7,0);

        //transfor edge pixels by estimated homography
        float distance_error = 0;

        int px_num = 0;
        for(int i = 0; i < idxEF.size(); i++){
            for(int j = 0; j < edgeFragments[idxEF[i]].size(); j++){

                px_num++;

                double x = double(edgeFragments[idxEF[i]][j].x);
                double y = double(edgeFragments[idxEF[i]][j].y);

                double hx = tmpH.at<double>(0,0)*x + tmpH.at<double>(0,1)*y + tmpH.at<double>(0,2);
                double hy = tmpH.at<double>(1,0)*x + tmpH.at<double>(1,1)*y + tmpH.at<double>(1,2);
                double hz = tmpH.at<double>(2,0)*x + tmpH.at<double>(2,1)*y + tmpH.at<double>(2,2);

                int r = hy/hz;
                int c = hx/hz;

                if(r>=0 && r < frame_height && c >= 0 && c < frame_width){

                    distance_error += distMap_TT.at<float>(r,c);

                }

            }
        }

        iteration++;
        double aveErr = distance_error/px_num;
 //       cout<<"iter: "<<iteration<<" total error: "<< distance_error<<" pixels number: "<<px_num<<" error: "<<aveErr<<" dif: "<<abs(aveErr - lastError)<<endl;
        if( abs(aveErr - lastError) < 1e-6 || iteration > 30){
            break;
        }
        lastError = aveErr;

    }//for(;;)

    if(norm(tmpH.inv()) < 30){
        Hi[Hi.size()-1] = tmpH.inv()*Hi[Hi.size()-1];
    }


}//searchMethod


void shapeTracker(Mat &frame, EdgeMap *map, vector<vector<vector<Point> > > &shapeTemplates, vector<Mat> &dist_maps, vector<Mat> &label_maps, vector<vector<Mat> > &Hs){

    vector<vector<Point> > edgeFragments;
    //edgeFragments = edgePixelFilter(map, dist_maps[i]);//divide edge segments into fragments

    vector<Mat> Hi;
    for(unsigned int i = 0; i < shapeTemplates.size(); i++){//multiple independent edge tamplates tracking
        Hi.push_back(Hs[Hs.size()-1][i]);

        Timer time_split;
        time_split.Start();

        edgeFragments = edgePixelFilter(map, dist_maps[i], Hi[Hi.size()-1]);//divide edge segments into fragment

        time_split.Stop();
        edge_splitting_time += time_split.ElapsedTime();

        Timer timer_SM;
        timer_SM.Start();

        searchMethod(frame, edgeFragments, shapeTemplates[i], dist_maps[i], label_maps[i], Hi);

        timer_SM.Stop();
        sm_time += timer_SM.ElapsedTime();

        //total_time = total_time + edge_detction_time + edge_splitting_time + sm_time;
        //cout<<" time of tracking: "<<timer.ElapsedTime()<<" ms"<<endl;
    }
    Hs.push_back(Hi);
    Hi.clear();
    vector<Mat>().swap(Hi);
}
