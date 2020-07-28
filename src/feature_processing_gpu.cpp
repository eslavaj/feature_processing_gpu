//============================================================================
// Name        : project_base.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

#include <boost/circular_buffer.hpp>

#include "displacement_calculator.hpp"

using namespace cv;
using namespace std;

int main(int argc, const char *argv[])
{

	int num_args = argc;
	if(num_args!=4)
	{
		cout<<"Incorrect argument number"<<endl<<endl;
		cout<<"Usage: "<<endl;
		cout<<"      feature_processing_gpu <Match selector type> <image_folder>"<<endl;
		cout<<"Detector types: FAST , ORB"<<endl;
		cout<<"Selector types: SEL_NN , SEL_KNN"<<endl;
		cout<<"image_folder: the name of your image folder"<<endl;
		return -1;
	}

	string detectorType = argv[1];
	string selectorType = argv[2];
	string img_folder = argv[3];

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = img_folder + "/"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    //int imgEndIndex = 9;   // last file index to load
    int imgEndIndex = 24;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time

    boost::circular_buffer<DataFrame> dataBuffer(dataBufferSize);
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    double t;

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        //imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        imgNumber << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        cout <<"image full name: "<<imgFullFilename<< endl;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        DataFrame frame;
        frame.cameraImg = imgGray;
        frame.gpu_cameraImg.upload(imgGray);
        dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image

        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        cv::Mat descriptors; // create empty feature list for current image


        cv::cuda::GpuMat gkeypoints; // this holds the keys detected
        cv::cuda::GpuMat gdescriptors; // this holds the descriptors for the detected keypoints
        cv::cuda::GpuMat mask1; // this holds any mask you may want to use, or can be replace by noArray() in the call below if no mask is needed
        cv::cuda::Stream istream;

        if(detectorType.compare("ORB"))
        {
        	/*ORB*/
            cv::Ptr<cv::cuda::ORB> orb = cuda::ORB::create(250, 1.2f, 8, 31, 0, 2, 0, 31, 20, true);
            t = (double)cv::getTickCount();
            orb->detectAndComputeAsync(frame.gpu_cameraImg, mask1, gkeypoints, gdescriptors, false, istream);
            istream.waitForCompletion();
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            orb->convert(gkeypoints, keypoints);
            cout << "#2 : DETECT KEYPOINTS and DESCRIPTORS done " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        }
        else
        {
        	/*FAST*/
        	int threshold = 30;
        	cv::Ptr<cv::cuda::FastFeatureDetector> fastdetector = cv::cuda::FastFeatureDetector::create(threshold, true, cv::FastFeatureDetector::TYPE_9_16);
        	t = (double)cv::getTickCount();
        	fastdetector->detectAndComputeAsync(frame.gpu_cameraImg, mask1, gkeypoints, gdescriptors, false, istream);
        	istream.waitForCompletion();
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        	fastdetector->convert(gkeypoints, keypoints);
        	cout << "#2 : DETECT KEYPOINTS and DESCRIPTORS done " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
        }


        gdescriptors.download(descriptors);

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        (dataBuffer.end() - 1)->gpu_keypoints = gkeypoints;

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->gpu_descriptors = gdescriptors;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            cv::Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
            cv:cuda::GpuMat gmatches;
            vector< DMatch > matches;

            t = (double)cv::getTickCount();

            /* MATCH KEYPOINT DESCRIPTORS */
            if (selectorType.compare("SEL_KNN") == 0)
            {
            	cv::cuda::GpuMat gknnMatches;
            	vector<vector<DMatch>> knnMatches;
            	matcher->knnMatchAsync( (dataBuffer.end() - 2)->gpu_descriptors, (dataBuffer.end() - 1)->gpu_descriptors, gknnMatches, 2, noArray(), istream);
            	matcher->knnMatchConvert(gknnMatches, knnMatches);
            	double minDescDistRatio = 0.8;
            	for(auto it = knnMatches.begin(); it!=knnMatches.end(); it++)
            	{
            		if((*it)[0].distance < minDescDistRatio*( (*it)[1].distance ) )
            		{
            			matches.push_back((*it)[0]);
            		}
            	}
            }
            else
            {
            	matcher->matchAsync( (dataBuffer.end() - 2)->gpu_descriptors, (dataBuffer.end() - 1)->gpu_descriptors, gmatches, noArray(), istream);
            	matcher->matchConvert(gmatches, matches);
            }


            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            cout << "#3 : MATCH KEYPOINT DESCRIPTORS done " << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;


            /*calculate displacement*/
            displacement_calculator displ_calc;
            vector<cv::Point2f> displacements;
            double t = (double)cv::getTickCount();
            displ_calc.calc_displacements((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,matches, displacements);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            cout << "******* displacement calculation done in " << 1000 * t / 1.0 << " ms" << endl;

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
            (dataBuffer.end() - 1)->gpu_kptMatches = gmatches;


            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();

                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}
