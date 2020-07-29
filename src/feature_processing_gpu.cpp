//============================================================================
// Name        : feature_processing_gpu.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description :
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
#include "KeypointProcessorGpu.hpp"

using namespace cv;
using namespace std;

int main(int argc, const char *argv[])
{
	int num_args = argc;
	if(num_args!=5)
	{
		cout<<"Incorrect argument number"<<endl<<endl;
		cout<<"Usage: "<<endl;
		cout<<"      feature_processing_gpu <Match selector type> <image_folder>"<<endl;
		cout<<"Detector types: FAST , ORB"<<endl;
		cout<<"Selector types: SEL_NN , SEL_KNN"<<endl;
		cout<<"Match point refining strategy: FUND , HOMOGR, AUTO, NONE"<<endl;
		cout<<"image_folder: the name of your image folder"<<endl;
		return -1;
	}

	string detectorType = argv[1];
	string selectorType = argv[2];
	string mpointStrat = argv[3];
	string img_folder = argv[4];

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
    KeypointProcessorGpu pointProcGPU(dataBuffer, detectorType, selectorType, true);

    /* MAIN LOOP OVER ALL IMAGES */
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
    	// assemble filenames for current index
    	ostringstream imgNumber;
    	imgNumber << imgStartIndex + imgIndex;
    	string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
    	cout <<"image full name: "<<imgFullFilename<< endl;

    	cv::Mat img;
    	img = cv::imread(imgFullFilename);

    	pointProcGPU.extractKpointDescriptors(img);
    	pointProcGPU.matchKpoints(mpointStrat);

    } // eof loop over all images

    return 0;
}
