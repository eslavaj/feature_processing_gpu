/*
 * KeypointProcessorGpu.cpp
 *
 *  Created on: Jul 29, 2020
 *      Author: jeslava
 */


#include <vector>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


#include "KeypointProcessorGpu.hpp"


using namespace std;


void KeypointProcessorGpu::extractKpointDescriptors(cv::Mat & newImage)
{
	/*To measure time*/
	double t;

	/*Convert to grayscale*/
    cv::Mat imgGray;
    cv::cvtColor(newImage, imgGray, cv::COLOR_BGR2GRAY);

    DataFrame frame;
    frame.cameraImg = imgGray;
    frame.gpu_cameraImg.upload(imgGray);
    m_dataFrameBuffer.push_back(frame);

    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

    vector<cv::KeyPoint> keypoints; // create empty feature list for current image
    cv::Mat descriptors; // create empty feature list for current image

    cv::cuda::GpuMat gkeypoints; // this holds the keys detected
    cv::cuda::GpuMat gdescriptors; // this holds the descriptors for the detected keypoints
    cv::cuda::GpuMat mask1; // this holds any mask you may want to use, or can be replace by noArray() in the call below if no mask is needed
    cv::cuda::Stream istream;

    if(m_detectorType.compare("ORB"))
    {
    	/*ORB*/
        cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(250, 1.2f, 8, 31, 0, 2, 0, 31, 20, true);
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
    (m_dataFrameBuffer.end() - 1)->keypoints = keypoints;
    (m_dataFrameBuffer.end() - 1)->gpu_keypoints = gkeypoints;

    // push descriptors for current frame to end of data buffer
    (m_dataFrameBuffer.end() - 1)->gpu_descriptors = gdescriptors;


}


void KeypointProcessorGpu::matchKpoints()
{
	/*To measure time*/
	double t;
	cv::cuda::Stream istream;

    if (m_dataFrameBuffer.size() > 1) // wait until at least two images have been processed
    {
        cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        cv::cuda::GpuMat gmatches;
        vector< cv::DMatch > matches;

        t = (double)cv::getTickCount();

        /* MATCH KEYPOINT DESCRIPTORS */
        if (m_selectorType.compare("SEL_KNN") == 0)
        {
        	cv::cuda::GpuMat gknnMatches;
        	vector<vector<cv::DMatch>> knnMatches;
        	matcher->knnMatchAsync( (m_dataFrameBuffer.end() - 2)->gpu_descriptors, (m_dataFrameBuffer.end() - 1)->gpu_descriptors, gknnMatches, 6, cv::noArray(), istream);
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
        	matcher->matchAsync( (m_dataFrameBuffer.end() - 2)->gpu_descriptors, (m_dataFrameBuffer.end() - 1)->gpu_descriptors, gmatches, cv::noArray(), istream);
        	matcher->matchConvert(gmatches, matches);
        }

        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "#3 : MATCH KEYPOINT DESCRIPTORS done " << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        /*calculate displacement*/
        /*
        displacement_calculator displ_calc;
        vector<cv::Point2f> displacements;
        double t = (double)cv::getTickCount();
        displ_calc.calc_displacements((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,matches, displacements);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "******* displacement calculation done in " << 1000 * t / 1.0 << " ms" << endl;
         */

        // store matches in current data frame
        (m_dataFrameBuffer.end() - 1)->kptMatches = matches;
        (m_dataFrameBuffer.end() - 1)->gpu_kptMatches = gmatches;

        // visualize matches between current and previous image
        if (m_visuEnable)
        {
            cv::Mat matchImg = ((m_dataFrameBuffer.end() - 1)->cameraImg).clone();

            cv::drawMatches((m_dataFrameBuffer.end() - 2)->cameraImg, (m_dataFrameBuffer.end() - 2)->keypoints,
                            (m_dataFrameBuffer.end() - 1)->cameraImg, (m_dataFrameBuffer.end() - 1)->keypoints,
                            matches, matchImg,
                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            string windowName = "Matching keypoints between two camera images";
            cv::namedWindow(windowName, 7);
            cv::imshow(windowName, matchImg);
            cout << "Press key to continue to next image" << endl;
            cv::waitKey(0); // wait for key to be pressed
        }
    }
}




