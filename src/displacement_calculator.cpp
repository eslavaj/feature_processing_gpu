/*
 * displacement_calculator.cpp
 *
 *  Created on: Mar 21, 2020
 *      Author: jeslava
 */




#include "displacement_calculator.hpp"
#include <stdio.h>
#include <iostream>
#include <math.h>


void displacement_calculator::calc_displacements(std::vector<cv::KeyPoint> &kPtsQuery, std::vector<cv::KeyPoint> &kPtsTrain, std::vector<cv::DMatch> &matches,
							std::vector<cv::Point2f> & displacements)
{

	for(auto match : matches)
	{
		float delta_x = kPtsQuery[match.queryIdx].pt.x - kPtsTrain[match.trainIdx].pt.x;
		float delta_y = kPtsQuery[match.queryIdx].pt.y - kPtsTrain[match.trainIdx].pt.y;
		float delta_ratio = delta_y/delta_x;

		displacements.push_back(cv::Point2f(delta_x, delta_y));

		//std::cout<<"delta_x: "<< delta_x << "  delta_y: " << delta_y << "  delta_ratio: "<< delta_ratio << std::endl;

	}
}




void displacement_calculator::calcRelatVertDisplacement(std::vector<cv::KeyPoint> &kPtsQuery, std::vector<cv::KeyPoint> &kPtsTrain, std::vector<cv::DMatch> &matches,
							       std::vector<float> & relVertDisplacement)
{
	float gCenterQuery_x = 0;
	float gCenterQuery_y = 0;
	float gCenterTrain_x = 0;
	float gCenterTrain_y = 0;

	/*Calculate gravity center of matched keypoints*/
	for(auto match : matches)
	{
		gCenterQuery_x = gCenterQuery_x + kPtsQuery[match.queryIdx].pt.x;
		gCenterQuery_y = gCenterQuery_y + kPtsQuery[match.queryIdx].pt.y;

		gCenterTrain_x = gCenterTrain_x + kPtsTrain[match.trainIdx].pt.x;
		gCenterTrain_y = gCenterTrain_y + kPtsTrain[match.trainIdx].pt.y;
		//std::cout<<"delta_x: "<< delta_x << "  delta_y: " << delta_y << "  delta_ratio: "<< delta_ratio << std::endl;
	}

	gCenterQuery_x = gCenterQuery_x/matches.size();
	gCenterQuery_y = gCenterQuery_y/matches.size();
	gCenterTrain_x = gCenterTrain_x/matches.size();
	gCenterTrain_y = gCenterTrain_y/matches.size();

	/*Calculate distance to center of gravity ratio*/
	for(auto match : matches)
	{
		float distQuerySqr =  powf(kPtsQuery[match.queryIdx].pt.x - gCenterQuery_x, 2) + powf(kPtsQuery[match.queryIdx].pt.y - gCenterQuery_y, 2);
		float distTrainSqr =  powf(kPtsTrain[match.trainIdx].pt.x - gCenterTrain_x, 2) + powf(kPtsTrain[match.trainIdx].pt.y - gCenterTrain_y, 2);
		float relVertRatio = sqrtf(distQuerySqr/distTrainSqr);
		relVertDisplacement.push_back(relVertRatio);
		std::cout<<"vertical displaement ratio: "<< relVertRatio << std::endl;
	}

}


void displacement_calculator::calc_displacements1(std::vector<cv::KeyPoint> &kPtsQuery, std::vector<cv::KeyPoint> &kPtsTrain, std::vector<cv::DMatch> &matches,
							std::vector<cv::Point2f> & displacements)
{

	for(auto match : matches)
	{
		float delta_x = kPtsQuery[match.queryIdx].pt.x/0.8 - kPtsTrain[match.trainIdx].pt.x/0.6;
		float delta_y = kPtsQuery[match.queryIdx].pt.y/0.8 - kPtsTrain[match.trainIdx].pt.y/0.6;
		float delta_ratio = delta_y/delta_x;

		displacements.push_back(cv::Point2f(delta_x, delta_y));

		std::cout<<"delta_x: "<< delta_x << "  delta_y: " << delta_y << "  delta_ratio: "<< delta_ratio << std::endl;

	}
}

void displacement_calculator::calc_displacements2(std::vector<cv::KeyPoint> &kPtsQuery, std::vector<cv::KeyPoint> &kPtsTrain, std::vector<cv::DMatch> &matches,
							std::vector<cv::Point2f> & displacements)
{

	for(auto match : matches)
	{
		float delta_x = kPtsQuery[match.queryIdx].pt.x/0.8 - kPtsTrain[match.trainIdx].pt.x/0.6;
		float delta_y = kPtsQuery[match.queryIdx].pt.y/0.8 - kPtsTrain[match.trainIdx].pt.y/0.6;
		float delta_ratio = delta_y/delta_x;

		displacements.push_back(cv::Point2f(delta_x, delta_y));

		std::cout<<"delta_x: "<< delta_x << "  delta_y: " << delta_y << "  delta_ratio: "<< delta_ratio << std::endl;

	}
}
