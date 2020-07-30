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

/* This function does not work*/
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

/* This function does not work*/
void displacement_calculator::calcDisplacWithVertCorr(std::vector<cv::KeyPoint> &kPtsQuery, std::vector<cv::KeyPoint> &kPtsTrain, std::vector<cv::DMatch> &matches,
							std::vector<cv::Point2f> & displacements, float relVertDisplacement)
{


	for(auto match : matches)
	{
		float delta_x = (kPtsQuery[match.queryIdx].pt.x -300)/relVertDisplacement - (kPtsTrain[match.trainIdx].pt.x - 300);
		float delta_y = (kPtsQuery[match.queryIdx].pt.y + 300)/relVertDisplacement - (kPtsTrain[match.trainIdx].pt.y + 300);
		float delta_ratio = delta_y/delta_x;

		displacements.push_back(cv::Point2f(delta_x, delta_y));

		//std::cout<<"delta_x: "<< delta_x << "  delta_y: " << delta_y << "  delta_ratio: "<< delta_ratio << std::endl;

	}
}


void displacement_calculator::calcRelatVertDisplacement(std::vector<cv::KeyPoint> &kPtsQuery, std::vector<cv::KeyPoint> &kPtsTrain, std::vector<cv::DMatch> &matches,
							       float & vertRatioMedia)
{
	/*TODO: optimize this function, currently vertical relative displacement are computed twice*/
	/*TODO: Manage operations with results close or equal to zero/infinite */

	float gCenterQuery_x = 0;
	float gCenterQuery_y = 0;
	float gCenterTrain_x = 0;
	float gCenterTrain_y = 0;
	std::vector<float> ratios_tmp;

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
		ratios_tmp.push_back(relVertRatio);
		//std::cout<<"vertical displacement ratio: "<< relVertRatio << std::endl;
	}

	/*Calculate median*/
	std::sort(ratios_tmp.begin(), ratios_tmp.end());
	int ratios_nbr = ratios_tmp.size();
	if(ratios_nbr%2 == 0)
	{
		vertRatioMedia = (ratios_tmp[ratios_nbr/2] + ratios_tmp[ratios_nbr/2+1])/2;
	}
	else
	{
		vertRatioMedia = ratios_tmp[ int(ratios_nbr/2) + 1];
	}

	std::cout<<" vertical displacement ratio: "<< vertRatioMedia << std::endl;

}


/* This function does not work*/
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


/* This function does not work*/
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
