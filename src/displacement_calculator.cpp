/*
 * displacement_calculator.cpp
 *
 *  Created on: Mar 21, 2020
 *      Author: jeslava
 */




#include "displacement_calculator.hpp"
#include <stdio.h>
#include <iostream>


void displacement_calculator::calc_displacements(std::vector<cv::KeyPoint> &kPtsQuery, std::vector<cv::KeyPoint> &kPtsTrain, std::vector<cv::DMatch> &matches,
							std::vector<cv::Point2f> & displacements)
{

	for(auto match : matches)
	{
		float delta_x = kPtsQuery[match.queryIdx].pt.x - kPtsTrain[match.trainIdx].pt.x;
		float delta_y = kPtsQuery[match.queryIdx].pt.y - kPtsTrain[match.trainIdx].pt.y;
		float delta_ratio = delta_y/delta_x;

		displacements.push_back(cv::Point2f(delta_x, delta_y));

		std::cout<<"delta_x: "<< delta_x << "  delta_y: " << delta_y << "  delta_ratio: "<< delta_ratio << std::endl;

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
