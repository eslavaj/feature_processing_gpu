/*
 * translation_calculator.hpp
 *
 *  Created on: Mar 21, 2020
 *      Author: jeslava
 */

#ifndef SRC_DISPLACEMENT_CALCULATOR_HPP_
#define SRC_DISPLACEMENT_CALCULATOR_HPP_

#include <vector>
#include <opencv2/core.hpp>


class displacement_calculator
{

public:
	displacement_calculator(){};

	void calc_displacements(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, std::vector<cv::DMatch> &matches,
							std::vector<cv::Point2f> & displacements);

	void calc_displacements1(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, std::vector<cv::DMatch> &matches,
							std::vector<cv::Point2f> & displacements);

	void calc_displacements2(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, std::vector<cv::DMatch> &matches,
							std::vector<cv::Point2f> & displacements);

};








#endif /* SRC_DISPLACEMENT_CALCULATOR_HPP_ */
