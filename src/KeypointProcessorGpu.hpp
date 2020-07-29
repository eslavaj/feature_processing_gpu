/*
 * KeypointProcessorGpu.hpp
 *
 *  Created on: Jul 29, 2020
 *      Author: jeslava
 */

#ifndef KEYPOINTPROCESSORGPU_HPP_
#define KEYPOINTPROCESSORGPU_HPP_

#include <string>

#include <boost/circular_buffer.hpp>

#include <opencv2/core/cuda.hpp>

#include "dataStructures.h"


class KeypointProcessorGpu
{

public:
	KeypointProcessorGpu(boost::circular_buffer<DataFrame> &dataFrameBuffer, std::string detectorType, std::string selectorType, bool visuEnable):
						m_dataFrameBuffer(dataFrameBuffer),
						m_detectorType(detectorType),
						m_selectorType(selectorType),
						m_visuEnable(visuEnable)
						{};

	//virtual ~KeypointProcessorGpu();

	void extractKpointDescriptors(cv::Mat & newImage);
	void matchKpoints(std::string mpointStrategy="FUND");
	void refineMatches(const std::vector<cv::DMatch>& matches,
		                 std::vector<cv::KeyPoint>& keypoints1,
						 std::vector<cv::KeyPoint>& keypoints2,
					     std::vector<cv::DMatch>& outMatches,
						 std::string matchRefineStrategy);


private:
	boost::circular_buffer<DataFrame> & m_dataFrameBuffer;
	std::string m_detectorType;
	std::string m_selectorType;
	bool m_visuEnable;
	cv::Mat m_fundMatrix;
	cv::Mat m_homographyMatrix;
	double m_distToEpipLine = 1.0;
	double m_ransacConfid = 0.98;
	bool m_refineFund = true; /*Refine fundamental matrix*/
	bool m_refineMatches = true; /*Refine the matches*/



};






#endif /* KEYPOINTPROCESSORGPU_HPP_ */
