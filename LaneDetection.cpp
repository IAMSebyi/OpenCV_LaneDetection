#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>


cv::Mat perspectiveMapping(cv::Mat image, std::vector<cv::Point> cameraLanePoints, bool inverse);
cv::Mat applyThreshold(cv::Mat image);
std::pair<int, int> getLinesStartingPosition(cv::Mat image);
std::vector<cv::Point> getLanePoints(cv::Mat image, std::pair<int, int> linesStartingPosition, int& leftPointsCount, int& rightPointsCount);
float getRelativeDistanceToLaneCenter(std::vector<cv::Point> lanePoints, int leftPointsCount, int rightPointsCount);
void displayLane(cv::Mat image, std::vector<cv::Point> cameraLanePoints, std::vector<cv::Point> lanePoints);
cv::Mat detectLaneImage(cv::Mat image);


int main() {
	uint16_t testcase;
	std::cout << "Please input 1 for image test or 2 for video test: ";
	std::cin >> testcase;

	switch (testcase) {
	case 1: {
		// Read image
		cv::Mat image = cv::imread("Images/image1.jpg");
		cv::resize(image, image, cv::Size(1280, 720));
		
		cv::Mat resultImage = detectLaneImage(image);

		cv::imshow("Image", resultImage);
		cv::waitKey(0);

		break;
	}
	case 2: {
		// Read video
		int videoId = 0;
		while (!(videoId >= 1 && videoId <= 3)) {
			std::cout << "Please choose the video ID (a number between 1 and 3): ";
			std::cin >> videoId;
		}

		cv::VideoCapture cap("Videos/video" + std::to_string(videoId) + ".mp4");
		while (cap.isOpened()) {
			cv::Mat frame;
			cap.read(frame);
			cv::resize(frame, frame, cv::Size(1280, 720));

			cv::Mat resultFrame = detectLaneImage(frame);

			cv::imshow("Video", resultFrame);
			if (cv::waitKey(10) == 'q') break;
		}

		cap.release();
		cv::destroyAllWindows();

		break; 
	}
	default:
		std::cout << "Fair enough." << std::endl;

		break;
	}
	
	return 0;
}



cv::Mat perspectiveMapping(cv::Mat image, std::vector<cv::Point> cameraLanePoints, bool inverse) {
	cv::Mat imageWarp;

	std::vector<cv::Point2f> src = { cv::Point2f(cameraLanePoints[0].x, cameraLanePoints[0].y), cv::Point2f(cameraLanePoints[3].x, cameraLanePoints[3].y),
		cv::Point2f(cameraLanePoints[1].x, cameraLanePoints[1].y), cv::Point2f(cameraLanePoints[2].x, cameraLanePoints[2].y) };
	
	std::vector<cv::Point2f> dest = { cv::Point2f(200, 0), cv::Point2f(200, 720), cv::Point2f(1080, 0), cv::Point2f(1080, 720) };

	cv::Mat perspectiveMatrix;
	if (inverse) perspectiveMatrix = cv::getPerspectiveTransform(src, dest);
	else perspectiveMatrix = cv::getPerspectiveTransform(dest, src);

	cv::warpPerspective(image, imageWarp, perspectiveMatrix, cv::Size(1280, 720));

	return imageWarp;
}



cv::Mat applyThreshold(cv::Mat image) {
	cv::Mat imageThres, redChannel, sobel;
	
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	redChannel = channels[2];
	cv::GaussianBlur(redChannel, redChannel, cv::Size(5, 5), 0);
	cv::threshold(redChannel, redChannel, 190, 255, cv::THRESH_BINARY);

	return redChannel;
}



std::pair<int, int> getLinesStartingPosition(cv::Mat image) {
	cv::Size size = image.size();

	// Make histogram
	std::vector<int> histogram(1280, 0);
	for (int x = 0; x < size.width; x++) {
		for (int y = 0; y < size.height; y++) {
			if (image.at<uchar>(y, x) == 255) histogram[x]++;
		}
	}

	// Get the probable start position of the left line and right line
	int leftLineP = 0, rightLineP = 1279, leftMax = -1, rightMax = -1;
	for (int i = 0; i < 640; i++) {
		if (histogram[i] >= leftMax) {
			leftMax = histogram[i];
			leftLineP = i;
		}
	}

	for (int i = 640; i < 1280; i++) {
		if (histogram[i] >= rightMax) {
			rightMax = histogram[i];
			rightLineP = i;
		}
	}

	return std::pair<int, int>(leftLineP, rightLineP);
}



std::vector<cv::Point> getLanePoints(cv::Mat image, std::pair<int, int> linesStartingPosition, int& leftPointsCount, int& rightPointsCount) {
	int y = 720;

	std::pair<int, int> linesPosition = linesStartingPosition;
	std::vector<cv::Point> lanePoints;

	while (y >= 50) {
		cv::Mat imageCrop;

		// Left line
		imageCrop = image(cv::Rect(cv::Point2i(std::max(linesPosition.first - 50, 0), y - 50), cv::Point2i(std::min(linesPosition.first + 50, 1280), y)));

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(imageCrop, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
		bool foundLeftLanePoint = false, foundRightLanePoint = false; // If there was no point pushed in the lanePoints vector during contour iteration, use previous line position value

		for (auto contour : contours) {
			cv::Moments moments = cv::moments(contour);
			if (moments.m00 != 0) {
				int cx = moments.m10 / moments.m00;
				int cy = moments.m01 / moments.m00;

				linesPosition.first = linesPosition.first - 50 + cx;
				lanePoints.push_back(cv::Point(linesPosition.first, y));
				foundLeftLanePoint = true;
				leftPointsCount++;
			}
		}
		if (!foundLeftLanePoint) { lanePoints.push_back(cv::Point(linesPosition.first, y)); leftPointsCount++; }

		// Right line
		imageCrop = image(cv::Rect(cv::Point2i(std::max(linesPosition.second - 50, 0), y - 50), cv::Point2i(std::min(linesPosition.second + 50, 1280), y)));

		contours.clear();
		cv::findContours(imageCrop, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		for (auto contour : contours) {
			cv::Moments moments = cv::moments(contour);
			if (moments.m00 != 0) {
				int cx = moments.m10 / moments.m00;
				int cy = moments.m01 / moments.m00;

				linesPosition.second = linesPosition.second - 50 + cx;
				lanePoints.push_back(cv::Point(linesPosition.second, y));
				foundRightLanePoint = true;
				rightPointsCount++;
			}
		}
		if (!foundRightLanePoint) { lanePoints.push_back(cv::Point(linesPosition.second, y)); rightPointsCount++; }


		cv::rectangle(image, cv::Rect(cv::Point2i(linesPosition.first - 50, y - 50), cv::Point2i(linesPosition.first + 50, y)), cv::Scalar(255, 255, 255), 3);
		cv::rectangle(image, cv::Rect(cv::Point2i(linesPosition.second - 50, y - 50), cv::Point2i(linesPosition.second + 50, y)), cv::Scalar(255, 255, 255), 3);

		y -= 50;
	}

	return lanePoints;
}



float getRelativeDistanceToLaneCenter(std::vector<cv::Point> lanePoints, int leftPointsCount, int rightPointsCount) {
	float averageLeftPosition = 0, averageRightPosition = 0;

	for (int i = 0; i < leftPointsCount + rightPointsCount - 1; i++) {
		if (i < leftPointsCount) averageLeftPosition += lanePoints[i].x;
		else averageRightPosition += lanePoints[i].x;
	}

	averageLeftPosition /= leftPointsCount;
	averageRightPosition /= rightPointsCount;

	return ( ( averageLeftPosition + averageRightPosition ) / 2.0f - 640.0f ) / 10.0f;
}



void displayLane(cv::Mat image, std::vector<cv::Point> cameraLanePoints, std::vector<cv::Point> lanePoints) {
	cv::Mat laneImage = cv::Mat::zeros(cv::Size(1280, 720), CV_8UC3);

	std::vector<cv::Point> hullPoints;
	cv::convexHull(lanePoints, hullPoints);
	cv::fillPoly(laneImage, hullPoints, cv::Scalar(0, 255, 0));

	laneImage = perspectiveMapping(laneImage, cameraLanePoints, false);
	
	double opacity = 0.50;
	cv::addWeighted(image, 1, laneImage, 1 - opacity, 0, image);
}



cv::Mat detectLaneImage(cv::Mat image) {
	// Different camera positions on the car result different perspectives of the road.
	// If testing using your own or other videos, please change the variables of the points in cameraLanePoints vector.
	std::vector<cv::Point> cameraLanePoints = { cv::Point(565, 450), cv::Point(735, 450), cv::Point(1260, 700), cv::Point(100, 700) };

	cv::Mat imageWarp = perspectiveMapping(image, cameraLanePoints, true);

	cv::Mat imageThres = applyThreshold(imageWarp);

	std::pair<int, int> linesStartingPosition = getLinesStartingPosition(imageThres); // First integer is left line's position, second integer is right line's position

	int leftPointsCount = 0, rightPointsCount = 0;
	std::vector<cv::Point> lanePoints = getLanePoints(imageThres, linesStartingPosition, leftPointsCount, rightPointsCount);

	float relativeDistanceToLaneCenter = getRelativeDistanceToLaneCenter(lanePoints, leftPointsCount, rightPointsCount);
	std::string direction;
	if (relativeDistanceToLaneCenter < 0) direction = "LEFT";
	else direction = "RIGHT";
	cv::putText(image, "REL DISTANCE TO CENTER: " + std::to_string(std::abs(relativeDistanceToLaneCenter)) + " TO THE " + direction, cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(0, 255, 0), 2);

	cv::imshow("image thres", imageThres);
	displayLane(image, cameraLanePoints, lanePoints);
	return image;
}