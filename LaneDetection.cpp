#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>
#include <vector>
#include <iostream>

cv::Mat preprocessImage(cv::Mat image);
cv::Mat getImageROI(cv::Mat image);
cv::Mat getLaneLinesImage(cv::Mat image, cv::Mat original, std::vector<cv::Vec4i>& lines);
cv::Vec4i getCoordinates(cv::Size imageSize, cv::Vec2d lineSif);
std::vector<cv::Vec4i> averageSlopeInterceptForm(cv::Size imageSize, std::vector<cv::Vec4i> lines);
void displayLane(cv::Mat image, std::vector<cv::Point> lane);

int main() {
	uint16_t testcase;
	std::cout << "Please input 1 for image test or 2 for video test: ";
	std::cin >> testcase;

	switch (testcase) {
	case 1: {
		// Read image
		cv::Mat image = cv::imread("image1.jpg");
		cv::resize(image, image, cv::Size(1200, 700));

		cv::Mat imageProc = preprocessImage(image);
		cv::imshow("Image Proc", imageProc);
		cv::Mat imageMask = getImageROI(imageProc);
		cv::imshow("Image Mask", imageMask);

		// Get lane lines using Hough Transform technique
		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(imageMask, lines, 2, CV_PI / 180, 100, 40, 5);

		std::vector<cv::Vec4i> linesAvg = averageSlopeInterceptForm(image.size(), lines);
		cv::Mat imageLines = getLaneLinesImage(imageMask, image, linesAvg);

		// Display lane on original image
		cv::Mat imageLane;
		cv::addWeighted(image, 0.8, imageLines, 1, 1, imageLane);

		cv::imshow("Image", imageLane);
		cv::waitKey(0);

		break;
	}
	case 2: {
		// Read video
		int videoId = 0;
		while (!(videoId >= 1 && videoId <= 2)) {
			std::cout << "Please choose the video ID (a number between 1 and 2): ";
			std::cin >> videoId;
		}
		cv::VideoCapture cap("video" + std::to_string(videoId) + ".mp4");
		while (cap.isOpened()) {
			cv::Mat image; // Frame
			cap.read(image);
			cv::resize(image, image, cv::Size(1200, 700));

			cv::Mat imageProc = preprocessImage(image);
			cv::Mat imageMask = getImageROI(imageProc);

			// Get lane lines using Hough Transform technique
			std::vector<cv::Vec4i> lines;
			cv::HoughLinesP(imageMask, lines, 2, CV_PI / 180, 100, 40, 5);

			std::vector<cv::Vec4i> linesAvg = averageSlopeInterceptForm(image.size(), lines);
			cv::Mat imageLines = getLaneLinesImage(imageMask, image, linesAvg);

			// Display lane on original image
			cv::Mat imageLane = image;

			std::vector<cv::Point> lane = { cv::Point(linesAvg[0][0], linesAvg[0][1]), cv::Point(linesAvg[0][2], linesAvg[0][3]), cv::Point(linesAvg[1][2], linesAvg[1][3]), cv::Point(linesAvg[1][0], linesAvg[1][1]) };
			displayLane(imageLane, lane);

			cv::imshow("Image", imageLane);
			if (cv::waitKey(15) == 'q') break;
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

cv::Mat preprocessImage(cv::Mat image) {
	cv::Mat imageProc;
	cv::cvtColor(image, imageProc, cv::COLOR_RGB2GRAY); // Converts to grayscale
	cv::GaussianBlur(imageProc, imageProc, cv::Size(5, 5), 0); // Reduces image noise
	cv::Canny(imageProc, imageProc, 50, 150); // Identifies edges
	return imageProc;
}

cv::Mat getImageROI(cv::Mat image) {
	// The region of interest is the lane the car is currently on, which if contoured from the camera's perspective
	// until it is out of sight, resembles a triangle.

	// Using that information, we can create a mask by defining a matrix of zeros, which size is equal to our image
	// size, then changing the values of the elements situated at the indexes of our filled triangle to 255 (white).

	cv::Size imageSize = image.size();
	int imageHeight = imageSize.height, imageWidth = imageSize.width;
	std::vector<cv::Point> triangle = { cv::Point(200, imageHeight), cv::Point(1100, imageHeight), cv::Point(550, 250) };
	cv::Mat mask(imageSize, CV_8UC1, cv::Scalar(0, 0, 0));
	cv::fillPoly(mask, triangle, cv::Scalar(255, 255, 255));

	// Using the bitwise and operator, we can mask by comparing the binary representation of the values of our matrixes
	// located at the same indexes bit by bit, and storing the result of the and operator in an output matrix at the
	// same indexes as the ones' of our elements.

	// I.e. if we have a pixel in our image whose value is 135 (in binary it's 10000111), and at the same indexes 
	// in our mask we have the value 255 (11111111), by using the bitwise and operator, the result will be the pixel value;
	// if we are to compare the same pixel, but to the value of 0, the result will be 0.

	// Therefore, since the pixels that are in our region of interest are compared to 255 and the ones' outside of it
	// are compared to 0, the output matrix will only hold the pixel values' for our region of interest, with the rest of
	// the elements being zero.

	cv::Mat imageMask;
	cv::bitwise_and(image, mask, imageMask);

	return imageMask;
}

cv::Mat getLaneLinesImage(cv::Mat image, cv::Mat original, std::vector<cv::Vec4i>& lines) {
	cv::Mat imageLines(image.size(), original.type(), cv::Scalar(0, 0, 0));
	// For each line, 
	if (!lines.empty()) {
		for (auto line : lines) {
			cv::line(imageLines, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 255), 10);
		}
	}

	return imageLines;
}

cv::Vec4i getCoordinates(cv::Size imageSize, cv::Vec2d lineSif) {
	// Get the coordinates of the two points bounding the line segment
	double slope = lineSif[0], intercept = lineSif[1];
	int y1 = imageSize.height; // The Y of the first point is the bottom of the image
	int y2 = y1 * (3.0 / 5.0); // The Y of the second point is 3/5 * bottom of the image in order to 
	// From slope intercept form Y = m * X + n we get X = (Y - n) / m, where m is the slope and n is the intercept
	int x1 = (y1 - intercept) / slope;
	int x2 = (y2 - intercept) / slope;
	return cv::Vec4i(x1, y1, x2, y2);
}

std::vector<cv::Vec4i> averageSlopeInterceptForm(cv::Size imageSize, std::vector<cv::Vec4i> lines) {
	// The slope intercept form (sif) of a line is the line equation of form Y = m * X + n, where m is the slope of the line and n
	// is the intercept.
	std::vector<cv::Vec2d> leftSifVec, rightSifVec;

	for (auto line : lines) {
		double x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
		arma::vec X = { x1, x2 };
		arma::vec Y = { y1, y2 };

		// Fit a linear polynomial (order 1)
		arma::vec coefficients = arma::polyfit(X, Y, 1);

		// Extract slope and intercept
		double slope = coefficients[0];
		double intercept = coefficients[1];

		if (slope >= 0) {
			rightSifVec.push_back(cv::Vec2d(slope, intercept));
		}
		else {
			leftSifVec.push_back(cv::Vec2d(slope, intercept));
		}
	}

	double leftSlopeAvg = 0, leftInterceptAvg = 0, rightSlopeAvg = 0, rightInterceptAvg = 0;

	// Calculate the average slope and intercept for the left and right line
	for (auto elem : leftSifVec) {
		leftSlopeAvg += elem[0];
		leftInterceptAvg += elem[1];
	}
	leftSlopeAvg /= leftSifVec.size();
	leftInterceptAvg /= leftSifVec.size();

	for (auto elem : rightSifVec) {
		rightSlopeAvg += elem[0];
		rightInterceptAvg += elem[1];
	}
	rightSlopeAvg /= rightSifVec.size();
	rightInterceptAvg /= rightSifVec.size();

	cv::Vec2d leftAvgSif(leftSlopeAvg, leftInterceptAvg);
	cv::Vec2d rightAvgSif(rightSlopeAvg, rightInterceptAvg);

	// Compute
	cv::Vec4i leftLine = getCoordinates(imageSize, leftAvgSif);
	cv::Vec4i rightLine = getCoordinates(imageSize, rightAvgSif);

	return { leftLine, rightLine };
}

void displayLane(cv::Mat image, std::vector<cv::Point> lane) {
	double opacity = 0.50;
	cv::Mat layer = cv::Mat::zeros(image.size(), CV_8UC3);
	cv::fillPoly(layer, lane, cv::Scalar(0, 0, 255));
	cv::addWeighted(image, 1, layer, 1 - opacity, 0, image);
}