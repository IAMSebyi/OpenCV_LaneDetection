// OpenCV includes for image processing and GUI operations
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// C++ Standard Library includes for filesystem operations, set, vector, file I/O, and console I/O
#include <filesystem>
#include <set>
#include <vector>
#include <fstream>
#include <iostream>

// CONSTANTS
const int IMAGE_WIDTH = 1280;
const int IMAGE_HEIGHT = 720;
const cv::Size IMAGE_SIZE(IMAGE_WIDTH, IMAGE_HEIGHT);

// METHODS
void initSettings(std::vector<cv::Point>& cameraLanePoints);
std::vector<cv::Mat> loadTestImages();
cv::Mat perspectiveMapping(const cv::Mat& image, const std::vector<cv::Point>& cameraLanePoints, bool inverse);
cv::Mat processImage(cv::Mat image);
std::pair<int, int> getLinesStartingPosition(const cv::Mat& image);
std::vector<cv::Point> getLanePoints(const cv::Mat& image, const std::pair<int, int>& linesStartingPosition, std::vector<cv::Point>& leftLaneLinePoints, std::vector<cv::Point>& rightLaneLinePoints);
float displayData(cv::Mat& image, const std::vector<cv::Point>& leftLaneLinePoints, const std::vector<cv::Point>& rightLaneLinePoints);
void displayLane(cv::Mat& image, const std::vector<cv::Point>& cameraLanePoints, const std::vector<cv::Point>& lanePoints);
cv::Mat detectLaneImage(cv::Mat& image, const std::vector<cv::Point>& cameraLanePoints);

int main() {
	// SETTINGS VARIABLES
	std::vector<cv::Point> cameraLanePoints;

	initSettings(cameraLanePoints);

	// Prompt user to select test case
	uint16_t testcase;
	std::cout << "Please input 1 for image test, 2 for video test or 3 for settings test: ";
	std::cin >> testcase;

	switch (testcase) {
	case 1: {
		// Handle image test case
		std::vector<cv::Mat> testImages = loadTestImages();

		// Check if any images were loaded
		if (testImages.empty()) {
			std::cout << "No images detected. Please check Images folder." << std::endl;
			return 1;
		}

		std::cout << "Press any key (besides 'Q') to switch to the next image or 'Q' to quit." << std::endl;
		for (auto image : testImages) {
			cv::Mat resultImage = detectLaneImage(image, cameraLanePoints);
			cv::imshow("Image", resultImage);
			if (cv::waitKey(0) == 'q') break;
		}

		break;
	}
	case 2: {
		// Handle video test case

		// Define supported video extensions
		const std::set<std::string> videoExtensions = {
		".mp4", ".mov", ".avi", "mkv", "flv"
		};

		std::filesystem::path folder = "Videos";

		try {
			// Check if the given path is a directory
			if (std::filesystem::is_directory(folder)) {
				std::cout << "Press 'N' to switch to next video or 'Q' to quit." << std::endl;

				// Iterate through the directory
				for (const auto& entry : std::filesystem::directory_iterator(folder)) {
					// Check if the entry is a regular file and if it's an image file
					if (std::filesystem::is_regular_file(entry) && videoExtensions.find(entry.path().extension().string()) != videoExtensions.end()) {
						std::cout << "Video file detected: " << entry.path() << std::endl;

						cv::VideoCapture video (entry.path().string());
						if (video.isOpened()) {
							while (video.isOpened()) {
								cv::Mat frame;
								video.read(frame);
								if (frame.empty()) break;
								cv::resize(frame, frame, IMAGE_SIZE);

								cv::Mat resultFrame = detectLaneImage(frame, cameraLanePoints);

								cv::imshow("Video", resultFrame);

								char keyInput = cv::waitKey(1);
								if (keyInput == 'q') { cv::destroyAllWindows(); return 0; }
								else if (keyInput == 'n') break;
							}

							video.release();
						}
						else {
							std::cout << "Couldn't load video. Might be corrupted." << std::endl;
						}
					}
				}
			}
			else {
				std::cerr << "Videos directory missing. Check if Videos folder exists." << std::endl;
			}
		}
		catch (const std::filesystem::filesystem_error& e) {
			std::cerr << "Filesystem error: " << e.what() << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "General error: " << e.what() << std::endl;
		}

		break;
	}
	case 3: {
		// Handle settings test case
		std::vector<cv::Mat> testImages = loadTestImages();

		if (testImages.empty()) {
			std::cout << "No images detected. Please check Images folder." << std::endl;
			return 1;
		}
		
		std::cout << "Please check if the current camera lane settings produce an accurate Bird's Eye View image of the lane for each test image." << std::endl;
		std::cout << "Press Q to quit iteration or any other key to switch to the next image." << std::endl;

		for (auto image : testImages) {
			cv::Mat imageWarp = perspectiveMapping(image, cameraLanePoints, true), imageCopy;
			image.copyTo(imageCopy);
			for (auto point : cameraLanePoints)
				cv::circle(imageCopy, point, 7, cv::Scalar(0, 255, 0), cv::FILLED);
			cv::imshow("Image", imageCopy);
			cv::imshow("Bird's Eye View", imageWarp);
			if (cv::waitKey(0) == 'q') break;
		}

		std::cout << "If the produced Bird's Eye View images are not accurate, please try manually adjusting the point values in the cameraLaneSettings.txt file." << std::endl;
		break;
	}

	default:
		std::cout << "Fair enough." << std::endl;
		break;
	}

	cv::destroyAllWindows();

	return 0;
}

void initSettings(std::vector<cv::Point>& cameraLanePoints)
{
	// Different camera positions on the car result different perspectives of the road.
	// If testing using your own or other videos, please change the variables of the points in cameraLanePoints.txt file
	std::ifstream cameraLanePointsFile("cameraLanePoints.txt");
	for (int i = 0; i < 4; i++) {
		int x, y;
		cameraLanePointsFile >> x >> y;
		cameraLanePoints.push_back(cv::Point(x, y));
	}
}

std::vector<cv::Mat> loadTestImages() {
	std::vector<cv::Mat> testImages;

	std::cout << "Loading images..." << std::endl;

	// Define supported image extensions
	const std::set<std::string> imageExtensions = {
	".jpg", ".jpeg", ".png",
	};
	std::filesystem::path folder = "Images";
	try {
		// Check if the given path is a directory
		if (std::filesystem::is_directory(folder)) {
			// Iterate through the directory
			for (const auto& entry : std::filesystem::directory_iterator(folder)) {
				// Check if the entry is a regular file and if it's an image file
				if (std::filesystem::is_regular_file(entry) && imageExtensions.find(entry.path().extension().string()) != imageExtensions.end()) {
					std::cout << "Image file detected: " << entry.path() << std::endl;

					cv::Mat image = cv::imread(entry.path().string());

					if (!image.empty()) {
						cv::resize(image, image, IMAGE_SIZE);
						testImages.push_back(image);
						std::cout << "Image loaded successfully." << std::endl;
					}
					else {
						std::cout << "Couldn't load image. Might be corrupted." << std::endl;
					}
				}
			}
		}
		else {
			std::cerr << "Images directory missing. Check if Images folder exists." << std::endl;
		}
	}
	catch (const std::filesystem::filesystem_error& e) {
		std::cerr << "Filesystem error: " << e.what() << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "General error: " << e.what() << std::endl;
	}

	return testImages;
}

cv::Mat perspectiveMapping(const cv::Mat& image, const std::vector<cv::Point>& cameraLanePoints, bool inverse) {
	cv::Mat imageWarp;

	// Define source and destination points for perspective transformation
	std::vector<cv::Point2f> src = { cv::Point2f(cameraLanePoints[0].x, cameraLanePoints[0].y), cv::Point2f(cameraLanePoints[3].x, cameraLanePoints[3].y),
		cv::Point2f(cameraLanePoints[1].x, cameraLanePoints[1].y), cv::Point2f(cameraLanePoints[2].x, cameraLanePoints[2].y) };

	std::vector<cv::Point2f> dest = { cv::Point2f(200, 0), cv::Point2f(200, 720), cv::Point2f(1080, 0), cv::Point2f(1080, 720) };

	cv::Mat perspectiveMatrix;
	if (inverse) perspectiveMatrix = cv::getPerspectiveTransform(src, dest);
	else perspectiveMatrix = cv::getPerspectiveTransform(dest, src);
	
	cv::warpPerspective(image, imageWarp, perspectiveMatrix, IMAGE_SIZE);

	return imageWarp;
}



cv::Mat processImage(cv::Mat image) {
	cv::Mat redChannel;

	// Apply blur
	cv::GaussianBlur(image, image, cv::Size(5, 5), 0);

	// Compute red channel threshold image matrix
	std::vector<cv::Mat> channels;
	cv::split(image, channels);
	redChannel = channels[2];
	cv::threshold(redChannel, redChannel, 215, 255, cv::THRESH_BINARY);

	return redChannel;
}



std::pair<int, int> getLinesStartingPosition(const cv::Mat& image) {
	cv::Size size = image.size();

	// Make histogram
	std::vector<int> histogram(IMAGE_WIDTH, 0);
	for (int x = 0; x < size.width; x++) {
		for (int y = 0; y < size.height; y++) {
			if (image.at<uchar>(y, x) == 255) histogram[x]++;
		}
	}

	// Get the probable start position of the left line and right line
	int leftLineP = 0, rightLineP = IMAGE_WIDTH - 1, leftMax = -1, rightMax = -1;
	for (int i = 0; i < IMAGE_WIDTH / 2; i++) {
		if (histogram[i] >= leftMax) {
			leftMax = histogram[i];
			leftLineP = i;
		}
	}

	for (int i = 640; i < IMAGE_WIDTH; i++) {
		if (histogram[i] >= rightMax) {
			rightMax = histogram[i];
			rightLineP = i;
		}
	}

	return std::pair<int, int>(leftLineP, rightLineP);
}



std::vector<cv::Point> getLanePoints(const cv::Mat& image, const std::pair<int, int>& linesStartingPosition, std::vector<cv::Point>& leftLaneLinePoints, std::vector<cv::Point>& rightLaneLinePoints) {
	int y = IMAGE_HEIGHT;

	std::pair<int, int> linesPosition = linesStartingPosition;
	std::vector<cv::Point> lanePoints;

	while (y >= 50) {
		cv::Mat imageCrop;

		// Left line
		imageCrop = image(cv::Rect(cv::Point2i(std::max(linesPosition.first - 50, 0), y - 50), cv::Point2i(std::min(linesPosition.first + 50, IMAGE_WIDTH), y)));

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
				
				leftLaneLinePoints.push_back(cv::Point(linesPosition.first, y));
			}
		}
		if (!foundLeftLanePoint) { lanePoints.push_back(cv::Point(linesPosition.first, y));  leftLaneLinePoints.push_back(cv::Point(linesPosition.first, y)); }

		// Right line
		imageCrop = image(cv::Rect(cv::Point2i(std::max(linesPosition.second - 50, 0), y - 50), cv::Point2i(std::min(linesPosition.second + 50, IMAGE_WIDTH), y)));

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
				
				rightLaneLinePoints.push_back(cv::Point(linesPosition.second, y));
			}
		}
		if (!foundRightLanePoint) { lanePoints.push_back(cv::Point(linesPosition.second, y)); rightLaneLinePoints.push_back(cv::Point(linesPosition.second, y)); }

		y -= 50;
	}

	return lanePoints;
}



float displayData(cv::Mat &image, const std::vector<cv::Point>& leftLaneLinePoints, const std::vector<cv::Point>& rightLaneLinePoints) {
	std::string direction;

	float relativeDistanceToLaneCenter = (leftLaneLinePoints[0].x + rightLaneLinePoints[0].x) / 2.0f - IMAGE_WIDTH / 2;

	if (relativeDistanceToLaneCenter < 0) direction = "LEFT";
	else direction = "RIGHT";
	cv::putText(image, "REL DISTANCE TO CENTER: " + std::to_string(std::abs(relativeDistanceToLaneCenter)) + " TO THE " + direction, cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(0, 255, 0), 2);

	float relativeLaneDeviation = (leftLaneLinePoints[leftLaneLinePoints.size() - 1].x + rightLaneLinePoints[rightLaneLinePoints.size() - 1].x) / 2.0f - IMAGE_WIDTH / 2;

	if (relativeLaneDeviation < 0) direction = "LEFT";
	else direction = "RIGHT";

	relativeLaneDeviation = std::abs(relativeLaneDeviation);

	std::string deviationLevel = "INSIGNIFICANT";

	if (relativeLaneDeviation >= 30 && relativeLaneDeviation < 50) deviationLevel = "VERY SMALL";
	else if (relativeLaneDeviation >= 50 && relativeLaneDeviation < 70) deviationLevel = "SMALL";
	else if (relativeLaneDeviation >= 70 && relativeLaneDeviation < 130) deviationLevel = "MEDIUM";
	else if (relativeLaneDeviation >= 130) deviationLevel = "HIGH";
	
	cv::putText(image, "REL LANE DEVIATION: " + std::to_string(relativeLaneDeviation) + " (" + deviationLevel + " DEVIATION) TO THE " + direction, cv::Point(10, 50), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(0, 255, 0), 2);
}



void displayLane(cv::Mat& image, const std::vector<cv::Point>& cameraLanePoints, const std::vector<cv::Point>& lanePoints) {
	cv::Mat laneImage = cv::Mat::zeros(IMAGE_SIZE, CV_8UC3);

	std::vector<cv::Point> hullPoints;
	cv::convexHull(lanePoints, hullPoints);
	cv::fillPoly(laneImage, hullPoints, cv::Scalar(0, 255, 0));

	laneImage = perspectiveMapping(laneImage, cameraLanePoints, false);

	double opacity = 0.50;
	cv::addWeighted(image, 1, laneImage, 1 - opacity, 0, image);
}



cv::Mat detectLaneImage(cv::Mat& image, const std::vector<cv::Point>& cameraLanePoints) {
	cv::Mat imageWarp = perspectiveMapping(image, cameraLanePoints, true);

	cv::Mat imageThres = processImage(imageWarp);

	std::pair<int, int> linesStartingPosition = getLinesStartingPosition(imageThres); // First integer is left line's position, second integer is right line's position

	std::vector<cv::Point> leftLaneLinePoints, rightLaneLinePoints;
	std::vector<cv::Point> lanePoints = getLanePoints(imageThres, linesStartingPosition, leftLaneLinePoints, rightLaneLinePoints);

	displayData(image, leftLaneLinePoints, rightLaneLinePoints);
	displayLane(image, cameraLanePoints, lanePoints);
	return image;
}