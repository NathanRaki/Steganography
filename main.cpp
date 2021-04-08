#include <opencv2/opencv.hpp>
#include <iostream>
#include <numbers>
#include <vector>

# define M_PI 3.14159265358979323846
typedef std::vector<std::vector<float>> block;

void get_dct(block& in, block& out)
{
	float Cu, Cv, sum, intensity;
	int x, y, u, v;

	for (u = 0; u < 8; u++)
	{
		std::vector<float> row;
		for (v = 0; v < 8; v++)
		{
			sum = 0;
			if (u == 0) { Cu = 1 / sqrt(2); }
			else { Cu = 1; }
			if (v == 0) { Cv = 1 / sqrt(2); }
			else { Cv = 1; }
			for (x = 0; x < 8; x++)
			{
				for (y = 0; y < 8; y++)
				{
					intensity = in[x][y];
					sum +=
						intensity *
						cos(((2.0 * x + 1) * u * M_PI) / 16.0) *
						cos(((2.0 * y + 1) * v * M_PI) / 16.0);
				}
			}
			row.push_back(0.25 * Cu * Cv * sum);
		}
		out.push_back(row);
	}
}

void get_dcts(cv::Mat& in, std::vector<block>& out)
{
	cv::Mat in_f;
	in.convertTo(in_f, CV_32F);
	for (int x = 0; x < in.rows; x += 8)
	{
		for (int y = 0; y < in.cols; y += 8)
		{
			block pixels, dct;
			for (int x_ = 0; x_ < 8; x_++)
			{
				std::vector<float> row;
				for (int y_ = 0; y_ < 8; y_++)
				{
					row.push_back(in_f.at<float>(x + x_, y + y_));
				}
				pixels.push_back(row);
			}
			get_dct(pixels, dct);
			out.push_back(dct);
		}
	}
}

void print_dct(std::vector<block>& dcts, int id)
{
	block dct = dcts[id];
	for (std::vector<float> row : dct)
	{
		for (float v : row)
		{
			printf("%8.1f ", v);
		}
		std::cout << std::endl;
	}
}

int main()
{
	cv::Mat img = cv::imread("panther.jpg", cv::IMREAD_GRAYSCALE);
	std::vector<block> dcts;
	get_dcts(img, dcts);
	print_dct(dcts, 5500); // 5500th block
	return 0;
}