#include <opencv2\opencv.hpp>  
#include <opencv2\core\core.hpp>
#include <opencv2\core\mat.hpp>
#include <iostream>
#include <numbers>
#include <vector>
#include <string>
#include <bitset>
#include <cstdlib>
#include <typeinfo>

# define M_PI 3.14159265358979323846
typedef std::vector<std::vector<int>> block; //vector of vector int
typedef std::vector<std::vector<block>> grid; //vector of vector of block
typedef std::pair<int, int> pos; //pair int / int

using namespace std;
using namespace cv;


// Quantize a DCT block with luminance quantization table
// in : DCT 8x8 matrix
// out (output) : Quantized DCT 8x8 matrix
void quantize(block& in, block& out)
{
	// Importing the luminance quantization table
	float q_table[8][8] = { {16, 11, 10, 16,  24,  40,  51,  61},
							{12, 12, 14, 19,  26,  58,  60,  55},
							{14, 13, 16, 24,  40,  57,  69,  56},
							{14, 17, 22, 29,  51,  87,  80,  62},
							{18, 22, 37, 56,  68, 109, 103,  77},
							{24, 35, 55, 64,  81, 104, 113,  92},
							{49, 64, 78, 87, 103, 121, 120, 101},
							{72, 92, 95, 98, 112, 100, 103,  99} };
	// Go through the block and quantize every value
	for (int x = 0; x < 8; x++)
	{
		std::vector<int> row;
		for (int y = 0; y < 8; y++)
		{
			row.push_back(round(in[x][y] / q_table[x][y]));
		}
		out.push_back(row);
	}
}

// Convert image to a grid of 8x8 DCT blocks
// img : Image you want to convert
// dct (output) : Grid of 8x8 DCT blocks
void pixel2DCT(cv::Mat& img, grid& dct)
{
	cv::Mat img_f;
	img.convertTo(img_f, CV_32F); // Convert image to float values instead of uchar
	block b(8, std::vector<int>(8)); // 8x8 vector, filled with zeros

	// Iterating through blocks (step=8)
	for (int i = 0; i < img.rows; i += 8)
	{
		std::vector<block> grid_row; // Row of blocks, to fill the grid
		for (int j = 0; j < img.cols; j += 8)
		{
			// Iterating through each block value
			for (float u = 0; u < 8; u++)
			{
				for (float v = 0; v < 8; v++)
				{
					float Cu, Cv, sum = 0;
					if (u == 0) { Cu = 1 / sqrt(2); }
					else { Cu = 1; } // cf. DCT formula
					if (v == 0) { Cv = 1 / sqrt(2); }
					else { Cv = 1; } // cf. DCT formula
					// Iterating through each pixel
					for (int x = 0; x < 8; x++)
					{
						for (int y = 0; y < 8; y++)
						{
							sum += img_f.at<float>(i + x, j + y)
								* cos(((2.0 * x + 1.0) * u * M_PI) / 16.0)
								* cos(((2.0 * y + 1.0) * v * M_PI) / 16.0);
						}
					}
					// Once we've seen every pixel, we can update the dct coefficient value and go to the next one
					b[u][v] = round((1 / sqrt(2)) * Cu * Cv * sum);
				}
			}
			grid_row.push_back(b); // Adding the block to the row
		}
		dct.push_back(grid_row); // Adding the row to the grid
	}
}

// C++ ZigZag Scan Algorithm
// Based on https://gist.github.com/gokercebeci/10556381
// Thanks to gokercebeci
bool buble(std::vector<int> i, std::vector<int> j)
{
	return  (i[0] == j[0]) ? (i[1] < j[1]) : (i[0] < j[0]);
}
std::vector<pos> zigzagscan(int n, int m)
{
	std::vector<pos> res;
	std::vector< std::vector<int> > p(n * m, std::vector<int>(3));
	for (int i = 0; i < n * m; i++) {
		p[i][0] = i % n + i / n;
		p[i][1] = (((i % n - i / n) % 2) == 0 ? i % n : i / n);
		p[i][2] = i;
	}
	std::sort(p.begin(), p.end(), buble);

	for (int i = 0; i < n * m; i++)
		res.push_back(std::make_pair(int(p[i][2] / m), p[i][2] % n));
	return res;
}


string char_to_bin(char ch)
{
	bitset<8> temp(ch);
	return temp.to_string();
}
bitset<32> int_to_bin(int i) {
	return std::bitset<32>(i); //to binary
}


//tranform a grid to a matrice
Mat gridToMat(grid g, int sizeI, std::vector<pos> order) {
	Mat matIDCT = Mat::zeros(sizeI, sizeI, CV_64FC1);
	int k, l = 0;
	//for each DCT value, put it into a matrice
	for (auto& a : g) {
		k = 0;
		for (auto& bl : a) {
			for (auto x : order)
			{
				matIDCT.at<double>(x.first + l, x.second + k) = double(bl[x.first][x.second])/255;
			}
			k = k + 8;
		}
		l = l + 8;
	}
	return matIDCT;
}

//invert the DCT coefficient using idct() method
Mat dctToIdct(Mat src) {
	src.convertTo(src, CV_64F);
	//imshow("img2test", matIDCT);

	Mat iDCTtemp, iDCT;
	idct(src, iDCT);

	int divideSize = 8;
	Mat p1;

	for (int i = 0; i < src.rows; i += divideSize)
	{
		for (int j = 0; j < src.cols; j += divideSize)
		{
			//invert using 8*8 blocks
			p1 = src(cv::Rect(i, j, divideSize, divideSize));
			idct(p1, iDCTtemp);
			iDCTtemp.copyTo(iDCT(cv::Rect(i, j, divideSize, divideSize)));
		}
	}

	return iDCT;
}

string decode(Mat img, int n)
{
	grid g;
	pixel2DCT(img, g);
	cout << "--- valeurs DCT dans le premier bloc à décoder ---" << endl;
	block b = g[0][0];
	for (auto row : b)
	{
		for (auto val : row)
		{
			cout << val << " ";
		}
		cout << endl;
	}

	// Display block content in zigzag scan order
	std::vector<pos> order = zigzagscan(8, 8);
	std::vector<pos> order_reverse(order.size()); //reverse zigzag scan order
	std::reverse_copy(std::begin(order), std::end(order), std::begin(order_reverse));

	string messageBits = "";
	int bitCount = 0;
	for (auto& a : g) {
		for (auto& bl : a) {
			for (int i = 15; i < order.size() - 1; ++i)
			{
				pos x = order_reverse[i];
				messageBits += bl[x.first][x.second];
				bitCount++;
				if (bitCount == n) { break; }
			}
			if (bitCount == n) { break; }
		}
		if (bitCount == n) { break; }
	}

	string message = "";
	for (int i = 0; i < messageBits.length(); i += 8)
	{
		for (int j = i; j < i + 8; j++)
		{
			char c = static_cast<char>(std::bitset<8>(messageBits[j]).to_ulong() + 64);
			message += c;
		}
	}
	return message;
}

void dctcoeffreplacement(grid& g, string msg) {
	//int sizeI = 512;

	//Convert image to DCT grid
	//resize(img, img, Size(sizeI, sizeI)); //resize image

	// Display block content in zigzag scan order
	std::vector<pos> order = zigzagscan(8, 8);
	std::vector<pos> order_reverse(order.size()); //reverse zigzag scan order
	std::reverse_copy(std::begin(order), std::end(order), std::begin(order_reverse));

	int DC_pos = 0; // DC coefficient pos in zigzag scan
	int lastMF_pos = 48; // Last Middle Frequency pos in zigzag scan

	//message string converted in binary
	string msgBin = "";
	for (char& c : msg) {
		msgBin += char_to_bin(c);
	}
	int msgsize = msgBin.size(); //size of the binary message
	int posmsg = 0; //current pos of the bit in the message to encode

	//parameters
	int b = 1; //number of bits modified in each DC coefficient


	for (auto& a : g) {
		for (auto& bl : a) {
			for (int i = 15; i < order.size() - 1; ++i)
			{
				pos x = order_reverse[i];
				//only before the last middle frequency pos at (7,2)

				int xLSB = abs(bl[x.first][x.second] % 2); //get the Least Significant Bit of the coefficient
				string msgBit(1, msgBin[posmsg]);
				int xval = bl[x.first][x.second];

				//if the MB (Message Bit to encode) and the LSB are different
				if ((xLSB == 0 && msgBit == "1") || (xLSB == 1 && msgBit == "0")) {
					int temp = 0;
					if (xLSB == 0 && msgBit == "1") {
						temp = xval + 1;
					}
					else if (xLSB == 1 && msgBit == "0") {
						temp = xval - 1;
					}

					bool changed = false;
					int yval, yLSB;
					pos closest(-1, 0);
					for (int j = i + 1; j < order.size() - 1; j++)
					{
						pos y = order_reverse[j];
						int yval = bl[y.first][y.second];
						if (temp == yval)
						{
							bl[y.first][y.second] = xval;
							bl[x.first][x.second] = yval;
							changed = true;
						}
						if (changed) { break; }
						yLSB = abs(bl[y.first][y.second] % 2);
						if (yLSB == stoi(msgBit))
						{
							if (closest.first == -1) { closest.first = y.first; closest.second = y.second; }
							else if (abs(temp - yval) < abs(temp - bl[closest.first][closest.second])) { closest.first = y.first; closest.second = y.second; }
						}
					}

					if (!changed) {
						if (closest.first == -1) { bl[x.first][x.second] = temp; }
						else
						{
							int cval = bl[closest.first][closest.second];
							bl[closest.first][closest.second] = xval;
							bl[x.first][x.second] = cval;
						}
					}
				}
				posmsg++;
				if (posmsg == msgsize) {
					break;
				}
			}
			if (posmsg == msgsize) {
				break;
			}
		}
		if (posmsg == msgsize) {
			break;
		}
	}
}

//calculation of inverted DCT coefficient
void DCT2pixel(grid in, Mat& out)
{
	int width = in[0].size() * 8;
	out = Mat::zeros(width, width, CV_8UC1);

	grid g;

	int cu, cv, sum;
	for (int i = 0; i < out.rows; i += 8)
	{
		for (int j = 0; j < out.cols; j += 8)
		{
			for (int x = 0; x < 8; x++)
			{
				for (int y = 0; y < 8; y++)
				{
					sum = 0;
					for (int u = 0; u < 8; u++)
					{
						for (int v = 0; v < 8; v++)
						{
							if (u == 0) { cu = 1.0 / sqrt(2.0); }
							else { cu = 1; }
							if (v == 0) { cv = 1.0 / sqrt(2.0); }
							else { cv = 1; }
							sum += cu * cv * in[i / 8][j / 8][u][v]
								* cos(((2.0 * x + 1) * u * M_PI) / 16.0)
								* cos(((2.0 * y + 1) * v * M_PI) / 16.0);
						}
					}
					//cout << "Somme : " << abs(sum) << endl;
					out.at<uchar>(i + x, j + y) = ((1 / sqrt(16.0)) * abs(sum));
				}
			}
		}
	}
}

void encode(Mat in, Mat& out, string msg)
{
	cout << "--- valeurs de l'image d'entree dans le premier bloc ---" << endl;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			cout << (int)in.at<uchar>(i, j) << " ";
		}
		cout << endl;
	}

	grid g;
	pixel2DCT(in, g); 
	cout << "--- valeurs DCT dans le premier bloc avant remplacement ---" << endl;
	block b = g[0][0];
	for (auto row : b)
	{
		for (auto val : row)
		{
			cout << val << " ";
		}
		cout << endl;
	}

	dctcoeffreplacement(g, msg);
	cout << "--- valeurs DCT dans le premier bloc apres remplacement ---" << endl;
	b = g[0][0];
	for (auto row : b)
	{
		for (auto val : row)
		{
			cout << val << " ";
		}
		cout << endl;
	}

	DCT2pixel(g, out);

	cout << "--- valeurs de l'image de sortie dans le premier bloc ---" << endl;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			cout << (int)out.at<uchar>(i, j) << " ";
		}
		cout << endl;
	}
}

int main()
{
	string msg = "On est des tubes on est pas des pots, mais on a tout ce qu'il vous faut"; //Message to hide

	cv::Mat img = cv::imread("panther.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img_encoded;

	encode(img, img_encoded, msg);

	imshow("img", img);
	imshow("img_encoded", img_encoded);

	string message = "";
	message = decode(img_encoded, 72 * 8);
	cout << "Message code : " <<endl << message << endl;

	//Mat stegano = dctcoeffreplacement(img, msg);
	//imshow("stegano", stegano);
	waitKey();

	return 0;
}
