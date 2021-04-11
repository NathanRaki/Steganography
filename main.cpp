#include <opencv2/opencv.hpp>
#include <iostream>
#include <numbers>
#include <vector>

# define M_PI 3.14159265358979323846
typedef std::vector<std::vector<int>> block;
typedef std::vector<std::vector<block>> grid;
typedef std::pair<int, int> pos;

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
void convertToDCT(cv::Mat& img, grid& dct)
{
	cv::Mat img_f;
	img.convertTo(img_f, CV_32F); // Convert image to float values instead of uchar
	block b(8, std::vector<int>(8)); // 8x8 vector, filled with zeros

	// Iterating through blocks (step=8)
	for (int i = 0; i < img.rows; i+=8)
	{
		std::vector<block> grid_row; // Row of blocks, to fill the grid
		for (int j = 0; j < img.cols; j+=8)
		{
			// Iterating through each block value
			for (int u = 0; u < 8; u++)
			{
				for (int v = 0; v < 8; v++)
				{
					float Cu, Cv, sum = 0;
					if (u == 0) { Cu = 1 / sqrt(2); } else { Cu = 1; } // cf. DCT formula
					if (v == 0) { Cv = 1 / sqrt(2); } else { Cv = 1; } // cf. DCT formula
					// Iterating through each pixel
					for (int x = 0; x < 8; x++)
					{
						for (int y = 0; y < 8; y++)
						{
							sum += img_f.at<float>(i+x, j+y)
									* cos(((2.0 * x + 1) * u * M_PI) / 16.0)
									* cos(((2.0 * y + 1) * v * M_PI) / 16.0);
						}
					}
					// Once we've seen every pixel, we can update the dct coefficient value and go to the next one
					b[u][v] = round(0.25 * Cu * Cv * sum);
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
		res.push_back(std::make_pair(int(p[i][2]/m), p[i][2]%n));
	return res;
}

int main()
{
	// Convert image to DCT grid
	cv::Mat img = cv::imread("panther.jpg", cv::IMREAD_GRAYSCALE);
	grid g;
	convertToDCT(img, g);

	// Display block content
	block b = g[30][99]; // 31st row, 100th column
	for (auto x : b)
	{
		for (auto y : x)
		{
			printf("%5i ", y);
		}
		std::cout << std::endl;
	}

	// Display block content in zigzag scan order
	std::vector<pos> order = zigzagscan(8, 8);
	for (auto x : order)
	{
		printf("%5i ", b[x.first][x.second]);
	}

	int DC_pos = 0; // DC coefficient pos in zigzag scan
	int lastMF_pos = 48; // Last Middle Frequency pos in zigzag scan

	// TODO:
	// Create a function to transform DCT int in binary
	// Create a function to hide the message in Middle Frequencies :
	//	Pour cacher le message, il faut le convertir en binaire
	// 	Ensuite, on a un paramètre à définir : b (le nombre de bits modifiés par coefficient)
	//	On va cacher le message en partant de la dernière fréquence moyenne et en remontant à l'envers
	//	Si on prend le scan zigzag, la dernière fréquence moyenne est à la position 48 (58 dans le sens de lecture normal)
	//	Après on va check les LSB des coef, si b=1 on prend le dernier bit, si b=2 les deux derniers bits etc.
	//	Si b=2 et que les deux bits sont pareils que les deux premiers de notre message, on laisse comme ça et on passe à la suite
	//	Si b=2 et que les bits sont différents c'est un peu plus compliqué
	//	Imaginons le DCT c'est 8 (donc 1000 en binaire) et que nous les deux bits qu'on veut mettre c'est 01. Si on changeait les deux derniers on aurait 1001 = 9
	//	Et changer comme ça on aime pas, donc on va check si y'a un coefficient=9, si c'est le cas on va échanger les deux coef, sinon on échange avec le coeff dont la fin binaire ressemble le + à 01
	
	return 0;
}