// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <fstream>

enum Pieces { BISHOP_B, BISHOP_W, PAWN_B, PAWN_W, ROOK_B, ROOK_W, KNIGHT_B, KNIGHT_W, QUEEN_B, QUEEN_W, KING_B, KING_W };

const int noPiecesImgs = 12;
std::string trainFolder = "./Images/Train_3/";
std::string modelFolder = "./Images/Model/";
std::string piecesImgs[noPiecesImgs] = { "bishop_b.jpeg", "bishop_w.jpeg",
							  "pawn_b.jpeg", "pawn_w.jpeg",
							  "rook_b.jpeg", "rook_w.jpeg",
							  "knight_b.jpeg", "knight_w.jpeg",
							  "queen_b.jpeg", "queen_w.jpeg",
							  "king_b.jpeg", "king_w.jpeg" };

int size = 90;

Vec3b background = Vec3b(0, 255, 0);
Vec3b boardBlackColor;
Vec3b boardWhiteColor;
int th_w = 255;
int th_b = 255;

Mat chessboard[8][8];
std::string chessboardNotation[8][8];
Mat pieces[noPiecesImgs];

const int epsilon = 20;

bool equals(Vec3b color1, Vec3b color2, int epsilon) {

	if (abs(color1[0] - color2[0]) > epsilon)
		return false;

	if (abs(color1[1] - color2[1]) > epsilon)
		return false;

	if (abs(color1[2] - color2[2]) > epsilon)
		return false;

	return true;
}

std::vector<Point2i> getNeighbours(Point2i p, int k) {
	int di[12] = { 0, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, -1 };
	int dj[12] = { 1, 1, 0, -1, -1, -1, 0, 1, -1, -1, 0, 1 };
	std::vector<Point2i> neighours;

	switch (k) {
	case 1:
		for (int i = 0; i < 4; i++)
			neighours.push_back(Point2i(p.x + di[i], p.y + dj[i]));
		break;
	case 2:
		for (int i = 0; i < 8; i++)
			neighours.push_back(Point2i(p.x + di[i], p.y + dj[i]));
		break;
	case 3:
		for (int i = 8; i < 12; i++)
			neighours.push_back(Point2i(p.x + di[i], p.y + dj[i]));
		break;
	default:
		break;
	}

	return neighours;
}

Mat getPieceModel(Mat piece) {
	Mat piece_t = Mat(size, size, CV_8UC1);
	cvtColor(piece, piece_t, COLOR_BGR2GRAY);
	threshold(piece_t, piece_t, th_w, 255, THRESH_BINARY);

	std::queue<Point2i> Q;
	Q.push(Point2i(0, 0));
	piece_t.at<uchar>(0, 0) = 128;

	while (!Q.empty()) {
		Point2i p = Q.front();
		Q.pop();

		std::vector<Point2i> neighbours = getNeighbours(p, 2);

		for (int i = 0; i < neighbours.size(); i++) {
			Point2i neighbour = neighbours[i];
			if (neighbour.x >= 0 && neighbour.y >= 0 && neighbour.x < size && neighbour.y < size) {
				if (piece_t.at<uchar>(neighbour.x, neighbour.y) == 255) {
					piece_t.at<uchar>(neighbour.x, neighbour.y) = 128;
					Q.push(neighbour);
				}
			}
		}
	}

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (piece_t.at<uchar>(i, j) == 255) {
				piece_t.at<uchar>(i, j) = 0;
			}
		}
	}

	return piece_t;
}

void savePiece(Mat src, const std::string& fileName) {
	Mat piece = Mat(size, size, CV_8UC3);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			piece.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
		}
	}

	Mat piece_t = getPieceModel(piece);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (piece_t.at<uchar>(i, j) == 128) {
				piece.at<Vec3b>(i, j) = background;
			}
			else if (equals(piece.at<Vec3b>(i, j), boardWhiteColor, epsilon)) {
				piece.at<Vec3b>(i, j) = background;
			}
		}
	}

	imwrite(fileName, piece);
}

void generateModel() {
	for (int i = 0; i < noPiecesImgs; i++) {
		std::string path = trainFolder + piecesImgs[i];
		
		Mat src = imread(path);
		
		std::string fileName = modelFolder + piecesImgs[i];

		savePiece(src, fileName);
	}
}

void setColorsAndSize(bool initialize = true) {
	std::string filePath = modelFolder + "size_colors.txt";
	std::string boardImg = trainFolder + "board.jpeg";

	if (initialize) {
		std::ifstream file(filePath);

		file >> boardBlackColor[0];
		file >> boardBlackColor[1];
		file >> boardBlackColor[2];

		file >> boardWhiteColor[0];
		file >> boardWhiteColor[1];
		file >> boardWhiteColor[2];

		file >> th_w;
		file >> th_b;

		file.close();
	}
	else {
		std::ofstream file(filePath);
		Mat src = imread(boardImg);
		int height = src.rows;
		int width = src.cols;

		Vec3i sum_b;
		Vec3i sum_w;
		for (int i = 1; i < size - 1; i++) {
			for (int j = 1; j < size - 1; j++) {
				sum_w[0] += (int)src.at<Vec3b>(i, j)[0];
				sum_w[1] += (int)src.at<Vec3b>(i, j)[1];
				sum_w[2] += (int)src.at<Vec3b>(i, j)[2];

				sum_b[0] += (int)src.at<Vec3b>(height - i, j)[0];
				sum_b[1] += (int)src.at<Vec3b>(height - i, j)[1];
				sum_b[2] += (int)src.at<Vec3b>(height - i, j)[2];

				int grey = ((int)src.at<Vec3b>(i, j)[0] + (int)src.at<Vec3b>(i, j)[1] + (int)src.at<Vec3b>(i, j)[2]) / 3;
				th_w = th_w > grey ? grey : th_w;
			}
		}
		boardWhiteColor = sum_w / (size * size);
		boardBlackColor = sum_b / (size * size);

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				int grey = ((int)src.at<Vec3b>(i, j)[0] + (int)src.at<Vec3b>(i, j)[1] + (int)src.at<Vec3b>(i, j)[2]) / 3;
				th_b = th_b > grey ? grey : th_b;
			}
		}

		file << boardBlackColor[0] << " ";
		file << boardBlackColor[1] << " ";
		file << boardBlackColor[2] << " ";

		file << boardWhiteColor[0] << " ";
		file << boardWhiteColor[1] << " ";
		file << boardWhiteColor[2] << " ";

		file << th_w - 1<< " ";
		file << th_b - 1 << " ";

		file.close();
	}
}

void train() {
	setColorsAndSize(false);
	generateModel();
}

void initializePieces() {
	for (int i = 0; i < noPiecesImgs; i++) {
		std::string path = modelFolder + piecesImgs[i];

		pieces[i] = imread(path);
	}
}

void initializeChessboard(Mat src) {
	int height = src.rows;
	int width = src.cols;

	for (int i = 0; i < height; i += size) {
		for (int j = 0; j < width; j += size) {
			chessboard[i / size][j / size] = Mat(size, size, CV_8UC3);

			for (int k = 0; k < size; k++) {
				for (int l = 0; l < size; l++) {
					chessboard[i / size][j / size].at<Vec3b>(k, l) = src.at<Vec3b>(i + k, j + l);
				}
			}
		}
	}
}

int match(Mat square) {
	Mat square_t = square.clone();
	cvtColor(square, square_t, COLOR_BGR2GRAY);
	threshold(square_t, square_t, th_b, 255, THRESH_BINARY);

	bool empty = true;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (square_t.at<uchar>(i, j) != 255) {
				empty = false;
				i = size;
				j = size;
			}
		}
	}

	if (empty) {
		return -1;
	}

	int pixelsMatched = 0;
	int type = -1;

	for (int k = 0; k < noPiecesImgs; k++) {
		int currPixelsMatched = 0;

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (pieces[k].at<Vec3b>(i, j) != background && equals(pieces[k].at<Vec3b>(i, j), square.at<Vec3b>(i, j), epsilon)) {
					currPixelsMatched++;
				}
			}
		}

		if (currPixelsMatched > pixelsMatched) {
			pixelsMatched = currPixelsMatched;
			type = k;
		}
	}
	
	return type;
}

std::string getNotation(int type, int i, int j) {
	switch (type) {
		case BISHOP_B: 
			return "b";
		case BISHOP_W:
			return "B";
		case PAWN_B:
			return "p";
		case PAWN_W:
			return "P";
		case ROOK_B:
			return "r";
		case ROOK_W:
			return "R";
		case KNIGHT_B:
			return "n";
		case KNIGHT_W:
			return "N";
		case QUEEN_B:
			return "q";
		case QUEEN_W:
			return "Q";
		case KING_B:
			return "k";
		case KING_W:
			return "K";
		default:
			return (i+j) % 2 == 0 ? "," : ".";
	}
}

void analyzeBoard() {

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			int type = match(chessboard[i][j]);
			chessboardNotation[i][j] = getNotation(type, i, j);
		}
	}
}

void printBoard() {
	std::cout << "----------------\n";

	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			std::cout << chessboardNotation[i][j] << " ";
		}
		std::cout << "\n";
	}

	std::cout << "----------------\n";
}

void test() {
	setColorsAndSize();
	initializePieces();

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		
		initializeChessboard(src);
		analyzeBoard();
		printBoard();

		imshow("image", src);
		waitKey();
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();

		printf("Menu:\n");
		printf(" 1 - Train Model\n");
		printf(" 2 - Test Model\n");
		printf("Option: ");

		scanf("%d",&op);

		switch (op)
		{
			case 1:
				train();
				break;
			case 2:
				test();
				break;
		}

	} while (op!=0);

	return 0;
}