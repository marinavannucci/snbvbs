#ifndef _UTILITY
#define _UTILITY
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <stdexcept>

using std::exp;
using std::log;
using std::tanh;
using std::invalid_argument;
using Eigen::VectorXd;
using Eigen::Map;
typedef Map<VectorXd> MapVecd;
typedef unsigned int uint;


// compute sigmoid given linear term x
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// compute log of sigmoid given linear term x
double logsigmoid(double x) {
	if (x < -16.0) {
		return x;
	}
	else {
		return -1.0 * log(1 + exp(-x));
	}
}

// compute digamma function
double digamma(double x) {
	double result = 0, xx, xx2, xx4;
	if (x <= 0) {
		throw invalid_argument("digamma's argument could not less than or equal to 0.");
	}
	else {
		for (; x < 7; ++x)
			result -= 1 / x;
		x -= 1.0 / 2.0;
		xx = 1.0 / x;
		xx2 = xx * xx;
		xx4 = xx2 * xx2;
		result += log(x) + (1. / 24.)*xx2 - (7.0 / 960.0)*xx4 + (31.0 / 8064.0)*xx4*xx2 - (127.0 / 30720.0)*xx4*xx4;
		return result;
	}
}

// return a nonzeros array given y
VectorXd nonezeros(const MapVecd &y) {
	uint n = (uint)(y.array() > 0).count();
	VectorXd z = VectorXd::Ones(n);
	uint j = 0;
	for (uint i = 0; i < (uint)y.size(); i++) {
		if (y(i) > 0) {
			z(j) = y(i);
			j++;
		}
	}
	return z;
}

// print all the elements from a list
template<typename List>
void print_list(const List &x) {
	std::cout << "list: ";
	for (auto i : x) {
		std::cout << i << " ";
	}
	std::cout << std::endl;
}


// print all the elements from a vector
template<typename Vector>
void print_vector(const Vector &x) {
	std::cout << "vector: ";
	for (auto i : x) {
		std::cout << i << " ";
	}
	std::cout << std::endl;
}

// extract the nonzero elements from a VectorXd y
VectorXd nonezeros(const VectorXd &y) {
	unsigned int n = (unsigned int)(y.array() > 0).count();
	VectorXd z = VectorXd::Ones(n);
	unsigned int j = 0;
	for (unsigned int i = 0; i < y.size(); i++) {
		if (y(i) > 0) {
			z(j) = y(i);
			j++;
		}
	}
	return z;
}


#endif // !_UTILITY


