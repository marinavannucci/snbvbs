#ifndef _NEGBINSSMCMC_
#define _NEGBINSSMCMC_
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <list>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <gsl/gsl_sf_psi.h>
#include "RNG.hpp"
#include "PolyaGammaHybrid.h"
#include "utility.h"


using namespace std;
using namespace Eigen;
typedef unsigned int uint;

// define Negative Binomial Spike and Slab Sampler class
class NegBinSSMCMC {
private:
	// seed 
	uint seed;
	RNG r;
	PolyaGammaHybridDouble pg;

	// Prior
	double a_r, b_r;   // prior for the over-dispersion
	double a_pi, b_pi; // prior for the inclusion probability

	// Data
	MatrixXd X;   // design matrix (n x p)
	MatrixXd Xr;  // selected design matrix (n x m) 
	VectorXd y;   // outcome y in counts (n x 1)
	uint p, n, m; // sample, feature and model size

	// parameter (initialize with c for coefficient) 
	double cr;        // over-dispersion parameter
	double cbeta0;    // linear bias
	double ctau;      // global shrinkage coefficient
	VectorXd cbetas;  // beta coefficients (m x 1)
	VectorXd czs;     // linear term X * beta + beta0 (n x 1)
	VectorXd cws;     // augmented variable omega (n x 1)

	// some statistics
	VectorXd cks, hcks;  // (n x 1)
	MatrixXd hcW;        // (n x n) current normalization kernel
	VectorXd cl_beta;    // (m x 1) current linear part of the Gaussian beta
	MatrixXd cQ_beta;    // (m x m) current quadratic part of the Gaussian beta
	MatrixXd cQ_chol;    // (m x m) an upper triangular matrix Cholesky factor of cQ_beta

	// define variable storing the samples
	vector<uint> cmodel;              // a list of the current model (added variables)
	vector<uint> cdelete;             // a list of the current delete variables (deleted variables)
	list<VectorXd> betas_list;      // a list store all coefficient list
	list<vector<uint>> model_list;    // a list store all model list
	MatrixXd beta_mat;
	MatrixXd gamma_mat;
	MatrixXd omega_mat;
	VectorXd beta0_vec;
	VectorXd r_vec;
	VectorXd tau_vec;

	// control mcmc sampling
	bool add;
	uint n_sample;		// n mcmc samples 
	uint n_burnin;		// n burnin samples
	uint n_truncate;	// truncate level for Poisson approximation
	clock_t begin, end;
	double time_spent;	// track time spent for sampling in seconds

	double draw_crt_l_sum(VectorXd &pys, double pr);
	double sum_log_exp(VectorXd &pczs);
public:
	NegBinSSMCMC(uint pseed);
	void set_data(const MatrixXd& X_data, const VectorXd& y_data);
	void set_prior(double pa_r, double pb_r, double pa_pi, double pb_pi);
	void set_mcmc(uint pn_sample = 20000, uint pn_burnin = 3000, uint pn_truncate = 1000);
	void run_mcmc(bool verbose); // print sampling info or not
	double draw_crt_l(double py, double pr);
	MatrixXd Q_downdate(MatrixXd Q, uint d);      // delete the dth column and row in precision matrix Q
	MatrixXd Q_update(MatrixXd Q, MatrixXd &Xr, VectorXd x);    // append a new column and row at the end of the precision matrix Q
	MatrixXd X_downdate(MatrixXd X, uint d);      // delete the dth column of the design matrix X
	MatrixXd X_update(MatrixXd X, VectorXd x);    // append a new column at the end of the design matrix X
	void X_init();                                // initialize the design matrix X
	MatrixXd chol_delete(MatrixXd S, uint d);     // Cholesky update for delete step 
	MatrixXd chol_add(MatrixXd R, MatrixXd V, uint d); // cholesky update for add step
	MatrixXd chol_add(MatrixXd R, MatrixXd V);    // append a new column & row at the end of the Cholesky matrix
	double log_likelihood(const MatrixXd &Xr, const MatrixXd &chol_cS, double ctau);
	
	MatrixXd get_Q_chol();
	VectorXd get_hcks();
	
	// export the samples
	MatrixXd get_beta_mat();
	MatrixXd get_gamma_mat();
	MatrixXd get_omega_mat();
	VectorXd get_beta0_vec();
	VectorXd get_r_vec();
	VectorXd get_tau_vec();
	double get_time();

	void delete_move();
	void add_move();

};

NegBinSSMCMC::NegBinSSMCMC(uint pseed) {
	seed = pseed;
	r.set(seed);
}


void NegBinSSMCMC::set_data(const MatrixXd& X_data, const VectorXd& y_data) {
	// check if data is valid or not
	p = (uint)X_data.cols();
	n = (uint)X_data.rows();
	if (n != ((uint)y_data.rows())) {
		throw invalid_argument("Number of rows in y should be equal to the number of rows in X");
	}

	X = X_data;
	y = y_data;
	cout << "--------------------------------------------------" << endl;
	printf("Features: %d\tSamples: %d\n", p, n);
	cout << "--------------------------------------------------" << endl;
}

// set up control variables such as number of samples, burin samples and truncation approximation for the mcmc chains
void NegBinSSMCMC::set_mcmc(uint pn_sample, uint pn_burnin, uint pn_truncate) {
	if (pn_sample > 0 && pn_burnin > 0 && pn_truncate > 0) {
		n_sample = pn_sample;
		n_burnin = pn_burnin;
		n_truncate = pn_truncate;
		cout << "--------------------------------------------------" << endl;
		printf("MCMC: Samples: %d\tBurnin: %d\nTruncation: %d\tSeed: %d\n", pn_sample, pn_burnin, pn_truncate, seed);
		cout << "--------------------------------------------------" << endl;
	}
	else {
		throw std::invalid_argument("n_sample, n_burin and n_truncate should be larger than 0.");
	}
}


// set up hyper-priors parameters for beta binomial distribution
void NegBinSSMCMC::set_prior(double pa_r, double pb_r, double pa_pi, double pb_pi) {
	if (pa_r > 0 && pb_r > 0 && pa_pi > 0 && pb_pi > 0) {
		a_r = pa_r;
		b_r = pb_r;
		a_pi = pa_pi;
		b_pi = pb_pi;
		cout << "--------------------------------------------------" << endl;
		printf("Prior on r: a_r = %6.3f\tb_r = %6.3f\n", a_r, b_r);
		printf("Prior on pi: a_pi = %6.3f\tb_pi = %6.3f\n", a_pi, b_pi);
		cout << "--------------------------------------------------" << endl;
	}
	else {
		throw invalid_argument("All prior parameters should be larger than 0.");
	}
}


// draw the number of tables from CRP given one count y and over-dispersion r
double NegBinSSMCMC::draw_crt_l(double py, double pr) {
	double l = 0;
	if (pr <= 0) {
		throw invalid_argument("the over-dispersion parameter r could not be smaller than 0.");
		return (EXIT_FAILURE);
	}
	if (py == 0) {
		l = 0;
	}
	else if (py > n_truncate) {
		// if py much larger then approximate l from drawing a Poisson
		l = pr * (gsl_sf_psi(py + pr) - gsl_sf_psi(pr));
		l = gsl_ran_poisson(r.r, l);
	}
	else {
		// draw l directly from a Chinese Restaurant Process
		for (uint i = 0; i < (uint)py; i++) {
			l += (gsl_rng_uniform(r.r) <= (pr / (pr + i)));
		}
	}
	return l;
}


// draw the sum of number of tables from CRP given count vector y and over-dispersion r
double NegBinSSMCMC::draw_crt_l_sum(VectorXd &pys, double pr) {
	double  l_sum = 0;
	for (uint i = 0; i < n; i++) {
		l_sum += draw_crt_l(pys(i), pr);
	}
	return l_sum;
}

// compute the sum of the log(1 + exp(z)) given a vector of linear term z
double NegBinSSMCMC::sum_log_exp(VectorXd &pczs) {
	double sum = 0;
	double cz;
	for (uint i = 0; i < n; i++) {
		cz = pczs(i);
		if (cz > 35) {
			sum += cz;
		}
		else if (cz < -10) {
			sum += exp(cz);
		}
		else {
			sum += log(1 + exp(cz));
		}
	}
	return sum;
}

// delete the dth row and column of covariance or Cholesky factor matrix Q
MatrixXd NegBinSSMCMC::Q_downdate(MatrixXd Q, uint d) {
	uint dim = (uint)Q.rows();
	if (dim == 0) {
		std::cerr << "Q is an empty matrix, can not downdate any more." << endl;
	}
	else if (d > (dim - 1)) {
		std::cerr << "The valid range of d is from 0 to " << dim - 1 << "\nWe drop the last column and row of Q instead." << endl;
	}
	// copy rows
	for (uint i = d; i < dim - 1; i++) {
		Q.row(i) = Q.row(i + 1);
	}

	// copy columns
	for (uint i = d; i < dim - 1; i++) {
		Q.col(i) = Q.col(i + 1);
	}
	Q.conservativeResize(dim - 1, dim - 1);
	return Q;
}

// append a new column and row to the old precision matrix Q
MatrixXd NegBinSSMCMC::Q_update(MatrixXd Q, MatrixXd &Xr, VectorXd x) {
	uint dim = (uint)Q.rows();
	Q.conservativeResize(dim + 1, dim + 1);
	Q(dim, dim) = x.transpose() * hcW * x;
	Q.col(dim).head(dim) = Xr.transpose() * hcW * x;
	Q.row(dim).head(dim) = Q.col(dim).head(dim);
	return Q;
}


// initialize the design matrix X to be (n x m) from a sampled cmodel vector
void NegBinSSMCMC::X_init() {
	Xr = MatrixXd::Zero(n, cmodel.size());
	uint k = 0;
	for (auto it = cmodel.begin(); it != cmodel.end(); it++, k++) {
		Xr.col(k) = X.col(*it);
	}
}

// delete the dth column of design matrix X 
// X is a (n x m) matrix with the features in the columns
MatrixXd NegBinSSMCMC::X_downdate(MatrixXd X, uint d) {
	uint dim = (uint)X.cols();
	if (dim == 0) {
		std::cerr << "X is an empty matrix, can not downdate any more." << endl;
	}
	else if (d > (dim - 1)) {
		std::cerr << "The valid range of d is from 0 to " << dim - 1 << "\nWe drop the last column of X instead." << endl;
	}
	
	// copy columns
	for (uint i = d; i < dim - 1; i++) {
		X.col(i) = X.col(i + 1);
	}
	X.conservativeResize(X.rows(), dim - 1);
	return X;
}

// append a new column x at the end of the design matrix X
// X is a (n x m) matrix with the features in the columns
MatrixXd NegBinSSMCMC::X_update(MatrixXd X, VectorXd x) {
	uint dim = (uint)X.cols();
	if (x.size() != X.rows()) {
		std::cerr << "The number of samples in design matrix should match up the size of new column x." << endl;
	}
	// change the dimension of X
	X.conservativeResize(X.rows(), dim + 1);
	// append the new column X at the end of X
	X.col(dim) = x;
	return X;
}


// given old Cholesky factor S and row or column index d, return new factor R
MatrixXd NegBinSSMCMC::chol_delete(MatrixXd S, uint d) {
	uint dim = (uint)S.rows();
	MatrixXd R = MatrixXd::Zero(dim - 1, dim - 1);
	if (d == (dim - 1)) { // delete a column at the end of the design matrix
		R = S.topLeftCorner(dim - 1, dim - 1);
	}
	else if (d < (dim - 1) && d > 0) {  // delete a column at (0, dim) of the design matrix
		MatrixXd S11 = S.topLeftCorner(d, d);
		MatrixXd S13 = S.topRightCorner(d, dim - d - 1);
		MatrixXd S23 = S.row(d).tail(dim - d - 1);
		MatrixXd S33 = S.bottomRightCorner(dim - d - 1, dim - d - 1);
		LLT<MatrixXd> R33(S23.transpose() * S23 + S33.transpose() * S33);
		R.topLeftCorner(d, d) = S11;
		R.topRightCorner(d, dim - d - 1) = S13;
		R.bottomRightCorner(dim - d - 1, dim - d - 1) = R33.matrixU();
	}
	else if (d == 0) { // delete a column at the beginning of the design matrix
		MatrixXd S33 = S.bottomRightCorner(dim - 1, dim - 1);
		MatrixXd S23 = S.row(0).tail(dim - 1);
		LLT<MatrixXd> R33(S23.transpose() * S23 + S33.transpose() * S33);
		R = R33.matrixU();
	}
	else {
		std::cerr << "deletion d (" << d << ") must within the range from 0 to " << dim - 1 << endl;
	}
	return R;
}

// compute the new Cholesky factor when add one column and row to the old factor
// given old Cholesky factor R, new added matrix V and row or column index d return new Cholesky factor S
// however, we always add new variable in the end or d = nrow(R)
MatrixXd NegBinSSMCMC::chol_add(MatrixXd R, MatrixXd V, uint d) {
	uint dim = (uint) R.rows();
	MatrixXd S = MatrixXd::Zero(dim + 1, dim + 1);
	
	cout << dim << endl;
	if (dim == 0) {
		S = V.array().sqrt();
	}
	else if (d == dim) { // append a new column at the end of the design matrix
		MatrixXd S11 = R;
		VectorXd V12 = V.topRightCorner(dim, 1);
		VectorXd S12 = S11.triangularView<Eigen::Upper>().adjoint().solve(V12);
		double V22 = V(dim, dim);
		double S22 = sqrt(V22 - S12.transpose() * S12);

		S.topLeftCorner(dim, dim) = R;
		S.topRightCorner(dim, 1) = S12;
		S(dim, dim) = S22;
	}
	else if (d == 0) { // append a new column at the beginning of the design matrix
		double V22 = V(0, 0);
		VectorXd V23 = V.row(0).tail(dim);
		double S22 = sqrt(V22);
		VectorXd S23 = V23.array() / S22;
		LLT<MatrixXd> S33(R.transpose() * R - S23 * S23.transpose());
		S(0, 0) = S22;
		S.row(0).tail(dim) = S23;
		S.bottomRightCorner(dim, dim) = S33.matrixU();
	}

	// TODO Update Cholesky factor for column at (0, dim)
	/*
	else if (d > 0 && d < dim) { // append a new column at (0, dim) of the design matrix
		MatrixXd R11 = R.topLeftCorner(d, d);
		VectorXd V12 = R.col(d).head(d);
		VectorXd S12 = R11.triangularView<Eigen::Upper>().adjoint().solve(V12);
		MatrixXd S13 = R.topRightCorner(d, dim - d);
		double V22 = V(d, d);
		double S22 = sqrt(V22 - S12.transposeInPlace() * S12);
		MatrixXd V23 = V.row(d).tail(dim - d);
		MatrixXd S23 = (V23 - S12 * S13).array() / S22;
		MatrixXd R33 = R.bottomRightCorner(dim - d, dim - d);
		LLT<MatrixXd> S33(R.transpose() * R - S23 * S23.transpose());
		
		// update the Cholesky factor S
		S.topLeftCorner(d, d) = R11;
		S.col(d).head(d) = S12;
		S.topRightCorner(d, dim - d) = S13;
		S(d, d) = S22;
		S.row(d).tail(dim - d) = S23;
		S.bottomRightCorner(dim - d, dim - d) = S33.matrixU();
	}*/
	
	return S;
}

// compute the new Cholesky factor when add one column and row to the old factor
// given old Cholesky factor R, new added matrix V and row or column index d return new Cholesky factor S
// however, we always add new variable in the end or d = nrow(R)
MatrixXd NegBinSSMCMC::chol_add(MatrixXd R, MatrixXd V) {
	uint dim = (uint)R.rows();
	MatrixXd S = MatrixXd::Zero(dim + 1, dim + 1);

	if (dim == 0) {
		S = V.array().sqrt();
	}
	else { // append a new column at the end of the design matrix
		MatrixXd S11 = R;
		VectorXd V12 = V.topRightCorner(dim, 1);
		VectorXd S12 = S11.triangularView<Eigen::Upper>().adjoint().solve(V12);
		double V22 = V(dim, dim);
		double S22 = sqrt(V22 - S12.transpose() * S12);

		S.topLeftCorner(dim, dim) = R;
		S.topRightCorner(dim, 1) = S12;
		S(dim, dim) = S22;
	}
	return S;
}

// compute the log_likelihood of the model 
// given updated design matrix Xr, its associated Cholesky factor chol_cS and normalized y hcK
double NegBinSSMCMC::log_likelihood(const MatrixXd &Xr, const MatrixXd &chol_cS, double ctau) {
	uint m = (uint)Xr.cols();
	uint n = (uint)Xr.rows();
	VectorXd temp;
	double SSR = 0.0, ldS = 0.0;
	if (m != 0) { // if model space is not empty
		temp = chol_cS.triangularView<Eigen::Upper>().adjoint().solve(Xr.transpose() * hcks);
		SSR = temp.transpose() * temp;     // sum of square due to the regression
		ldS = -log(chol_cS.determinant());
	}
	else {
		SSR = 1.0 / ctau;
		ldS = -log(ctau) / 2.0;
	}
	return m * log(ctau) / 2.0 + ldS + (SSR) / 2.0;
}


// delete move for the spike and slab metropolis hastings
void NegBinSSMCMC::delete_move() {
	vector<uint> pmodel = cmodel;
	vector<uint> pdelete = cdelete;

	// compute the base log-likelihood
	double base_log_like = log_likelihood(Xr, cQ_chol, ctau);
	double proposed_log_like;    // propose log-likelihood from the deletion step
	double ratio1, ratio2, ratio, accept_ratio;
	uint k = 0;
	MatrixXd cdQ_chol, caQ_chol; // define the deleted Q_chol and added Q_chol matrix
	MatrixXd cdQ_beta, cdXr;     // define the deleted Q_beta (the precision matrix) and deleted design matrix cdXr
	MatrixXd::Index argmax;
	VectorXd delete_log_like1 = VectorXd::Zero(cmodel.size());
	VectorXd delete_prob = VectorXd::Zero(cmodel.size());
	VectorXd delete_log_like2 = VectorXd::Zero(cdelete.size() + 1);

	// loop through the model space to determine which variable to delete
	for (auto it = cmodel.begin(); it != cmodel.end(); it ++, k++) {
		delete_log_like1[k] = log_likelihood(X_downdate(Xr, k), chol_delete(cQ_chol, k), ctau);
	}

	// compute ratio1 shown in the numerator of the equation (61)
	ratio1 = (delete_log_like1.array() - base_log_like).exp().sum();
	//cout << "ratio1 computed successful!" << endl;

	delete_prob = delete_log_like1.array() - delete_log_like1.maxCoeff();
	delete_prob = delete_prob.array().exp() / delete_prob.array().exp().sum();
	delete_prob.maxCoeff(&argmax); // find the argmax of the delete_prob vector
	// note that argmax is the relative index from the cmodel vector
	proposed_log_like = delete_log_like1((uint)argmax);
	ratio = exp(proposed_log_like - base_log_like);
	//cout << "ratio computed successful!" << endl;

	// update the Cholesky factor given the model proposal
	cdQ_chol = chol_delete(cQ_chol, (uint)argmax);
	cdQ_beta = Q_downdate(cQ_beta, (uint)argmax);
	cdXr = X_downdate(Xr, (uint)argmax);

	// delete the argmax th from the model space and add it to the delete vector
	// then we have the proposed model and delete vectors
	pmodel.erase(pmodel.begin() + (uint)argmax);
	pdelete.push_back(cmodel.at((uint)argmax));

	k = 0;
	for (auto it = pdelete.begin(); it != pdelete.end(); it++, k++) {
		delete_log_like2[k] = log_likelihood(X_update(cdXr, X.col(*it)), chol_add(cdQ_chol, Q_update(cdQ_beta, cdXr, X.col(*it))), ctau);
	}
	ratio2 = (delete_log_like2.array() - proposed_log_like).exp().sum();
	//cout << "ratio2 computed successful!" << endl;
	accept_ratio = (ratio * ratio1 / (ratio2)) * (b_pi + p - m) / (a_pi + m - 1);

	if (gsl_rng_uniform(r.r) < accept_ratio) {
		// accepted the proposal and delete one variable
		m -= 1;
		Xr = cdXr;
		cQ_chol = cdQ_chol;
		cmodel = pmodel;
		cdelete = pdelete;
		
		// downdate cbetas & delete the corresponding argmax entry
		for (uint i = (uint)argmax; i < m; i++) {
			cbetas(i) = cbetas(i + 1);
		}
		cbetas.conservativeResize(m);
	}
}


// add move for the spike and slab metropolis hastings
void NegBinSSMCMC::add_move() {
	vector<uint> pmodel = cmodel;
	vector<uint> pdelete = cdelete;
	// compute the base log-likelihood
	double base_log_like = log_likelihood(Xr, cQ_chol, ctau);
	double proposed_log_like;    // proposed log-likelihood from the addition step
	double ratio, ratio1, ratio2, accept_ratio;
	uint k = 0, add_idx;
	MatrixXd caQ_chol, cdQ_chol; // define the added Q_chol and the deleted Q_chol matrix respectively
	MatrixXd caQ_beta, caXr;     // define the added Q_beta (the precision matrix) and the added design matrix
	MatrixXd::Index argmax;
	VectorXd add_log_like1 = VectorXd::Zero(pdelete.size());	// added model log-likelihood1
	VectorXd add_prob = VectorXd::Zero(pdelete.size());			// added model probability
	VectorXd add_log_like2 = VectorXd::Zero(pmodel.size() + 1); // added model log-likelihood2

	// loop through the delete space to determine which variable to add
	for (auto it = pdelete.begin(); it != pdelete.end(); it++, k++) {
		add_log_like1[k] = log_likelihood(X_update(Xr, X.col(*it)), 
			chol_add(cQ_chol, Q_update(cQ_beta, Xr, X.col(*it))), ctau);
	}

	// compute the ratio one in the numerator shown in equation (60)
	ratio1 = (add_log_like1.array() - base_log_like).exp().sum();

	// compute the addition probability
	add_prob = add_log_like1.array() - add_log_like1.maxCoeff();
	add_prob = add_prob.array().exp() / add_prob.array().exp().sum();
	add_prob.maxCoeff(&argmax);  // find the argmax of the add_prob vector relative index
	proposed_log_like = add_log_like1((uint)argmax);
	ratio = exp(proposed_log_like - base_log_like);

	add_idx = pdelete.at(argmax);   // absolute index

	// update the Cholesky factor given the model proposal using the absolute index
	caQ_chol = chol_add(cQ_chol, Q_update(cQ_beta, Xr, X.col((uint) add_idx)));
	caXr = X_update(Xr, X.col((uint) add_idx));

	// delete the argmax th from the delete space and add it to the model vector
	// then we have the proposed model and delete vectors
	pdelete.erase(pdelete.begin() + (uint)argmax);
	pmodel.push_back(add_idx);

	k = 0;
	for (auto it = pmodel.begin(); it != pmodel.end(); it++, k++) {
		add_log_like2[k] = log_likelihood(X_downdate(caXr, k), chol_delete(caQ_chol, k), ctau);
	}
	ratio2 = (add_log_like2.array() - proposed_log_like).exp().sum();
	accept_ratio = (ratio * ratio1 / (ratio2)) * (a_pi + p) / (b_pi + p - m + 1);
	if (gsl_rng_uniform(r.r) < accept_ratio) {
		// accepted the proposal and delete one variable
		m += 1;
		Xr = caXr;
		cQ_chol = caQ_chol;
		cmodel = pmodel;
		cdelete = pdelete;

		// update cbetas & delete the corresponding argmax entry
		cbetas.conservativeResize(m);
		cbetas(m - 1) = 0; // append a zero coefficient
	}
}


void NegBinSSMCMC::run_mcmc(bool verbose) {
	// set the seed
	r.set(seed);
	uint iter;
	double pctau;			     // proposed ctau
	double accept_ratio = 0;     // acceptance probability
	int s, k;                    // sample index s and coefficient index k
	// clear parameter
	cws = VectorXd::Zero(n);
	cks = VectorXd::Zero(n);
	hcks = VectorXd::Zero(n);
	hcW = MatrixXd::Zero(n, n);  // normalization kernel

	// create space for samples
	omega_mat = MatrixXd::Zero(n_sample, n);
	beta_mat = MatrixXd::Zero(n_sample, p);
	gamma_mat = MatrixXd::Zero(n_sample, p);
	r_vec = VectorXd::Zero(n_sample);
	beta0_vec = VectorXd::Zero(n_sample);
	tau_vec = VectorXd::Zero(n_sample);
	
	
	/*
	// initialize the model with all variables included (n x p)
	for (uint i = 0; i < p; i++) {
		cmodel.push_back(i);
	}
	Xr = X;
	*/
	
	// initialize a non-full model with the given prior
	do {
		for (uint i = 0; i < p; i++) {
			if (gsl_rng_uniform(r.r) < (a_pi / (a_pi + b_pi))) {
				cmodel.push_back(i);
			}
			else {
				cdelete.push_back(i);
			}
		}
		m = (uint)cmodel.size();
	} while (m == 0);
	X_init();  // initialize the selected design matrix Xr given the current model
	
	// initialize a full model
	cr = 500;
	cbeta0 = 0;
	ctau = 10;

	cbetas = VectorXd::Zero(m);
	r.norm(cbetas, .1);   

	cout << "init model size " << m << endl;
	// start mcmc
	begin = clock();
	for (iter = 0; iter < n_sample + n_burnin; iter++) {
		// compute the linear term vector (n x 1)
		czs = (Xr * cbetas).array() + cbeta0 + log(cr);

		// update omega vector (n by 1)
		for (uint i = 0; i < n; i++) {
			cws(i) = pg.draw(y(i) + cr, czs(i) - log(cr), r);
		}

		// update over-dispersion scalar r
		cr = gsl_ran_gamma(r.r, a_r + draw_crt_l_sum(y, cr), 1 / (b_r + sum_log_exp(czs)));

		// update some statistics
		hcW = MatrixXd(cws.asDiagonal()) - (cws * cws.transpose()) / cws.sum();
		cks = (y.array() - cr) / 2;
		hcks = cks - (cks.sum() / cws.sum()) * cws;  // normalized version of cks

		if (m != 0) {
			// update beta vector (m x m)
			cQ_beta = (Xr.transpose() * hcW * Xr); // (m x m)
			cQ_beta.diagonal().array() += ctau;  // add a constant shrinkage to the diagonal of the matrix 
			cl_beta = Xr.transpose() * hcks;

			cbetas = VectorXd::Zero(m);
			LLT<MatrixXd> chol_Q(cQ_beta);  // lower triangular Cholesky decomposition of Q_beta
			r.norm(cbetas, 1); // draw a vector of standard normal (m x 1)
			
			// forward & backward solve for betas
			cbetas = chol_Q.matrixU().solve(chol_Q.matrixL().solve(cl_beta) + cbetas);
			cQ_chol = chol_Q.matrixU();
		}

		// update the intercept beta0
		//cbeta0 = (cks.array() - cws.array() * (Xr * cbetas).array()).sum() / cws.sum();
		cbeta0 = (cks.sum() - cws.transpose() * (Xr * cbetas)) / (cws).sum();
		
		// randomly choose between add_move and delete_move
		if (m == 1) {
			add = true;
		} else if (m == p) {
			add = false;
		}
		else {
			add = gsl_ran_bernoulli(r.r, 0.05);
		}
		
		if (add) {
			add_move();
			//cout << "add move " << endl;
		}
		else {
			delete_move();
			//cout << "delete move " << endl;
		}

	
		// update the shrinkage parameter ctau using a MH update on the log scale
		pctau = ctau * exp(gsl_ran_gaussian(r.r, .01));
		accept_ratio = exp(log_likelihood(Xr, cQ_chol, pctau) - log_likelihood(Xr, cQ_chol, ctau));
		if (gsl_rng_uniform(r.r) < accept_ratio) { // accept
			ctau = pctau;
		}

		if (ctau > 5) { // provent the slab variance to be too small
			ctau = 5;
		}

		if (iter % 1000 == 0 && iter > 0 && verbose) {
			end = clock();
			printf("seed: %04d, iter: %05d, r: %6.3f, beta0: %6.3f, tau: %6.3f, m: %02d | ETA: %6.3f\n",
				seed, iter, cr, cbeta0, ctau, m, double(end - begin) / (CLOCKS_PER_SEC) / (iter) * (n_sample + n_burnin - iter));
			print_vector(cmodel);
		}

		// save samples after the burnin period
		if (iter > n_burnin - 1) {
			s = (int)(iter - n_burnin);
			// save betas
			k = 0;
			for (auto it = cmodel.begin(); it != cmodel.end(); it++, k++) {
				beta_mat(s, *(it)) = cbetas(k);
				gamma_mat(s, *(it)) = 1;
			}

			omega_mat.row(s) = cws;          // save augmented variable omega
			r_vec(s) = cr;                   // save over-dispersion parameter r
			beta0_vec(s) = cbeta0;           // save baseline or intercept term beta0
			tau_vec(s) = ctau;               // save global shrinkage tau
		}
	}

	// print program information
	end = clock();
	time_spent = double(end - begin) / (CLOCKS_PER_SEC);
	cout << "Time takes: " << time_spent << endl;
	
	/*	
	// delete one column and row in the precision matrix
	// check cholesky factor update at column 0
	cout << "cholesky update at column 0: " << endl;
	LLT<MatrixXd> chol_old(Q_downdate(Q_beta, 0));
	cout << chol_add(chol_old.matrixU(), Q_beta, 0) << endl;
	cout << "correct matrix" << endl;
	cout << Q_chol << endl;

	// check cholesky factor update at column 19
	cout << "cholesky update at column 19: " << endl;
	cout << "updated cholesky factor:" << endl;
	cout << chol_add(Q_chol.topLeftCorner(19, 19), Q_beta, 19) << endl;
	cout << "correct matrix" << endl;
	cout << Q_chol << endl;
	*/
}


MatrixXd NegBinSSMCMC::get_Q_chol() {
	return cQ_chol;
}

VectorXd NegBinSSMCMC::get_hcks() {
	return hcks;
}

/* export the samples to python*/
MatrixXd NegBinSSMCMC::get_beta_mat() {
	return beta_mat;
}

MatrixXd NegBinSSMCMC::get_gamma_mat() {
	return gamma_mat;
}


MatrixXd NegBinSSMCMC::get_omega_mat() {
	return omega_mat;
}

VectorXd NegBinSSMCMC::get_beta0_vec() {
	return beta0_vec;
}

VectorXd NegBinSSMCMC::get_r_vec() {
	return r_vec;
}

VectorXd NegBinSSMCMC::get_tau_vec() {
	return tau_vec;
}

double NegBinSSMCMC::get_time() {
	return time_spent;
}

#endif // !_NEGBINSSMCMC_