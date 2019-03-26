#ifndef _NEGBINSSVIIS_
#define _NEGBINSSVIIS_
#include <omp.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include <omp.h>
#include <gsl/gsl_sf_psi.h>
#include "RNG.hpp"
#include "utility.h"

using namespace std;
using namespace Eigen;
typedef unsigned int uint;

// https://github.com/pcarbo/varbvs/blob/73a0fe4ef0117ee9dd85025777c2f5bf801b0da8/varbvs-R/R/varbvs.R
// define Negative Binomial Variational Inference and Importance Sampling class
class NegBinSSVIIS {
private:
	uint seed;	  // global seed
	RNG r;		  // gsl sampler

	// Prior
	double sa0;   // slab variance
	double nu0;   // slab variance
	double a_r;   // over-dispersion r
	double b_r;   // over-dispersion r

	// Posterior of r
	double a_r_tilde;
	double b_r_tilde;

	// Data
	MatrixXd X;   // design matrix (n x p)
	VectorXd y;   // outcome y in counts (n x 1)
	uint p, n;    // sample and feature size

	// Parameters (Posterior Expectation) to be estimated
	VectorXd alpha;   // inclusion probability
	VectorXd mu;      // posterior mode of beta
	VectorXd s;       // posterior variance of beta

	double sa;        // posterior variance
	double logodds;   // prior log odds for pi
	double mu_r;      // over-dispersion parameter
	double mu_log_r;  // log of over-dispersion parameter
	double mu_L;      // CRT distribution parameter
	double logw;	  // log of ELBO
	double entropy;   // entropy of the model space
	double beta0;     // beta0
	double eps = 1e-10;  // for numerical stability of the update

	// temperate variables useful for inference
	VectorXd Xm;      // eta = Xm + beta0
	VectorXd eta;
	VectorXd d;
	VectorXd yhat;
	VectorXd xy;
	VectorXd xd;
	VectorXd xdx;
	VectorXd kappa;

	// define variable to keep track of the optimization
	vector<double> elbo_vec;	   // store all the ELBO through the VIEM optimization
	vector<double> entropy_vec;    // store all the entropy through the VIEM optimization

	// control the VIEM optimization
	uint maxiter = (uint)1e4;      // maximum number of iteration
	double tol = 1e-3;			   // minimum tolerance for program termination
	uint mode;					   // mode to terminate the program 0 for ELBO and 1 for entropy
	clock_t begin, end;
	double time_spent;			   // track time spent for sampling in seconds


public:

	NegBinSSVIIS(uint pseed);
	void set_data(const MatrixXd& X_data, const VectorXd& y_data);
	void set_prior(double pa_r, double pb_r, double plogodds, double psa0, double pnu0);
	void set_viem(uint pmaxiter, double ptol, uint pmode);
	
	double sum_log_exp(VectorXd &peta);
	VectorXd expt_omega(const VectorXd &eta, const VectorXd &y, const double r);  // compute the expectation of omega
	double expt_crt(const VectorXd &y, const double r);                           // compute the expectation of the number of tables in a CRT process
	VectorXd diagsq(const MatrixXd &X, const VectorXd &d);      // compute XtdX
	VectorXd comp_betavar(const VectorXd &alpha, const VectorXd &mu, const VectorXd &s);// compute beta variance

	// compute some statistics for VIEM
	void comp_negstats(const MatrixXd &X, const VectorXd &y, const double &r, const VectorXd &eta,
		VectorXd &d, VectorXd &s, const double &sa, VectorXd &yhat, VectorXd &xy,
		VectorXd &xd, VectorXd &xdx, VectorXd &kappa);

	// compute ELBO the evidence lower bound
	void comp_ELBO(const VectorXd &d, const VectorXd &yhat, const VectorXd &xdx, const VectorXd &kappa,
		const VectorXd &Xm, const double logodds, const VectorXd &alpha,
		const VectorXd &mu, const VectorXd &s,
		const double &sa, double &logw);
	void comp_entropy(const VectorXd &alpha, double &entropy);
	void init_viem();
	// set the parameters
	void set_param(double pr, double psa, double pbeta0, VectorXd &palpha, VectorXd &pmu, VectorXd &ps, VectorXd &pomega);
	void run_viem(bool verbose);    // run variational em
	void print_model();
	VectorXd get_elbo_vec();
	VectorXd get_entropy_vec();
	VectorXd get_alpha_vec();
	VectorXd get_mu_vec();
	VectorXd get_s_vec();
	VectorXd get_omega_vec();
	double get_beta0();
	double get_r();
	double get_sa();
	double get_logw(); // get ELBO
	double get_time_spent();
};

NegBinSSVIIS::NegBinSSVIIS(uint pseed) {
	seed = pseed;
	r.set(seed);
}

void NegBinSSVIIS::set_data(const MatrixXd& X_data, const VectorXd& y_data) {
	// check if data is valid or not
	p = (uint)X_data.cols();
	n = (uint)X_data.rows();
	if (n != ((uint)y_data.rows())) {
		throw invalid_argument("Number of rows in y should be equal to the number of rows in X");
	}

	X = X_data;
	y = y_data;
}

void NegBinSSVIIS::set_prior(double pa_r, double pb_r, double plogodds, double psa0, double pnu0) {
	//cout << "--------------------------------------------------" << endl;
	if (psa0 > 0 && pnu0 >0 && pa_r > 0 && pb_r > 0 && plogodds <= 0) {
		logodds = plogodds;
		sa0 = psa0;
		nu0 = pnu0;
		a_r = pa_r;
		b_r = pb_r;
	}
	else {
		throw std::invalid_argument("All prior parameters should be larger than 0.");
	}

	if (plogodds <= 0) {
		//printf("Prior on logodds of inclusion: logodds = %6.3f\n", logodds);
	}
	else {
		throw std::invalid_argument("Prior logodds should be smaller or equal to zero.");
	}
	//cout << "--------------------------------------------------" << endl;
}

void NegBinSSVIIS::set_viem(uint pmaxiter, double ptol, uint pmode) {
	//cout << "--------------------------------------------------" << endl;
	if (pmaxiter >0 && ptol > 0) {
		maxiter = pmaxiter;
		tol = ptol;
		//printf("VIEM: Iteration: %d\tTolerance: %f\n", maxiter, tol);
	}
	else {
		throw std::invalid_argument("maximum iteration (maxiter) and tolerance (tol) should be larger than 0.");
	}

	if (pmode == 0 || pmode == 1 || pmode == 2) {
		mode = pmode;
	}
	else {
		throw std::invalid_argument("termination method could be either ELBO (0) or entropy (1).");
	}
	//cout << "--------------------------------------------------" << endl;
}

// compute the sum of the log(1 + exp(z)) given a vector of linear term z
double NegBinSSVIIS::sum_log_exp(VectorXd &peta) {
	double sum = 0;
	double cz;
	for (uint i = 0; i < n; i++) {
		cz = peta(i);
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

// expectation of the augmented variables omega
VectorXd NegBinSSVIIS::expt_omega(const VectorXd &eta, const VectorXd &y, const double r) {
	return (y.array() + r).cwiseProduct((eta.array() / 2.0).tanh()) / (eta.array() * 2.0 + eps);
}

// expectation of the number of tables in Chinese Restaurant process
double NegBinSSVIIS::expt_crt(const VectorXd &y, const double r) {
	double val = 0.0f;
	for (unsigned int i = 0; i < y.size(); i++) {
		if (y(i) > 0) {
			val += r * (digamma(r + y(i)) - digamma(r));
		}
	}
	return std::min(val, y.sum());
	//return std::min(r * ((nonezeros(y).array() + r).array().unaryExpr(&digamma) - digamma(r)).sum(), y.sum());
}

// compute XtdX where d is the omega vector
VectorXd NegBinSSVIIS::diagsq(const MatrixXd &X, const VectorXd &d) {
	VectorXd t = VectorXd::Zero(p);

	for (uint i = 0; i < n; i++) {
		for (uint j = 0; j < p; j++) {
			t(j) += X(i, j) * X(i, j) * d(i);
		}
	}
	return t;
}

// compute beta variance
VectorXd NegBinSSVIIS::comp_betavar(const VectorXd &alpha, const VectorXd &mu, const VectorXd &s) {
	return alpha.array() * (s.array() + (1 - alpha.array()) * (mu.array().pow(2)));
}


// compute some statistics for VIEM
void NegBinSSVIIS::comp_negstats(const MatrixXd &X, const VectorXd &y, const double &r, const VectorXd &eta,
	VectorXd &d, VectorXd &s, const double &sa, VectorXd &yhat, VectorXd &xy,
	VectorXd &xd, VectorXd &xdx, VectorXd &kappa) {
	d = expt_omega(eta, y, r);
	kappa = (y.array() - r) / 2.0;
	yhat = kappa - (kappa.sum() / d.sum()) * d;
	xy = X.transpose() * yhat;
	xd = X.transpose() * d;

	// Compute the diagonal entries of X' * dhat * X
	VectorXd dzr = d / sqrt(d.sum());
	xdx = diagsq(X, d) - (X.transpose() * dzr).array().pow(2).matrix();
	s = sa * (sa * xdx.array() + 1).inverse().matrix();
}

// compute the evidence lower bound
void NegBinSSVIIS::comp_ELBO(const VectorXd &d, const VectorXd &yhat, const VectorXd &xdx, const VectorXd &kappa,
	const VectorXd &Xm, const double logodds, const VectorXd &alpha,
	const VectorXd &mu, const VectorXd &s,
	const double &sa, double &logw) {
	double a = 1.0 / d.sum();
	VectorXd temp = comp_betavar(alpha, mu, s);
	logw = 0.0;
	logw += log(a) / 2.0 + a * pow(kappa.sum(), 2) / 2.0 + yhat.transpose() * Xm;
	//logw += -sqrt((Xm.array().pow(2) * d.array()).sum()) / 2.0 + a * pow(d.transpose() * Xm, 2) / 2.0;
	logw += -(Xm.array().pow(2) * d.array()).sum() / 2.0 + a * pow(d.transpose() * Xm, 2) / 2.0;
	logw += -xdx.dot(temp) / 2; //double(xdx.transpose() * temp) / 2.0;
	logw += ((alpha.array() - 1) * (logodds) + logsigmoid(logodds)).sum();
	logw += (alpha.sum() + alpha.dot(((s / sa).array().log() - (s.array() + mu.array().pow(2)) / sa).matrix())) / 2.0;
	logw += -alpha.dot((alpha.array() + eps).log().matrix());
	logw += -(VectorXd::Ones(p) - alpha).dot((1 - alpha.array() + eps).log().matrix());
}

// compute the model entropy
void NegBinSSVIIS::comp_entropy(const VectorXd &alpha, double &entropy) {
	ArrayXd temp = (alpha.array() * (alpha.array().log() / log(2)) +
		(1 - alpha.array()) * ((1 - alpha.array()).log() / log(2)));
	entropy = -temp.isNaN().select(0, temp).sum(); // remove NaN in the temp
}

void NegBinSSVIIS::print_model() {
	cout << "model: ";
	for (uint k = 0; k < p; k++) {
		if (alpha(k) > 0.5) {
			cout << k << ", ";
		}
	}
	cout << endl;
}

void NegBinSSVIIS::init_viem() {
	// initialize parameter
	mu_r = 50;
	mu_log_r = log(mu_r);
	beta0 = 0;
	sa = 1;

	// initialize gsl sampler
	alpha = VectorXd::Zero(p);
	mu = VectorXd::Zero(p);
	r.norm(mu, 0, 0.01);
	r.flat(alpha, 0, 1);  // uniform between [0, 1] for each alpha
}

void NegBinSSVIIS::set_param(double pr, double psa, double pbeta0, VectorXd &palpha, VectorXd &pmu, VectorXd &ps, VectorXd &pomega) {
	mu_r = pr;
	mu_log_r = log(mu_r + eps);
	beta0 = pbeta0;
	sa = psa;

	alpha = palpha;
	mu = pmu;
	s = ps;
	d = pomega;
}


void NegBinSSVIIS::run_viem(bool verbose) {
	double logw0;    // keep track of the previous ELBO
	double entropy0; // keep track of the previous entropy
	double r_old, r_new; // here r = alpha(k) * mu(k)
	double SSR;		 // sum of squares due to regression
	double error;    // keep track of the error between alpha and alpha0
	VectorXd x;		 // one column of the design matrix X
	VectorXd alpha0; // keep track of the previous alpha

	// initialize all statistics
	Xm = (X * (alpha.cwiseProduct(mu)).matrix());
	// compute neg-bin statistics
	comp_negstats(X, y, mu_r, Xm + beta0 * VectorXd::Ones(n), d, s, sa, yhat, xy, xd, xdx, kappa);
	// compute ELBO
	comp_ELBO(d, yhat, xdx, kappa, Xm, logodds, alpha, mu, s, sa, logw);
	// compute entropy
	comp_entropy(alpha, entropy);
	// cout << "initial ELBO: " << logw << " Entropy " << entropy << endl;

	begin = clock();
	for (uint iter = 0; iter < maxiter; iter++) {
		logw0 = logw;
		entropy0 = entropy;
		alpha0 = alpha;
		// loop through all the variables
		for (uint k = 0; k < p; k++) {
			x = X.col(k);
			// update the VI of the posterior variance
			s(k) = sa / (sa * xdx(k) + 1);

			// update the VI of the posterior mean
			r_old = alpha(k) * mu(k);
			mu(k) = s(k) * (xy(k) + xdx(k) * r_old + xd(k) * d.dot(Xm) / d.sum() - x.cwiseProduct(d).dot(Xm));

			// update the posterior inclusion probability
			SSR = pow(mu(k), 2) / s(k);
			alpha(k) = sigmoid(logodds + (log(s(k) / sa) + SSR) / 2.0);

			// truncated the alpha level
			if (alpha(k) < tol) {
				alpha(k) = 0;
			}
			else if (alpha(k) > (1 - tol)) {
				alpha(k) = 1;
			}

			// update Xm
			r_new = alpha(k) * mu(k);
			Xm = Xm + (r_new - r_old) * x;
		}

		// update beta0
		beta0 = (kappa.sum() - d.transpose() * Xm) / d.sum();

		// update linear term eta
		eta = Xm.array() + beta0 + mu_log_r;

		// update L
		mu_L = expt_crt(y, exp(mu_log_r));

		// update r
		a_r_tilde = a_r + mu_L;
		b_r_tilde = b_r + sum_log_exp(eta); // (eta.array().exp() + 1).log().sum();
		mu_r = a_r_tilde / (b_r_tilde + eps);
		mu_log_r = digamma(a_r_tilde) - log(b_r_tilde + eps);

		// update the slab variance for beta
		sa = sa0 * nu0 + (alpha.dot((s.array() + mu.array().pow(2)).matrix())) / (nu0 + alpha.sum());

		// update negbin statistics
		comp_negstats(X, y, mu_r, Xm + beta0 * VectorXd::Ones(n), d, s, sa, yhat, xy, xd, xdx, kappa);

		// update the ELBO
		comp_ELBO(d, yhat, xdx, kappa, Xm, logodds, alpha, mu, s, sa, logw);
		elbo_vec.push_back(logw);

		// update the entropy
		comp_entropy(alpha, entropy);
		entropy_vec.push_back(entropy);
		
		// check convergence
		error = (alpha - alpha0).cwiseAbs().maxCoeff();

		if (abs(logw - logw0) < tol && mode == 0) { //  if ELBO converges then break
			break;
		}
		else if (error < tol && mode == 1) { // if maximum relative difference between alpha and alpha0 less than tol then break
			break;
		}
		else if (abs(entropy - entropy0) < tol && mode == 2) { // if relative difference between entropy and entropy0 less than tol then break
			break;
		}

		if (iter % 1000 == 0 && iter > 0 && verbose) {
			end = clock();
			printf("seed: %04d, iter: %05d, logw: %6.3f, entropy: %6.3f, r: %6.3f, beta0: %6.3f, sa: %6.3f | ETA: %6.3f\n",
				seed, iter, logw, entropy, mu_r, beta0, sa, double(end - begin) / (CLOCKS_PER_SEC) / (iter) * (maxiter - iter));
		}
	}

	// print program information
	end = clock();
	time_spent = double(end - begin) / (CLOCKS_PER_SEC);
	//cout << "Time takes: " << time_spent << endl;
	//print_model();
}


VectorXd NegBinSSVIIS::get_elbo_vec() {
	Map<VectorXd> ELBO(elbo_vec.data(), elbo_vec.size());
	return ELBO;
}

VectorXd NegBinSSVIIS::get_entropy_vec() {
	Map<VectorXd> ENTROPY(entropy_vec.data(), entropy_vec.size());
	return ENTROPY;
}

VectorXd NegBinSSVIIS::get_alpha_vec() {
	return alpha;
}

VectorXd NegBinSSVIIS::get_mu_vec() {
	return mu;
}

VectorXd NegBinSSVIIS::get_s_vec() {
	return s;
}


VectorXd NegBinSSVIIS::get_omega_vec() {
	return d;
}

double NegBinSSVIIS::get_beta0() {
	return beta0;
}

double NegBinSSVIIS::get_r() {
	return mu_r;
}

double NegBinSSVIIS::get_time_spent() {
	return time_spent;
}

double NegBinSSVIIS::get_logw() {
	return logw;
}

double NegBinSSVIIS::get_sa() {
	return sa;
}

// parallel computing with importance sampling on the outer loop

class parNegBinSSVIIS {
private:
	// Data
	MatrixXd X;   // design matrix (n x p)
	VectorXd y;   // outcome y in counts (n x 1)
	uint p, n;    // sample and feature size
	
	// Prior
	double sa0;   // slab variance
	double nu0;   // slab variance
	double a_r;   // over-dispersion r
	double b_r;   // over-dispersion r

	double max_logodds;
	double min_logodds;
	int num_logodds;   // number of important samples
	VectorXd logodds_vec;

	// variable to store the importance samples
	VectorXd w;		   // importance weights
	VectorXd logw_vec;
	VectorXd r_vec;
	VectorXd sa_vec;
	VectorXd beta0_vec;
	MatrixXd mu_mat;
	MatrixXd alpha_mat;
	MatrixXd s_mat;
	MatrixXd omega_mat;

	// outputs from importance sampling
	VectorXd pip;      // posterior inclusion probability
	VectorXd beta;	   // posterior estimation of beta
	VectorXd beta_sa;  // posterior estimation of beta variance

	// optimal variables from importance samples
	double r, sa, beta0, logw;
	VectorXd mu, alpha, s, omega;

	// control
	uint seed;
	int num_threads;
	// control the VIEM optimization
	uint maxiter = (uint)1e4;      // maximum number of iteration
	double tol = 1e-3;			   // minimum tolerance for program termination
	uint mode;					   // mode to terminate the program 0 for ELBO and 1 for entropy
	clock_t begin, end;
	double time_spent;			   // track time spent for sampling in seconds

public:
	parNegBinSSVIIS(uint pseed);
	void set_data(const MatrixXd& X_data, const VectorXd& y_data);
	void set_prior(double pa_r, double pb_r, double pmax_logodds, double pmin_logodds, int num_logodds, double psa0, double pnu0);
	void set_viem(uint pmaxiter, double ptol, uint pmode, int pnum_threads);
	void run_viem(bool verbose);
	VectorXd normalizelogweights(VectorXd logw);
	VectorXd get_pip();
	VectorXd get_beta();
	double get_time_spent();
	void print_model(VectorXd &alpha);
	VectorXd get_s_vec();
	VectorXd get_omega_vec();
	double get_beta0();
	double get_r();
	double get_sa();
	double get_logw(); // get ELBO
};

parNegBinSSVIIS::parNegBinSSVIIS(uint pseed) {
	seed = pseed;
	cout << "Initialize parallel importance sampling on prior logodds ..." << endl;
}

void parNegBinSSVIIS::set_data(const MatrixXd& X_data, const VectorXd& y_data) {
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

void parNegBinSSVIIS::set_prior(double pa_r, double pb_r, double max_plogodds, double min_plogodds, int num_plogodds, double psa0, double pnu0) {
	cout << "--------------------------------------------------" << endl;
	if (psa0 > 0 && pnu0 >0 && pa_r > 0 && pb_r > 0) {
		sa0 = psa0;
		nu0 = pnu0;
		a_r = pa_r;
		b_r = pb_r;
		
		printf("Prior on r: a_r = %6.3f\tb_r = %6.3f\n", a_r, b_r);
		printf("Prior on sig_b: sa0 = %6.3f\tnu0 = %6.3f\n", sa0, nu0);
	}
	else {
		throw std::invalid_argument("Prior parameters on both over-dispersion r and slab variance sa should be larger than 0.");
	}

	if (max_plogodds <= 0 && min_plogodds <= 0 && max_plogodds > min_plogodds) {
		max_logodds = max_plogodds;
		min_logodds = min_plogodds;
		printf("Importance sampling range for prior logodds is \n\t[%6.3f, %6.3f]\n", min_logodds, max_logodds);
	}
	else if (min_plogodds > max_plogodds) {
		throw std::invalid_argument("max_logodds should be larger than min_logodds.");
	}
	else if (max_plogodds > 0 || min_plogodds > 0) {
		throw std::invalid_argument("Prior logodds should be smaller or equal to zero.");
	}

	if (num_plogodds > 0) {
		num_logodds = num_plogodds;
		logodds_vec = VectorXd::LinSpaced(num_logodds, min_logodds, max_logodds);
		printf("Sample %d evenly spaced logodds equally in the above range.\n", num_logodds);
	}
	else {
		throw std::invalid_argument("Number of hyper-parameter logodds should be larger than 0.");
	}

	cout << "--------------------------------------------------" << endl;
}

void parNegBinSSVIIS::set_viem(uint pmaxiter, double ptol, uint pmode, int pnum_threads) {
	cout << "--------------------------------------------------" << endl;
	if (pmaxiter >0 && ptol > 0) {
		maxiter = pmaxiter;
		tol = ptol;
		printf("VIEM: Iteration: %d\tTolerance: %f\n", maxiter, tol);
	}
	else {
		throw std::invalid_argument("maximum iteration (maxiter) and tolerance (tol) should be larger than 0.");
	}

	if (pmode == 0 || pmode == 1 || pmode == 2) {
		mode = pmode;
		if (mode == 0) {
			cout << "termination method: ELBO" << endl;
		}
		else if (mode == 1) {
			cout << "termination method: error" << endl;
		}
		else if (mode == 2) {
			cout << "termination method: entropy" << endl;
		}
	}
	else {
		throw std::invalid_argument("termination method could be either ELBO (0) or error (1) or entropy (2).");
	}
	
	if (pnum_threads > 0) {
		num_threads = pnum_threads;
		printf("number of threads for parallel computing is %d\n", num_threads);
	}
	else {
		throw std::invalid_argument("number of threads for parallel computing should be larger than 0.");
	}
	cout << "--------------------------------------------------" << endl;
}


// normalize the importance weights
VectorXd parNegBinSSVIIS::normalizelogweights(VectorXd logw) {
	VectorXd w;
	double max_value = logw.maxCoeff();
	w = (logw.array() - max_value).exp();
	return w.array() / w.sum();
}

void parNegBinSSVIIS::run_viem(bool verbose) {
	VectorXd::Index argmax;
	logw_vec = VectorXd::Zero(num_logodds);
	r_vec = VectorXd::Zero(num_logodds);
	sa_vec = VectorXd::Zero(num_logodds);
	beta0_vec = VectorXd::Zero(num_logodds);
	mu_mat = MatrixXd::Zero(p, num_logodds);
	alpha_mat = MatrixXd::Zero(p, num_logodds);
	s_mat = MatrixXd::Zero(p, num_logodds);
	omega_mat = MatrixXd::Zero(n, num_logodds);
	begin = clock();
	// find the best set of initial parameters first outer loop
#pragma omp parallel for num_threads(num_threads)
	for (int k = 0; k < num_logodds; k++) {
		NegBinSSVIIS model(k);
		model.set_data(X, y);
		model.set_prior(a_r, b_r, logodds_vec(k), sa0, nu0);
		model.set_viem(maxiter, tol, 2);
		model.init_viem(); // initialize mu and alpha using seed k
		model.run_viem(verbose);
		
		// keep track of optimal parameters for each important sampling
		logw_vec(k) = model.get_logw();
		r_vec(k) = model.get_r();
		sa_vec(k) = model.get_sa();
		beta0_vec(k) = model.get_beta0();
		alpha_mat.col(k) = model.get_alpha_vec();
		mu_mat.col(k) = model.get_mu_vec();
		s_mat.col(k) = model.get_s_vec();
		omega_mat.col(k) = model.get_omega_vec();
	}

	// find the argmax of the ELBO vector & the best initial parameters
	logw_vec.maxCoeff(&argmax);
	r = r_vec(argmax);
	sa = sa_vec(argmax);
	beta0 = beta0_vec(argmax);
	alpha = alpha_mat.col(argmax);
	mu = mu_mat.col(argmax);
	s = s_mat.col(argmax);
	omega = omega_mat.col(argmax);
	
	// find the best set of initial parameters second outer loop
#pragma omp parallel for num_threads(num_threads)
	for (int k = 0; k < num_logodds; k++) {
		NegBinSSVIIS model(k);
		model.set_data(X, y);
		model.set_prior(a_r, b_r, logodds_vec(k), sa0, nu0);
		model.set_viem(maxiter, tol, mode);
		model.set_param(r, sa, beta0, alpha, mu, s, omega);
		model.run_viem(verbose);

		// keep track of optimal parameters for each important sampling
		logw_vec(k) = model.get_logw();
		r_vec(k) = model.get_r();
		sa_vec(k) = model.get_sa();
		beta0_vec(k) = model.get_beta0();
		alpha_mat.col(k) = model.get_alpha_vec();
		mu_mat.col(k) = model.get_mu_vec();
		s_mat.col(k) = model.get_s_vec();
		omega_mat.col(k) = model.get_omega_vec();
	}

	end = clock();
	time_spent = double(end - begin) / (CLOCKS_PER_SEC);
	cout << "Time takes: " << time_spent << endl;
	
	// create final output by average results from importance sampling
	w = normalizelogweights(logw_vec);

	// estimate final pip and beta coefficients
	pip = alpha_mat * w;
	beta = (alpha_mat.array() * mu_mat.array()).matrix() * w;
	omega = omega_mat * w;
	beta0 = beta0_vec.dot(w);
	r = r_vec.dot(w);
	sa = sa_vec.dot(w);
	logw = logw_vec.dot(w);
	cout << "final selected model by VIIS: ";
	print_model(pip); 
}


VectorXd parNegBinSSVIIS::get_pip() {
	return pip;
}

VectorXd parNegBinSSVIIS::get_beta() {
	return beta;
}

double parNegBinSSVIIS::get_time_spent() {
	return time_spent;
}

void parNegBinSSVIIS::print_model(VectorXd &alpha) {
	cout << "model: ";
	for (uint k = 0; k < p; k++) {
		if (alpha(k) > 0.5) {
			cout << k << ", ";
		}
	}
	cout << endl;
}

VectorXd parNegBinSSVIIS::get_omega_vec() {
	return omega;
}

double parNegBinSSVIIS::get_beta0() {
	return beta0;
}

double parNegBinSSVIIS::get_r() {
	return r;
}

double parNegBinSSVIIS::get_logw() {
	return logw;
}

double parNegBinSSVIIS::get_sa() {
	return sa;
}
#endif // !_NEGBINSSVIIS_