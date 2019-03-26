#ifndef _NEGBINSSVIEM_
#define _NEGBINSSVIEM_
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include <gsl/gsl_sf_psi.h>
#include "RNG.hpp"
#include "utility.h"


using namespace std;
using namespace Eigen;
typedef unsigned int uint;

// define Negative Binomial Varitional Inference and EM class
class NegBinSSVIEM {
private:
	uint seed;	  // global seed
	RNG r;		  // gsl sampler

	// Prior
	double a_pi;  // pi
	double b_pi;  // pi
	double sa0;   // slab variance
	double nu0;   // slab variance
	double a_r;   // overdisperison r
	double b_r;   // overdispersion r

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
	VectorXd logodds; // log odds of alpha
	VectorXd s;       // posterior variance of beta

	double sa;        // posterior variance
	double mu_r;      // overdispersion parameter
	double mu_log_r;  // log of overdispersion parameter
	double mu_L;      // CRT distribution parameter
	double logw;	  // log of ELBO
	double entropy;   // entropy of the model space
	double beta0;     // beta0
	double eps = 1e-10;  // for numerical stablility of the update

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
	vector<double> elbo_vec;	   // store all the elbo through the VIEM optimization
	vector<double> entropy_vec;    // store all the entropy through the VIEM optimization
	vector<double> error_vec;      // store all the errors through the VIEM optimization

	// control the VIEM optimization
	uint maxiter = (uint)1e4;      // maximum number of iteration
	double tol = 1e-3;			   // minimum tolerance for program termination
	uint mode;					   // mode to terminate the program 0 for ELBO and 1 for entropy
	clock_t begin, end;
	double time_spent;			   // track time spent for sampling in seconds


public:

	NegBinSSVIEM(uint pseed);
	void set_data(const MatrixXd& X_data, const VectorXd& y_data);
	void set_prior(double pa_r, double pb_r, double pa_pi, double pb_pi, double psa0, double pnu0);
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
		const VectorXd &Xm, const VectorXd &logodds, const VectorXd &alpha,
		const VectorXd &mu, const VectorXd &s,
		const double &sa, double &logw);
	void comp_entropy(const VectorXd &alpha, double &entropy);
	void run_viem(bool verbose);    // run variational em
	void print_model();
	VectorXd get_elbo_vec();
	VectorXd get_entropy_vec();
	VectorXd get_alpha_vec();
	VectorXd get_beta_vec();
	VectorXd get_omega_vec();
	double get_beta0();
	double get_r();
	double get_sa();
	double get_time_spent();
};

NegBinSSVIEM::NegBinSSVIEM(uint pseed) {
	seed = pseed;
	r.set(seed);
}

void NegBinSSVIEM::set_data(const MatrixXd& X_data, const VectorXd& y_data) {
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

void NegBinSSVIEM::set_prior(double pa_r, double pb_r, double pa_pi, double pb_pi, double psa0, double pnu0) {
	if (pa_pi > 0 && pb_pi > 0 && psa0 > 0 && pnu0 >0 && pa_r > 0 && pb_r > 0) {
		a_pi = pa_pi;
		b_pi = pb_pi;
		sa0 = psa0;
		nu0 = pnu0;
		a_r = pa_r;
		b_r = pb_r;
		cout << "--------------------------------------------------" << endl;
		printf("Prior on r: a_r = %6.3f\tb_r = %6.3f\n", a_r, b_r);
		printf("Prior on pi: a_pi = %6.3f\tb_pi = %6.3f\n", a_pi, b_pi);
		printf("Prior on sig_b: sa0 = %6.3f\tnu0 = %6.3f\n", sa0, nu0);
		cout << "--------------------------------------------------" << endl;
	}
	else {
		throw std::invalid_argument("All prior parameters should be larger than 0.");
	}
}

void NegBinSSVIEM::set_viem(uint pmaxiter, double ptol, uint pmode) {
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
	cout << "--------------------------------------------------" << endl;
}

// compute the sum of the log(1 + exp(z)) given a vector of linear term z
double NegBinSSVIEM::sum_log_exp(VectorXd &peta) {
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
VectorXd NegBinSSVIEM::expt_omega(const VectorXd &eta, const VectorXd &y, const double r) {
	return (y.array() + r).cwiseProduct((eta.array() / 2.0).tanh()) / (eta.array() * 2.0 + eps);
}

// expectation of the number of tables in Chinese Restaurant process
double NegBinSSVIEM::expt_crt(const VectorXd &y, const double r) {
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
VectorXd NegBinSSVIEM::diagsq(const MatrixXd &X, const VectorXd &d) {
	VectorXd t = VectorXd::Zero(p);

	for (uint i = 0; i < n; i++) {
		for (uint j = 0; j < p; j++) {
			t(j) += X(i, j) * X(i, j) * d(i);
		}
	}
	return t;
}

// compute beta variance
VectorXd NegBinSSVIEM::comp_betavar(const VectorXd &alpha, const VectorXd &mu, const VectorXd &s) {
	return alpha.array() * (s.array() + (1 - alpha.array()) * (mu.array().pow(2)));
}


// compute some statistics for VIEM
void NegBinSSVIEM::comp_negstats(const MatrixXd &X, const VectorXd &y, const double &r, const VectorXd &eta,
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
void NegBinSSVIEM::comp_ELBO(const VectorXd &d, const VectorXd &yhat, const VectorXd &xdx, const VectorXd &kappa,
	const VectorXd &Xm, const VectorXd &logodds, const VectorXd &alpha,
	const VectorXd &mu, const VectorXd &s,
	const double &sa, double &logw) {
	double a = 1.0 / d.sum();
	VectorXd temp = comp_betavar(alpha, mu, s);
	logw = 0.0;
	logw += log(a) / 2.0 + a * pow(kappa.sum(), 2) / 2.0 + yhat.transpose() * Xm;
	//logw += -sqrt((Xm.array().pow(2) * d.array()).sum()) / 2.0 + a * pow(d.transpose() * Xm, 2) / 2.0;
	logw += -(Xm.array().pow(2) * d.array()).sum() / 2.0 + a * pow(d.transpose() * Xm, 2) / 2.0;
	logw += -xdx.dot(temp) / 2; //double(xdx.transpose() * temp) / 2.0;
	logw += ((alpha - VectorXd::Ones(p)).dot(logodds) + logodds.unaryExpr(&logsigmoid).sum());
	logw += (alpha.sum() + alpha.dot(((s / sa).array().log() - (s.array() + mu.array().pow(2)) / sa).matrix())) / 2.0;
	logw += -alpha.dot((alpha.array() + eps).log().matrix());
	logw += -(VectorXd::Ones(p) - alpha).dot((1 - alpha.array() + eps).log().matrix());
}

// compute the model entropy
void NegBinSSVIEM::comp_entropy(const VectorXd &alpha, double &entropy) {
	ArrayXd temp = (alpha.array() * (alpha.array().log() / log(2)) +
		(1 - alpha.array()) * ((1 - alpha.array()).log() / log(2)));
	entropy = - temp.isNaN().select(0, temp).sum(); // remove NaN in the temp
}

void NegBinSSVIEM::print_model() {
	cout << "model: ";
	for (uint k = 0; k < p; k++) {
		if (alpha(k) > 0.5) {
			cout << k << ", ";
		}
	}
	cout << endl;
}

void NegBinSSVIEM::run_viem(bool verbose) {
	double logw0;    // keep track of the previous ELBO
	double entropy0; // keep track of the previous entropy
	double r_old, r_new; // here r = alpha(k) * mu(k)
	double SSR;		 // sum of squares due to regression
	double error;    // keep track of the error between alpha and alpha0
	VectorXd x;		 // one column of the design matrix X
	VectorXd alpha0; // keep track of the previous alpha

	// initialize parameter
	mu_r = 50;
	mu_log_r = log(mu_r);
	beta0 = 0;
	sa = 1;
	
	// init gsl sampler
	alpha = VectorXd::Zero(p);
	mu = VectorXd::Zero(p);
	r.norm(mu, 0, 0.01);
	//r.flat(alpha, 0, 1);
	for (uint k = 0; k < p; k++) {
		alpha(k) = gsl_ran_beta(r.r, a_pi, b_pi);
	}

	// init all statistics
	Xm = (X * (alpha.cwiseProduct(mu)).matrix());
	logodds = (log(a_pi) - log(b_pi - a_pi)) * VectorXd::Ones(p);
	// compute neg-bin statistics
	comp_negstats(X, y, mu_r, Xm + beta0 * VectorXd::Ones(n), d, s, sa, yhat, xy, xd, xdx, kappa);
	// compute ELBO
	comp_ELBO(d, yhat, xdx, kappa, Xm, logodds, alpha, mu, s, sa, logw);
	// compute entropy
	comp_entropy(alpha, entropy);
	cout << "initial ELBO: " << logw << " Entropy " << entropy << endl;
	
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
			alpha(k) = sigmoid(logodds(k) + (log(s(k) / sa) + SSR) / 2.0);
		
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
		b_r_tilde = b_r + sum_log_exp(eta);
		mu_r = a_r_tilde / (b_r_tilde + eps);
		mu_log_r = digamma(a_r_tilde) - log(b_r_tilde + eps);
		
		// update logodds
		logodds = (digamma(alpha.sum() + a_pi) - digamma(p - alpha.sum() + b_pi + eps)) * VectorXd::Ones(p);

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
		error_vec.push_back(error);
		
		if (abs(logw - logw0) < tol && mode == 0) { //  if ELBO converges then break
			cout << "condition 1" << endl;
			break;
		}
		else if (error < tol && mode == 1) { // if maximum relative difference between alpha and alpha0 less than tol then break
			cout << "condition 2" << endl;
			break;
		}
		else if (abs(entropy - entropy0) < tol && mode == 2) { // if relative difference between entropy and entropy0 less than tol then break
			cout << "condition 3" << endl;
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
	cout << "Time takes: " << time_spent << endl;
	print_model();
}


VectorXd NegBinSSVIEM::get_elbo_vec() {
	Map<VectorXd> ELBO(elbo_vec.data(), elbo_vec.size());
	return ELBO;
}

VectorXd NegBinSSVIEM::get_entropy_vec() {
	Map<VectorXd> ENTROPY(entropy_vec.data(), entropy_vec.size());
	return ENTROPY;
}

VectorXd NegBinSSVIEM::get_alpha_vec() {
	return alpha;
}

VectorXd NegBinSSVIEM::get_beta_vec() {
	return alpha.array() * mu.array();
}

VectorXd NegBinSSVIEM::get_omega_vec() {
	return d;
}

double NegBinSSVIEM::get_beta0() {
	return beta0;
}

double NegBinSSVIEM::get_sa() {
	return sa;
}

double NegBinSSVIEM::get_r() {
	return mu_r;
}

double NegBinSSVIEM::get_time_spent() {
	return time_spent;
}

#endif // !_NEGBINSSVIEM_