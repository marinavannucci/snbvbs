#ifndef _NEGBINHS_
#define _NEGBINHS_
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include "RNG.hpp"
#include "PolyaGammaHybrid.h"
#include <gsl/gsl_sf_psi.h>

using namespace std;
using namespace Eigen;
typedef unsigned int uint;

// define Negative Binomial Horseshore Sampler class
class NegBinHS {
private:
	// seed 
	uint seed;
	RNG r;
	PolyaGammaHybridDouble pg;

	// Prior
	double a_r;
	double b_r;

	// Data
	MatrixXd X;   // design matrix (n x p)
	VectorXd y;   // outcome y in counts (n x 1)
	uint p, n;    // sample and feature size

	// parameter (inits with c for coefficient) 
	double cr;        // over-dispersion paramter
	double cbeta0;    // linear bias
	double ctau;      // global shrinkage coef
	VectorXd cbetas;  // beta coefficients (p x 1)
	VectorXd cgammas; // local shrinkage coef (p x 1)
	VectorXd cetas;   // local shrinkage augmented variable (p x 1)
	VectorXd czs;     // linear term X * beta + beta0 (n x 1)
	VectorXd cws;     // augmented variable omega (n x 1)

	// some statistics
	VectorXd cks, hcks; // (n x 1)
	MatrixXd hcW;     // (n x n)
	VectorXd l_beta;  // (p x 1)
	MatrixXd Q_beta;  // (p x p)

	// define variable storing the samples
	MatrixXd beta_mat;
	MatrixXd eta_mat;   // local shrinkage 
	MatrixXd omega_mat;
	VectorXd beta0_vec;
	VectorXd r_vec;
	VectorXd tau_vec;

	// control mcmc sampling
	uint n_sample;		// n mcmc samples 
	uint n_burnin;		// n burnin samples
	uint n_truncate;	// truncate level for Poisson approximation
	clock_t begin, end;
	double time_spent;	// track time spent for sampling in seconds

	double draw_crt_l_sum(VectorXd &pys, double pr);
	double sum_log_exp(VectorXd &pczs);
public:
	NegBinHS(uint pseed);
	void set_data(const MatrixXd& X_data, const VectorXd& y_data);
	void set_prior(double pa_r, double pb_r);
	void set_mcmc(uint pn_sample = 20000, uint pn_burnin = 3000, uint pn_truncate = 1000);
	void run_mcmc(bool verbose); // print sampling info or not
	double draw_crt_l(double py, double pr);
	
	// export the samples
	MatrixXd get_beta_mat();
	MatrixXd get_eta_mat();
	MatrixXd get_omega_mat();
	VectorXd get_beta0_vec();
	VectorXd get_r_vec();
	VectorXd get_tau_vec();
	double get_time();
};

NegBinHS::NegBinHS(uint pseed) {
	seed = pseed;
	r.set(seed);
}


// set horseshoe priors
void NegBinHS::set_prior(double pa_r, double pb_r) {
	if (pa_r > 0 && pb_r > 0) {
		a_r = pa_r;
		b_r = pb_r;
		cout << "--------------------------------------------------" << endl;
		printf("Prior on r: a_r = %6.3f\tb_r = %6.3f\n", a_r, b_r);
		cout << "--------------------------------------------------" << endl;
	}
	else {
		throw invalid_argument("All prior parameters should be larger than 0.");
	}
}

// set the regression data X and y
void NegBinHS::set_data(const MatrixXd& X_data, const VectorXd& y_data) {
	// check if data is valid or not
	p = (uint) X_data.cols();
	n = (uint) X_data.rows();
	if (n != ((uint) y_data.rows())) {
		throw invalid_argument("Number of rows in y should be equal to the number of rows in X");
	}

	X = X_data;
	y = y_data;
	cout << "--------------------------------------------------" << endl;
	printf("Features: %d\tSamples: %d\n", p, n);
	cout << "--------------------------------------------------" << endl;
}

void NegBinHS::set_mcmc(uint pn_sample, uint pn_burnin, uint pn_truncate) {
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

// draw the number of tables from CRP given one count y and overdispersion r
double NegBinHS::draw_crt_l(double py, double pr) {
	double l = 0;
	if (pr <= 0) {
		throw invalid_argument("the overdispersion parameter r could not be smaller than 0.");
		return (EXIT_FAILURE);
	}
	if (py == 0) {
		l = 0;
	}
	else if (py > n_truncate) {
		// if py much larger then approximate l from drawing a poisson
		l = pr * (gsl_sf_psi(py + pr) - gsl_sf_psi(pr));
		l = gsl_ran_poisson(r.r, l);
	}
	else {
		// draw l directly from a Chinese Restaurant Process
		for (uint i = 0; i < (uint) py; i++) {
			l += (gsl_rng_uniform(r.r) <= (pr / (pr + i)));
		}
	}
	return l;
}


// draw the sum of number of tables from CRP given count vector y and overdispersion r
double NegBinHS::draw_crt_l_sum(VectorXd &pys, double pr) {
	double  l_sum = 0;
	for (uint i = 0; i < n; i++) {
		l_sum += draw_crt_l(pys(i), pr);
	}
	return l_sum;
}

// compute the sum of the log(1 + exp(z)) given a vector of linear term z
double NegBinHS::sum_log_exp(VectorXd &pczs) {
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

void NegBinHS::run_mcmc(bool verbose) {
	// set the seed
	r.set(seed);
	int s;  // sample index s

	// clear parameter
	cbetas = VectorXd::Zero(p);
	cetas = VectorXd::Zero(p);
	cgammas = VectorXd::Zero(p);
	cws = VectorXd::Zero(n);
	cks = VectorXd::Zero(n);
	hcks = VectorXd::Zero(n);
	hcW = MatrixXd::Zero(n, n);  // normalization kernel

	// create space for samples
	beta_mat = MatrixXd::Zero(n_sample, p);
	eta_mat = MatrixXd::Zero(n_sample, p);
	omega_mat = MatrixXd::Zero(n_sample, n);
	r_vec = VectorXd::Zero(n_sample);
	beta0_vec = VectorXd::Zero(n_sample);
	tau_vec = VectorXd::Zero(n_sample);


	// initialize parameters 
	cr = 500;
	cbeta0 = 0;
	ctau = 1;
	
	// initialize eta, gamma and beta from random distribution
	r.gamma_rate(cetas, 1, 1);
	r.gamma_rate(cgammas, 1, 1);
	r.norm(cbetas, 1);

	// start mcmc
	begin = clock();

	for (uint iter = 0; iter < n_sample + n_burnin; iter++) {
		// compute the linear term vector (n x 1)
		czs = (X * cbetas).array() + cbeta0 + log(cr);

		// update omega vector (n by 1)
		for (uint i = 0; i < n; i++) {
			cws(i) = pg.draw(y(i) + cr, czs(i) - log(cr), r);
		}

		// update overdispersion scalar r
		cr = gsl_ran_gamma(r.r, a_r + draw_crt_l_sum(y, cr), 1 / (b_r + sum_log_exp(czs)));

		// update some statistics
		hcW = MatrixXd(cws.asDiagonal()) - (cws * cws.transpose()) / cws.sum();
		cks = (y.array() - cr) / 2;
		hcks = cks - (cks.sum() / cws.sum()) * cws;  // normalized version of cks


		// update beta vector (p x 1)
		Q_beta = (X.transpose() * hcW * X) + MatrixXd(cetas.asDiagonal());
		l_beta = X.transpose() * hcks;


		// use the forward and backward solve to sample another beta vector (p x 1)
		LLT<MatrixXd> chol_Q(Q_beta);  // lower triangular Cholesky decomposition of Q_beta
		r.norm(cbetas, 1); // draw a vector of standard normal (p x 1)

		// forward & backward solve for betas
		cbetas = chol_Q.matrixU().solve(chol_Q.matrixL().solve(l_beta) + cbetas);

		// update beta0 scalar
		// beta0 = (kappa.sum() - d.transpose() * Xm) / d.sum();
		cbeta0 = (cks.sum() - cws.transpose() * (X * cbetas)) / (cws).sum();

		// update etas
		for (uint i = 0; i < p; i++) {
			cetas(i) = gsl_ran_gamma(r.r, 1, 1 / (cgammas(i) + pow(cbetas(i), 2) / 2.0));
		}

		// update gammas
		for (uint i = 0; i < p; i++) {
			cgammas(i) = gsl_ran_gamma(r.r, 1, 1 / (cetas(i) + ctau));
		}

		// update tau
		ctau = r.ltgamma(p / 2.0, cgammas.sum(), 0.01);

		// print useful diagnostic information every 1000 iterations
		if (iter % 1000 == 0 && iter > 0 && verbose) {
			end = clock();
			printf("seed: %04d, iter: %05d, r: %6.3f, beta0: %6.3f, tau: %9.3f | ETA: %6.3f\n",
				seed, iter, cr, cbeta0, ctau, double(end - begin) / (CLOCKS_PER_SEC) / (iter) * (n_sample + n_burnin - iter));
		}

		// save samples after the burinin period
		if (iter > n_burnin - 1) {					  
			s = (int)(iter - n_burnin);
			beta_mat.row(s) = cbetas;			 // save betas
			eta_mat.row(s) = cetas;				 // save local shrinkage etas
			omega_mat.row(s) = cws;				 // save augmented variable omega
			r_vec(s) = cr;						 // save over-dispersion parameter r
			beta0_vec(s) = cbeta0;				 // save baseline or intercept term beta0
			tau_vec(s) = ctau;                 	 // save global shrinkage tau
		}
	}
	// print program information
	end = clock();
	time_spent = double(end - begin) / (CLOCKS_PER_SEC);
	cout << "Time takes: " << time_spent << endl;
}

/* export the samples to python*/ 
MatrixXd NegBinHS::get_beta_mat() {
	return beta_mat;
}

MatrixXd NegBinHS::get_eta_mat() {
	return eta_mat;
}

MatrixXd NegBinHS::get_omega_mat() {
	return omega_mat;
}

VectorXd NegBinHS::get_beta0_vec() {
	return beta0_vec;
}

VectorXd NegBinHS::get_r_vec() {
	return r_vec;
}

VectorXd NegBinHS::get_tau_vec() {
	return tau_vec;
}

double NegBinHS::get_time() {
	return time_spent;
}

#endif // !_NEGBINHS_