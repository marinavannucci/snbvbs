#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "RNG.hpp"
#include "PolyaGammaHybrid.h"
#include "utility.h"
#include "NegBinHS.h"
#include "NegBinSSMCMC.h"
#include "NegBinSSVIEM.h"
#include "NegBinSSVIIS.h"

using namespace Eigen;
using namespace std;
namespace py = pybind11;

template<typename M1>
void fun2(MatrixBase<M1> &a) {
	a.array() += 10;
}

// test GRNG
void test() {
	RNG r;
	r.set(2018);
	MatrixXd A(5, 5);
	r.norm(A, 1.0);

	cout << "A:\n" << A << endl;

	// add one to A
	fun2(A);
	cout << "A:\n" << A << endl;

	// draw samples from PG
	
	PolyaGammaHybridDouble pg;
	cout << "Those are samples from pg:" << endl;
	cout << pg.draw(1, 2, r) << endl;
	cout << pg.draw(1, 2, r) << endl;
	cout << pg.draw(1, 2, r) << endl;
	cout << pg.draw(100, 2, r) << endl;
	cout << pg.draw(100, 2, r) << endl;
	cout << pg.draw(100, 2, r) << endl;
	cout << pg.draw(100, 2, r) << endl;
	cout << pg.draw(100, 4, r) << endl;


}


PYBIND11_MODULE(csnbvbs, m) {
	m.doc() = "pybind11 example plugin";
	m.def("test", &test, "function test the polyagamma sampler");
	py::class_<NegBinHS>(m, "NegBinHS")
		.def(py::init<uint>())
		.def("set_data", &NegBinHS::set_data, "set data for Negative Binomial Horse Shoe Regression.")
		.def("set_prior", &NegBinHS::set_prior, "set prior for Negative Binomial Horse Shoe Regression.")
		.def("set_mcmc", &NegBinHS::set_mcmc, "set the number of samples, burnin and truncation level in MCMC.")
		.def("draw_crt_l", &NegBinHS::draw_crt_l, "draw from number of tables from a Chinese Restaurant Process.")
		.def("run_mcmc", &NegBinHS::run_mcmc, "run Gibbs sampling for Negative Binomial Horse Shoe Regression.")
		.def("get_beta_mat", &NegBinHS::get_beta_mat, "get the samples of the coefficient betas.")
		.def("get_eta_mat", &NegBinHS::get_eta_mat, "get the samples of the local shrinkage etas.")
		.def("get_omega_mat", &NegBinHS::get_omega_mat, "get the samples of the augmented variable omegas.")
		.def("get_beta0_vec", &NegBinHS::get_beta0_vec, "get the samples of the baseline or intercept beta0.")
		.def("get_r_vec", &NegBinHS::get_r_vec, "get the samples of the over-dispersion r.")
		.def("get_tau_vec", &NegBinHS::get_tau_vec, "get the samples of the local shrinkage tau.")
		.def("get_time", &NegBinHS::get_time, "get the time spent on the mcmc sampling.");
	
	py::class_<NegBinSSMCMC>(m, "NegBinSSMCMC")
		.def(py::init<uint>())
		.def("set_data", &NegBinSSMCMC::set_data, "set data for Negative Binomial Spike & Slab Regression.")
		.def("set_prior", &NegBinSSMCMC::set_prior, "set prior for Negative Binomial Spike & Slab Regression.")
		.def("set_mcmc", &NegBinSSMCMC::set_mcmc, "set the number of samples, burnin and truncation level in MCMC.")
		.def("draw_crt_l", &NegBinSSMCMC::draw_crt_l, "draw from number of tables from a Chinese Restaurant Process.")
		.def("get_Q_chol", &NegBinSSMCMC::get_Q_chol, "get the Cholesky decomposition of Q.")
		.def("get_hcks", &NegBinSSMCMC::get_hcks, "get the normalized y.")
		.def("run_mcmc", &NegBinSSMCMC::run_mcmc, "run Gibbs sampling for Negative Binomial Horse Shoe Regression.")
		.def("get_beta_mat", &NegBinSSMCMC::get_beta_mat, "get the samples of the coefficient betas.")
		.def("get_gamma_mat", &NegBinSSMCMC::get_gamma_mat, "get the samples of the model indicator gamma.")
		.def("get_omega_mat", &NegBinSSMCMC::get_omega_mat, "get the samples of the augmented variable omegas.")
		.def("get_beta0_vec", &NegBinSSMCMC::get_beta0_vec, "get the samples of the baseline or intercept beta0.")
		.def("get_r_vec", &NegBinSSMCMC::get_r_vec, "get the samples of the over-dispersion r.")
		.def("get_tau_vec", &NegBinSSMCMC::get_tau_vec, "get the samples of the local shrinkage tau.")
		.def("get_time", &NegBinSSMCMC::get_time, "get the time spent on the mcmc sampling.");

	py::class_<NegBinSSVIEM>(m, "NegBinSSVIEM")
		.def(py::init<uint>())
		.def("set_data", &NegBinSSVIEM::set_data, "set data for Negative Binomial Spike & Slab Regression.")
		.def("set_prior", &NegBinSSVIEM::set_prior, "set prior for Negative Binomial Spike & Slab Regression.")
		.def("set_viem", &NegBinSSVIEM::set_viem, "set the number of iteration and tolerance value in VIEM.")
		.def("run_viem", &NegBinSSVIEM::run_viem, "run VIEM for Negative Binomial Spike and Slab Regression.")
		.def("get_elbo_vec", &NegBinSSVIEM::get_elbo_vec, "get the ELBO convergence vector from VIEM.")
		.def("get_entropy_vec", &NegBinSSVIEM::get_entropy_vec, "get the entropy convergence vector from VIEM.")
		.def("get_alpha_vec", &NegBinSSVIEM::get_alpha_vec, "get the PIP from VIEM.")
		.def("get_beta_vec", &NegBinSSVIEM::get_beta_vec, "get the estimated coefficient beta from VIEM.")
		.def("get_omega_vec", &NegBinSSVIEM::get_omega_vec, "get the estimated augmented variable omegas from VIEM.")
		.def("get_beta0", &NegBinSSVIEM::get_beta0, "get the estimated baseline or intercept beta0.")
		.def("get_r", &NegBinSSVIEM::get_r, "get the estimated over-dispersion r.")
		.def("get_sa", &NegBinSSVIEM::get_sa, "get the estimated slab variance sa.")
		.def("get_time", &NegBinSSVIEM::get_time_spent, "get the time spent on the VIEM optimization.");

	py::class_<NegBinSSVIIS>(m, "NegBinSSVIIS")
		.def(py::init<uint>())
		.def("set_data", &NegBinSSVIIS::set_data, "set data for Negative Binomial Spike & Slab Regression.")
		.def("set_prior", &NegBinSSVIIS::set_prior, "set prior for Negative Binomial Spike & Slab Regression.")
		.def("set_viem", &NegBinSSVIIS::set_viem, "set the number of iteration and tolerance value in VIEM.")
		.def("init_viem", &NegBinSSVIIS::init_viem, "set the number of iteration and tolerance value in VIEM.")
		.def("run_viem", &NegBinSSVIIS::run_viem, "run VIEM for Negative Binomial Spike and Slab Regression.")
		.def("get_elbo_vec", &NegBinSSVIIS::get_elbo_vec, "get the ELBO convergence vector from VIEM.")
		.def("get_entropy_vec", &NegBinSSVIIS::get_entropy_vec, "get the entropy convergence vector from VIEM.")
		.def("get_alpha_vec", &NegBinSSVIIS::get_alpha_vec, "get the PIP from VIEM.")
		.def("get_beta_vec", &NegBinSSVIIS::get_mu_vec, "get the estimated coefficient beta from VIEM.")
		.def("get_omega_vec", &NegBinSSVIIS::get_omega_vec, "get the estimated augmented variable omegas from VIEM.")
		.def("get_beta0", &NegBinSSVIIS::get_beta0, "get the estimated baseline or intercept beta0.")
		.def("get_r", &NegBinSSVIIS::get_r, "get the estimated over-dispersion r.")
		.def("get_time", &NegBinSSVIIS::get_time_spent, "get the time spent on the VIEM optimization.");

	py::class_<parNegBinSSVIIS>(m, "parNegBinSSVIIS")
		.def(py::init<uint>())
		.def("set_data", &parNegBinSSVIIS::set_data, "set data for Negative Binomial Spike & Slab Regression.")
		.def("set_prior", &parNegBinSSVIIS::set_prior, "set prior for Negative Binomial Spike & Slab Regression.")
		.def("set_viem", &parNegBinSSVIIS::set_viem, "set the number of iteration and tolerance value in VIEM.")
		.def("get_pip_vec", &parNegBinSSVIIS::get_pip, "get the PIP from VIEM.")
		.def("get_beta_vec", &parNegBinSSVIIS::get_beta, "get the estimated coefficient beta from VIEM.")
		.def("get_omega_vec", &parNegBinSSVIIS::get_omega_vec, "get the estimated augmented variable omegas from VIEM.")
		.def("get_beta0", &parNegBinSSVIIS::get_beta0, "get the estimated baseline or intercept beta0.")
		.def("get_r", &parNegBinSSVIIS::get_r, "get the estimated over-dispersion r.")
		.def("get_sa", &parNegBinSSVIIS::get_sa, "get the estimated slab variance sa.")
		.def("run_viem", &parNegBinSSVIIS::run_viem, "run multiple VIEM for Negative Binomial Spike and Slab Regression with different prior logodds.")
		.def("get_time", &parNegBinSSVIIS::get_time_spent, "get the time spent on the VIEM optimization.");
}

