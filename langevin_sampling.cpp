#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

static std::mt19937 rng(12345);
static std::normal_distribution<double> standard_normal(0.0, 1.0);

struct Gaussian {
    double mu;
    double sigma;

    double density(double x) const {
        static constexpr double inv_sqrt_2pi = 1.0 / std::sqrt(2.0 * M_PI);
        double z = (x - mu) / sigma;
        return inv_sqrt_2pi / sigma * std::exp(-0.5 * z * z);
    }

    double gradient(double x) const {
        return -(x - mu) / (sigma * sigma) * density(x);
    }

    double score(double x) const {
        return -(x - mu) / (sigma * sigma);
    }
};

struct GaussianMixture {
    Gaussian g1;
    Gaussian g2;
    double w;

    double density(double x) const {
        return w * g1.density(x) + (1.0 - w) * g2.density(x);
    }

    double gradient(double x) const {
        return w * g1.gradient(x) + (1.0 - w) * g2.gradient(x);
    }

    double score(double x) const {
        double p1 = w * g1.density(x);
        double p2 = (1.0 - w) * g2.density(x);
        double numerator = p1 * g1.score(x) + p2 * g2.score(x);
        double denominator = p1 + p2;
        return numerator / denominator;
    }
};

double sample_from_normal() {
    return standard_normal(rng);
}

double langevin_step(double current_x, double step_size, const GaussianMixture& gm) {
    return current_x + 0.5 * step_size * gm.score(current_x)
                     + std::sqrt(step_size) * sample_from_normal();
}

std::vector<double> run_langevin_sampling(
    int n_steps, double initial_x, double step_size, const GaussianMixture& gm
) {
    std::vector<double> samples(n_steps);
    samples[0] = initial_x;
    double current_x = initial_x;
    for (int i = 1; i < n_steps; i++) {
        current_x = langevin_step(current_x, step_size, gm);
        samples[i] = current_x;
    }
    return samples;
}

int main() {
    GaussianMixture gm{Gaussian{0.0, 1.0}, Gaussian{3.0, 1.0}, 0.5};
    auto samples = run_langevin_sampling(100000, 0.5, 0.01, gm);

    std::ofstream out("outputs/samples.csv");
    for (double s : samples) {
        out << s << "\n";
    }
    out.close();
    return 0;
}