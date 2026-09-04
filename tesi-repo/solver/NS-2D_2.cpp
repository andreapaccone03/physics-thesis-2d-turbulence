#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
const double pi = acos(-1.0);

// Funzioni accessorie
int periodic_index(int i, int N) {
    return (i + N) % N;
}

double wave_number(int k, int N) {
    const int k_basso = (k <= N / 2) ? k : k - N;
    return 2.0 * pi * k_basso;
}

double partial_x(const vector<double>& u, int i, int j, int N, double dx) {
    return (u[periodic_index(i + 1, N) * N + j] - u[periodic_index(i - 1, N) * N + j]) / (2.0 * dx);
}

double partial_y(const vector<double>& u, int i, int j, int N, double dx) {
    return (u[i * N + periodic_index(j + 1, N)] - u[i * N + periodic_index(j - 1, N)]) / (2.0 * dx);
}

void fft_inversa(fftw_complex* data, fftw_plan plan_backward, int N) {
    fftw_execute(plan_backward);

    double norm = 1.0 / (N * N);
    for (int i = 0; i < N * N; ++i) {
        data[i][0] *= norm;
        data[i][1] *= norm;
    }
}

// Velocita da vorticita
void velocita(fftw_complex* w_hat, fftw_complex* psi, vector<double>& ux, vector<double>& uy, int N, double dx, fftw_plan plan_backward_psi) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double kx = wave_number(i, N);
            double ky = wave_number(j, N);
            double k2 = kx * kx + ky * ky;

            if (i == 0 && j == 0) {
                psi[0][0] = 0.0;
                psi[0][1] = 0.0;
            } else {
                psi[i * N + j][0] = -w_hat[i * N + j][0] / k2;
                psi[i * N + j][1] = -w_hat[i * N + j][1] / k2;
            }
        }
    }

    fft_inversa(psi, plan_backward_psi, N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ux[i * N + j] = (psi[i * N + periodic_index(j + 1, N)][0] - psi[i * N + periodic_index(j - 1, N)][0]) / (2.0 * dx);
            uy[i * N + j] = -(psi[periodic_index(i + 1, N) * N + j][0] - psi[periodic_index(i - 1, N) * N + j][0]) / (2.0 * dx);
        }
    }
}

void dealias(fftw_complex* hat, int N) {
    const int k_max = N / 3;
    for (int i = 0; i < N; ++i) {
        int kx = (i <= N / 2) ? i : i - N;
        for (int j = 0; j < N; ++j) {
            int ky = (j <= N / 2) ? j : j - N;
            if (abs(kx) > k_max || abs(ky) > k_max) {
                hat[i * N + j][0] = 0.0;
                hat[i * N + j][1] = 0.0;
            }
        }
    }
}

void write_spectrum_average(ofstream& out, const vector<double>& spectrum_accum, int n_samples, double time) {
    if (n_samples <= 0) {
        return;
    }

    out << "# Time: " << time << "\n";
    out << "# shell k value\n";
    for (int shell = 1; shell < static_cast<int>(spectrum_accum.size()); ++shell) {
        double value = spectrum_accum[shell] / n_samples;
        if (value <= 0.0) {
            continue;
        }
        out << shell << " " << (2.0 * pi * shell) << " " << value << "\n";
    }
    out << "\n";
}

void save_step(ofstream& f_w, ofstream& f_psi, ofstream& f_vel, double time, const vector<double>& x, const vector<double>& y, fftw_complex* w_hat, int N, double& total_energy, double& total_enstrophy, vector<double>& energy_spectrum_accum, vector<double>& enstrophy_spectrum_accum){

    int NN = N * N;
    const double prefactor = 0.5 / (static_cast<double>(NN) * NN);

    total_energy = 0.0;
    total_enstrophy = 0.0;

    fftw_complex* buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);

    // psi e accumulo spettri
    for (int i = 0; i < N; ++i) {
        int nx = (i <= N / 2) ? i : i - N;
        double kx = wave_number(i, N);

        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            int ny = (j <= N / 2) ? j : j - N;
            double ky = wave_number(j, N);
            double k2 = kx * kx + ky * ky;

            if (i == 0 && j == 0) {
                buf[idx][0] = 0.0;
                buf[idx][1] = 0.0;
            } else {
                buf[idx][0] = -w_hat[idx][0] / k2;
                buf[idx][1] = -w_hat[idx][1] / k2;

                double amp2 = w_hat[idx][0] * w_hat[idx][0] + w_hat[idx][1] * w_hat[idx][1];
                double e_contribution = prefactor * amp2 / k2;
                double z_contribution = prefactor * amp2;
                int shell = static_cast<int>(lround(sqrt(static_cast<double>(nx * nx + ny * ny))));

                total_energy += e_contribution;
                total_enstrophy += z_contribution;

                if (shell >= 0 && shell < static_cast<int>(energy_spectrum_accum.size())) {
                    energy_spectrum_accum[shell] += e_contribution;
                    enstrophy_spectrum_accum[shell] += z_contribution;
                }
            }
        }
    }

    fftw_plan p = fftw_plan_dft_2d(N, N, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    double norm = 1.0 / NN;
    for (int i = 0; i < NN; ++i) {
        buf[i][0] *= norm;
    }

    f_psi << "# Time: " << time << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            f_psi << x[i] << " " << y[j] << " " << buf[i * N + j][0] << "\n";
        }
    }
    f_psi << "\n";

    // ux, uy
    fftw_complex* ux_buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* uy_buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);

    for (int i = 0; i < N; ++i) {
        double kx = wave_number(i, N);
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double ky = wave_number(j, N);
            double k2 = kx * kx + ky * ky;

            if (i == 0 && j == 0) {
                ux_buf[idx][0] = ux_buf[idx][1] = 0.0;
                uy_buf[idx][0] = uy_buf[idx][1] = 0.0;
            } else {
                double psi_re = -w_hat[idx][0] / k2;
                double psi_im = -w_hat[idx][1] / k2;
                ux_buf[idx][0] = -ky * psi_im;
                ux_buf[idx][1] =  ky * psi_re;
                uy_buf[idx][0] =  kx * psi_im;
                uy_buf[idx][1] = -kx * psi_re;
            }
        }
    }

    p = fftw_plan_dft_2d(N, N, ux_buf, ux_buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    p = fftw_plan_dft_2d(N, N, uy_buf, uy_buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    for (int i = 0; i < NN; ++i) {
        ux_buf[i][0] *= norm;
        uy_buf[i][0] *= norm;
    }

    f_vel << "# Time: " << time << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            f_vel << x[i] << " " << y[j] << " "
                  << ux_buf[i * N + j][0] << " " << uy_buf[i * N + j][0] << "\n";
        }
    }
    f_vel << "\n";

    //vorticita in spazio reale
    memcpy(buf, w_hat, sizeof(fftw_complex) * NN);
    p = fftw_plan_dft_2d(N, N, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    for (int i = 0; i < NN; ++i) {
        buf[i][0] *= norm;
    }

    f_w << "# Time: " << time << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            f_w << x[i] << " " << y[j] << " " << buf[i * N + j][0] << "\n";
        }
    }
    f_w << "\n";

    fftw_free(buf);
    fftw_free(ux_buf);
    fftw_free(uy_buf);
}

void save_final(const vector<double>& x, const vector<double>& y, fftw_complex* w, fftw_complex* psi, vector<double>& ux, vector<double>& uy, int N, double dx, fftw_plan plan_backward_w, fftw_plan plan_backward_psi){
    velocita(w, psi, ux, uy, N, dx, plan_backward_psi);

    ofstream out_vel("output/final_velocity.dat");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out_vel << x[i] << " " << y[j] << " " << ux[i * N + j] << " " << uy[i * N + j] << "\n";
        }
    }

    ofstream out_psi("output/final_psi.dat");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out_psi << x[i] << " " << y[j] << " " << psi[i * N + j][0] << "\n";
        }
    }

    fft_inversa(w, plan_backward_w, N);

    ofstream out_w("output/final_vorticity.dat");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out_w << x[i] << " " << y[j] << " " << w[i * N + j][0] << "\n";
        }
    }
}

void forzante(fftw_complex* f_hat, int N, double k_f, double A) {
    // Generatori casuali statici 
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<> dist(0.0, 1.0);

    // Azzera la forzante del passo precedente
    std::memset(f_hat, 0, sizeof(fftw_complex) * N * N);

    // Spessore del guscio spettrale 
    double pi = std::acos(-1.0);
    double dk = 2.0 * pi * 0.5; 

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int i_sym = (N - i) % N;
            int j_sym = (N - j) % N;
            int idx = i * N + j;
            int idx_sym = i_sym * N + j_sym;

            if (idx <= idx_sym) { 
                double kx = wave_number(i, N);
                double ky = wave_number(j, N);
                
                double k_modulo = std::sqrt(kx * kx + ky * ky);
            
                if (std::fabs(k_modulo - k_f) > dk) {
                    continue;
                }

                // Generazione dell'impulso stocastico gaussiano
                double gR = dist(gen);
                double gI = dist(gen);

                // Applicazione del prefattore di scala FFTW (N * N)
                f_hat[idx][0] = A * N * N * gR;
                f_hat[idx][1] = A * N * N * gI;

                // Imposizione dell'ermiticità 
                if (idx != idx_sym) {
                    f_hat[idx_sym][0] =  f_hat[idx][0]; 
                    f_hat[idx_sym][1] = -f_hat[idx][1]; 
                } else {
                    // Per i modi auto-simmetrici, la parte immaginaria deve essere nulla
                    f_hat[idx][1] = 0.0;
                }
            }
        }
    }
    
    // filtro di dealiasing 
    dealias(f_hat, N);      
    
    // Forza la media spaziale
    f_hat[0][0] = 0.0;
    f_hat[0][1] = 0.0; 
}

void initialize(vector<double>& x, vector<double>& y, vector<double>& ux, vector<double>& uy, fftw_complex*& w, fftw_complex*& psi, int N, double dx) {
    for (int i = 0; i < N; ++i) {
        x[i] = i * dx;
        y[i] = (i - N / 2) * dx;
    }

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dis(0, 1);
   
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            w[i * N + j][0] = 1e-4 * dis(gen);
            w[i * N + j][1] = 0.0;
        }
    }
    
    /*
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            w[i * N + j][0] = 0.0;
            w[i * N + j][1] = 0.0;
        }
    }*/
        

    fftw_plan plan_forward = fftw_plan_dft_2d(N, N, w, w, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_backward = fftw_plan_dft_2d(N, N, psi, psi, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double kx = wave_number(i, N);
            double ky = wave_number(j, N);
            double k2 = kx * kx + ky * ky;
            if (i == 0 && j == 0) {
                psi[0][0] = 0.0;
                psi[0][1] = 0.0;
            } else {
                psi[i * N + j][0] = -w[i * N + j][0] / k2;
                psi[i * N + j][1] = -w[i * N + j][1] / k2;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ux[i * N + j] = 0.0;
            uy[i * N + j] = 0.0;
        }
    }

    velocita(w, psi, ux, uy, N, dx, plan_backward);

    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
}

void rhs(fftw_complex* w_hat, fftw_complex* f, fftw_complex* out, int N, double nu, double alpha) {
    int NN = N * N;

    dealias(w_hat, N);

    fftw_complex* psi_hat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* ux_hat  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* uy_hat  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* wx_hat  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* wy_hat  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);

    for (int i = 0; i < N; ++i) {
        double kx = wave_number(i, N);
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double ky = wave_number(j, N);
            double k2 = kx * kx + ky * ky;

            if (i == 0 && j == 0) {
                psi_hat[idx][0] = psi_hat[idx][1] = 0.0;
                ux_hat[idx][0]  = ux_hat[idx][1]  = 0.0;
                uy_hat[idx][0]  = uy_hat[idx][1]  = 0.0;
            } else {
                psi_hat[idx][0] = -w_hat[idx][0] / k2;
                psi_hat[idx][1] = -w_hat[idx][1] / k2;

                ux_hat[idx][0] = -ky * psi_hat[idx][1];
                ux_hat[idx][1] =  ky * psi_hat[idx][0];

                uy_hat[idx][0] =  kx * psi_hat[idx][1];
                uy_hat[idx][1] = -kx * psi_hat[idx][0];
            }

            wx_hat[idx][0] = -kx * w_hat[idx][1];
            wx_hat[idx][1] =  kx * w_hat[idx][0];
            wy_hat[idx][0] = -ky * w_hat[idx][1];
            wy_hat[idx][1] =  ky * w_hat[idx][0];
        }
    }

    auto ifft = [&](fftw_complex* buf) {
        fftw_plan p = fftw_plan_dft_2d(N, N, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
        double norm = 1.0 / (N * N);
        for (int i = 0; i < NN; ++i) {
            buf[i][0] *= norm;
            buf[i][1] *= norm;
        }
    };

    ifft(ux_hat);
    ifft(uy_hat);
    ifft(wx_hat);
    ifft(wy_hat);

    fftw_complex* nl = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    for (int idx = 0; idx < NN; ++idx) {
        nl[idx][0] = ux_hat[idx][0] * wx_hat[idx][0] + uy_hat[idx][0] * wy_hat[idx][0];
        nl[idx][1] = 0.0;
    }

    fftw_plan p_nl = fftw_plan_dft_2d(N, N, nl, nl, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p_nl);
    fftw_destroy_plan(p_nl);
    dealias(nl, N);

    for (int i = 0; i < N; ++i) {
        double kx = wave_number(i, N);
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double ky = wave_number(j, N);
            double k2 = kx * kx + ky * ky;
            double diss = nu * k2 + alpha;
            out[idx][0] = -nl[idx][0] - diss * w_hat[idx][0] + f[idx][0];
            out[idx][1] = -nl[idx][1] - diss * w_hat[idx][1] + f[idx][1];
        }
    }

    fftw_free(psi_hat);
    fftw_free(ux_hat);
    fftw_free(uy_hat);
    fftw_free(wx_hat);
    fftw_free(wy_hat);
    fftw_free(nl);
}

void RK4_step(fftw_complex* w, fftw_complex* f, int N, double dt, double nu, double alpha) {
    int NN = N * N;
    fftw_complex* w_tmp = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* k1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* k2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* k3 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);
    fftw_complex* k4 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * NN);

    rhs(w, f, k1, N, nu, alpha);

    for (int idx = 0; idx < NN; ++idx) {
        w_tmp[idx][0] = w[idx][0] + 0.5 * dt * k1[idx][0];
        w_tmp[idx][1] = w[idx][1] + 0.5 * dt * k1[idx][1];
    }
    rhs(w_tmp, f, k2, N, nu, alpha);

    for (int idx = 0; idx < NN; ++idx) {
        w_tmp[idx][0] = w[idx][0] + 0.5 * dt * k2[idx][0];
        w_tmp[idx][1] = w[idx][1] + 0.5 * dt * k2[idx][1];
    }
    rhs(w_tmp, f, k3, N, nu, alpha);

    for (int idx = 0; idx < NN; ++idx) {
        w_tmp[idx][0] = w[idx][0] + dt * k3[idx][0];
        w_tmp[idx][1] = w[idx][1] + dt * k3[idx][1];
    }
    rhs(w_tmp, f, k4, N, nu, alpha);

    for (int idx = 0; idx < NN; ++idx) {
        w[idx][0] += (dt / 6.0) * (k1[idx][0] + 2.0 * k2[idx][0] + 2.0 * k3[idx][0] + k4[idx][0]);
        w[idx][1] += (dt / 6.0) * (k1[idx][1] + 2.0 * k2[idx][1] + 2.0 * k3[idx][1] + k4[idx][1]);
    }

    dealias(w, N);

    fftw_free(w_tmp);
    fftw_free(k1);
    fftw_free(k2);
    fftw_free(k3);
    fftw_free(k4);
}

int main() {
    const double T = 25.0;
    const int N = 256;
    const double dx = 1.0 / N;
    const double dt = 0.0002;
    const double nu = 5e-5;
    const double alpha = 0.15;
    const int n_steps = static_cast<int>(T / dt);

    vector<double> x(N);
    vector<double> y(N);
    vector<double> ux(N * N);
    vector<double> uy(N * N);

    fftw_complex* w = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_complex* psi = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * N);
    initialize(x, y, ux, uy, w, psi, N, dx);

    // Campo forzante
    fftw_complex* f_hat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * N);
    double k_f = 2.0 * pi * 10.0;
    double A = 1.0/sqrt(dt);
    memset(f_hat, 0, sizeof(fftw_complex) * N * N);

    const int save_stride = 100;
    const double total_window_dt = 0.5;
    const double spectrum_window_dt = 0.25;
    const double sample_dt = save_stride * dt;
    const int n_shells = static_cast<int>(ceil(sqrt(2.0) * (N / 3.0))) + 1;

    fftw_plan plan_backward_w = fftw_plan_dft_2d(N, N, w, w, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan plan_backward_psi = fftw_plan_dft_2d(N, N, psi, psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    ofstream evolution_w("output/vorticity_evolution.dat");
    ofstream evolution_psi("output/psi_evolution.dat");
    ofstream evolution_vel("output/velocity_evolution.dat");

    ofstream energy_total_out("output/energy_total.dat");
    ofstream energy_spectrum_out("output/energy_spectrum.dat");
    ofstream enstrophy_total_out("output/enstrophy_total.dat");
    ofstream enstrophy_spectrum_out("output/enstrophy_spectrum.dat");

    energy_total_out << "# sampled_every " << sample_dt << "\n";
    energy_total_out << "# averaged_over " << total_window_dt << "\n";
    energy_total_out << "# time E_tot\n";

    enstrophy_total_out << "# sampled_every " << sample_dt << "\n";
    enstrophy_total_out << "# averaged_over " << total_window_dt << "\n";
    enstrophy_total_out << "# time Z_tot\n";

    energy_spectrum_out << "# sampled_every " << sample_dt << "\n";
    energy_spectrum_out << "# averaged_over " << spectrum_window_dt << "\n";

    enstrophy_spectrum_out << "# sampled_every " << sample_dt << "\n";
    enstrophy_spectrum_out << "# averaged_over " << spectrum_window_dt << "\n";

    double energy_total_accum = 0.0;
    double enstrophy_total_accum = 0.0;
    int total_samples = 0;
    double next_total_time = total_window_dt;

    double energy_sample = 0.0;
    double enstrophy_sample = 0.0;
    vector<double> energy_spectrum_accum(n_shells, 0.0);
    vector<double> enstrophy_spectrum_accum(n_shells, 0.0);
    int spectrum_samples = 0;
    double next_spectrum_time = spectrum_window_dt;

    for (int step = 0; step < n_steps; ++step) {
        
        forzante(f_hat, N, k_f, A);
        
        RK4_step(w, f_hat, N, dt, nu, alpha);

        if (step % save_stride == 0) {
            double time = step * dt;

            save_step(evolution_w, evolution_psi, evolution_vel, time, x, y, w, N, energy_sample, enstrophy_sample, energy_spectrum_accum, enstrophy_spectrum_accum);

            energy_total_accum += energy_sample;
            enstrophy_total_accum += enstrophy_sample;
            ++total_samples;
            ++spectrum_samples;

            while (time + 0.5 * sample_dt >= next_total_time) {
                energy_total_out << next_total_time << " " << (energy_total_accum / total_samples) << "\n";
                enstrophy_total_out << next_total_time << " " << (enstrophy_total_accum / total_samples) << "\n";

                energy_total_accum = 0.0;
                enstrophy_total_accum = 0.0;
                total_samples = 0;
                next_total_time += total_window_dt;
            }

            while (time + 0.5 * sample_dt >= next_spectrum_time) {
                write_spectrum_average(energy_spectrum_out, energy_spectrum_accum, spectrum_samples, next_spectrum_time);
                write_spectrum_average(enstrophy_spectrum_out, enstrophy_spectrum_accum, spectrum_samples, next_spectrum_time);

                fill(energy_spectrum_accum.begin(), energy_spectrum_accum.end(), 0.0);
                fill(enstrophy_spectrum_accum.begin(), enstrophy_spectrum_accum.end(), 0.0);
                spectrum_samples = 0;
                next_spectrum_time += spectrum_window_dt;
            }
        }
    }

    if (total_samples > 0) {
        energy_total_out << T << " " << (energy_total_accum / total_samples) << "\n";
        enstrophy_total_out << T << " " << (enstrophy_total_accum / total_samples) << "\n";
    }

    if (spectrum_samples > 0) {
        write_spectrum_average(energy_spectrum_out, energy_spectrum_accum, spectrum_samples, T);
        write_spectrum_average(enstrophy_spectrum_out, enstrophy_spectrum_accum, spectrum_samples, T);
    }

    evolution_w.close();
    evolution_psi.close();
    evolution_vel.close();
    energy_total_out.close();
    energy_spectrum_out.close();
    enstrophy_total_out.close();
    enstrophy_spectrum_out.close();

    save_final(x, y, w, psi, ux, uy, N, dx, plan_backward_w, plan_backward_psi);

    fftw_destroy_plan(plan_backward_w);
    fftw_destroy_plan(plan_backward_psi);
    fftw_free(w);
    fftw_free(psi);
    fftw_free(f_hat);

    return 0;
}
