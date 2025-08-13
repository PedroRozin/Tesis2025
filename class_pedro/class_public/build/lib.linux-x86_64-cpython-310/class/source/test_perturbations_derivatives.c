#include <stdio.h>
#include <stdlib.h>
#include "background.h"
#include "thermodynamics.h"
#include "perturbations.h"
#include "common.h"

int main() {
    struct background pba;
    struct thermodynamics pth;
    struct perturbations ppt;
    struct precision pr;
    ErrorMsg errmsg;

    // Minimal initialization (set your own parameters as needed)
    background_init(&pr, &pba);
    thermodynamics_init(&pr, &pba, &pth);
    perturbations_init(&pr, &pba, &pth, &ppt);

    // Pick a mode, initial condition, k, and tau index to test
    int index_md = ppt.index_md_scalars;
    int index_ic = ppt.index_ic_ad;
    int index_k = 0; // first k
    int index_tau = 0; // first tau

    // Print the values
    printf("delta_cdm = %g\n", ppt.sources[index_md][index_ic * ppt.tp_size[index_md] + ppt.index_tp_delta_cdm][index_tau * ppt.k_size[index_md] + index_k]);
    printf("delta_b   = %g\n", ppt.sources[index_md][index_ic * ppt.tp_size[index_md] + ppt.index_tp_delta_b][index_tau * ppt.k_size[index_md] + index_k]);
    printf("delta_prime_cdm = %g\n", ppt.sources[index_md][index_ic * ppt.tp_size[index_md] + ppt.index_tp_delta_prime_cdm][index_tau * ppt.k_size[index_md] + index_k]);
    printf("delta_prime_b   = %g\n", ppt.sources[index_md][index_ic * ppt.tp_size[index_md] + ppt.index_tp_delta_prime_b][index_tau * ppt.k_size[index_md] + index_k]);

    // Clean up
    perturbations_free(&ppt);
    thermodynamics_free(&pth);
    background_free(&pba);

    return 0;
}