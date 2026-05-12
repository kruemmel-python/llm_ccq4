// Interface for the CipherCore noise controller. Provides functions to adjust
// and inspect the global noise scaling factor used to keep signal variance
// within operational boundaries.
#ifndef CIPHERCORE_NOISECTRL_H
#define CIPHERCORE_NOISECTRL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shared state holding the current noise multiplier applied across modules.
extern float g_noise_factor;

// Adjust the noise factor based on measured variance values.
void update_noise(float variance);
// Set the noise factor explicitly while enforcing valid bounds.
void set_noise_factor(float value);
// Fetch the current noise factor so callers can apply consistent scaling.
float get_noise_factor(void);
// Restore the noise factor to its neutral starting value.
void reset_noise_factor(void);
// Process a variance reading and optionally return derived error metrics.
void noisectrl_measure(float variance, float* error_out, float* variance_out);

#ifdef __cplusplus
}
#endif

#endif /* CIPHERCORE_NOISECTRL_H */
