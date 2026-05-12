/* SubQG v6: unchanged from v5; grid inference/projection fix is in CC_OpenCL.c. */
// Noise control implementation for CipherCore. Adjusts and reports a global
// noise scaling factor based on observed variance to keep downstream
// computations numerically stable.
#include "CipherCore_NoiseCtrl.h"

#include <math.h>

#ifndef THRESH_HIGH
#define THRESH_HIGH 0.0125f
#endif

#ifndef THRESH_LOW
#define THRESH_LOW 0.0030f
#endif

float g_noise_factor = 1.0f;

// Adapt the noise factor whenever the measured variance crosses the
// configured thresholds. This keeps the factor within a healthy operating
// window and avoids runaway amplification or suppression.
void update_noise(float variance) {
    if (!isfinite(variance)) {
        variance = THRESH_HIGH;
    }

    /*
     * v4: The driver feeds this controller with profiling-scale variance
     * (duration_ms * 0.001), not physical field variance.  The old 0.5/1.5
     * window therefore forced g_noise_factor to the 2.0 cap on every normal
     * SubQG run.  Keep the feedback band near the observed 3e-3..1.2e-2 range
     * and adapt gently so resonance structure is not washed out.
     */
    if (variance > THRESH_HIGH) {
        g_noise_factor *= 0.96f;
    } else if (variance < THRESH_LOW) {
        g_noise_factor *= 1.035f;
    }
    if (g_noise_factor < 0.45f) {
        g_noise_factor = 0.45f;
    } else if (g_noise_factor > 1.35f) {
        g_noise_factor = 1.35f;
    }
}

// Explicitly set the noise factor while clamping it to the supported range,
// so callers cannot accidentally drive the control loop into invalid states.
void set_noise_factor(float value) {
    if (!isfinite(value)) {
        value = 1.0f;
    }
    if (value < 0.45f) {
        value = 0.45f;
    } else if (value > 1.35f) {
        value = 1.35f;
    }
    g_noise_factor = value;
}

// Expose the current noise factor so other modules can scale their signals
// consistently with the control loop's internal state.
float get_noise_factor(void) {
    return g_noise_factor;
}

// Reset the noise factor to the neutral baseline used during initialisation.
void reset_noise_factor(void) {
    g_noise_factor = 1.0f;
}

// Convert a variance reading into an error metric that reflects the absolute
// deviation from the nominal value. The result is scaled to moderate the
// influence of extreme outliers.
static float compute_error_from_variance(float variance) {
    const float nominal = 0.0060f;
    float deviation = variance - nominal;
    return fabsf(deviation) * 0.5f;
}

// Public measurement entry point: update the control loop with the latest
// variance, optionally report the raw variance, and output the derived error
// metric for diagnostics or logging.
void noisectrl_measure(float variance, float* error_out, float* variance_out) {
    update_noise(variance);
    if (variance_out) {
        *variance_out = variance;
    }
    if (error_out) {
        *error_out = compute_error_from_variance(variance);
    }
}
