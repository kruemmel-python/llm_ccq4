#pragma once
/*
 * SOMNIA Shadow Enterprise native ABI for CC_OpenCl.
 *
 * This ABI turns the driver into a VRAM-resident shadow computer:
 *   - CPU writes scalar/packet pulses.
 *   - VRAM keeps semantic state resident.
 *   - CPU reads only a compressed signal membrane.
 */
#include <stdint.h>

#ifdef _WIN32
  #ifdef __cplusplus
    #define SOMNIA_EXPORT extern "C" __declspec(dllexport)
  #else
    #define SOMNIA_EXPORT __declspec(dllexport)
  #endif
#else
  #ifdef __cplusplus
    #define SOMNIA_EXPORT extern "C"
  #else
    #define SOMNIA_EXPORT
  #endif
#endif

typedef struct SomniaShadowPulsePacket {
    int kind;
    uint64_t target;
    float strength;
    float phase;
    float payload_a;
    float payload_b;
    uint32_t flags;
    double timestamp;
} SomniaShadowPulsePacket;

typedef struct SomniaShadowSignalPacket {
    int code;
    float confidence;
    float urgency;
    float risk;
    float novelty;
    float entropy;
    uint64_t cycle;
    uint32_t flags;
} SomniaShadowSignalPacket;

SOMNIA_EXPORT int shadow_init(int gpu_index, int cells, int channels, int neighbors, int dreams, int action_vector_len);

SOMNIA_EXPORT int shadow_inject_pulse(
    int gpu_index,
    int event_type,
    uint64_t target,
    float strength,
    float phase,
    float payload_a,
    float payload_b,
    int flags
);

SOMNIA_EXPORT int shadow_inject_pulse_packet(
    int gpu_index,
    const SomniaShadowPulsePacket* pulse
);

SOMNIA_EXPORT int shadow_start_loop(int gpu_index, int cycles, int mode);
SOMNIA_EXPORT int shadow_cycle(int gpu_index, int cycles, int mode);

SOMNIA_EXPORT int shadow_read_signal(
    int gpu_index,
    int* out_code,
    float* out_confidence,
    float* out_urgency,
    float* out_risk,
    float* out_novelty,
    float* out_entropy,
    float* out_action_vector,
    int action_vector_len
);

SOMNIA_EXPORT int shadow_read_signal_packet(
    int gpu_index,
    SomniaShadowSignalPacket* out_signal,
    float* out_action_vector,
    int action_vector_len
);

SOMNIA_EXPORT int shadow_set_abort_flag(int gpu_index, int enabled);
SOMNIA_EXPORT int shadow_checkpoint_save(int gpu_index, const char* path);
SOMNIA_EXPORT int shadow_checkpoint_load(int gpu_index, const char* path);
