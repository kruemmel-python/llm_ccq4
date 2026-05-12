// C interface for SymBio GPU routines. Exposes the data structures and entry
// points required to exchange agent and field state between host code and the
// OpenCL kernels used throughout the project.
#ifndef SYMBIO_INTERFACE_H
#define SYMBIO_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif


#if defined(_WIN32) || defined(_WIN64)
  #define SYMBIO_API __declspec(dllexport)
#else
  #define SYMBIO_API __attribute__((visibility("default")))
#endif

// High-level agent representation shared between host-side control logic and
// GPU kernels.
typedef struct {
    float x;
    float y;
    float energy;
    float coupling;
} HPIOAgent;

// Views over multi-channel scalar fields stored on the host so they can be
// transferred to or from GPU buffers in a consistent order.
typedef struct {
    float* energy;
    float* pressure;
    float* gravity;
    float* magnetism;
    float* temperature;
    float* potential;
    float* drift_x;
    float* drift_y;
    int    cell_count;
} SubQGMultiFieldHostView;

// Upload multifield state from the host into GPU buffers for simulation.
SYMBIO_API int subqg_set_multifield_state(int gpu_index,
                               int cell_count,
                               const float* energy,
                               const float* pressure,
                               const float* gravity,
                               const float* magnetism,
                               const float* temperature,
                               const float* potential,
                               const float* drift_x,
                               const float* drift_y);

// Retrieve multifield state from the GPU into host-accessible buffers for
// inspection or checkpointing.
SYMBIO_API int subqg_get_multifield_state(int gpu_index,
                               int max_cells,
                               float* energy,
                               float* pressure,
                               float* gravity,
                               float* magnetism,
                               float* temperature,
                               float* potential,
                               float* drift_x,
                               float* drift_y);

// Read back a specific debug channel from the GPU to help diagnose kernels.
SYMBIO_API int subqg_debug_read_channel(int gpu_index,
                             int channel,
                             float* out_host,
                             int max_len);

// Advance genetic agents on the GPU and emit the updated state buffer.
SYMBIO_API int update_genetic_agents(int gpu_index,
                          const float* agent_states_in,
                          float* agent_states_out,
                          int state_stride,
                          int agent_count,
                          float time_step);

// ---------------- Neuropersona GPU API ----------------
// Initialize GPU-side buffers for the Neuropersona state.
// weight_mode: 0 = dense, 1 = csr (reserved for future).
int neuropersona_gpu_init(int gpu_index, int node_count, int weight_mode, int seed);

// Execute one online training step on the GPU.
// input_signal: length node_count or NULL to reuse previous signal.
// input_category_index: -1 to skip explicit category cue.
// out_metric: optional pointer for loss/metric output (may be NULL).
int neuropersona_gpu_train_step(int gpu_index,
                                int node_count,
                                const float* input_signal,
                                int input_category_index,
                                float learning_rate,
                                float decay_rate,
                                float reward_factor,
                                float noise_level,
                                float* out_metric);

// Recall top-k categories for a cue vector.
int neuropersona_gpu_recall(int gpu_index,
                            int node_count,
                            const float* cue_vector,
                            int top_k,
                            int* out_indices,
                            float* out_scores);

// Persist GPU-side checkpoint to a file.
int neuropersona_gpu_checkpoint_save(int gpu_index, const char* path);

// Load GPU-side checkpoint from a file.
int neuropersona_gpu_checkpoint_load(int gpu_index, const char* path);


// ---------------- Enterprise Algorithm Pack API ----------------
// Resonant field propagation over a fixed-K neighbor graph.
// Buffers are OpenCL cl_mem handles passed as void* for ABI stability.
SYMBIO_API int execute_resonant_field_step_gpu(
    int gpu_index,
    void* state_buf,
    void* velocity_buf,
    void* drive_buf,
    void* energy_buf,
    void* neighbors_buf,
    void* weights_buf,
    int N,
    int K,
    float dt,
    float damping,
    float coupling,
    float inertia,
    float clamp_abs);

// GPU-side activity compaction plus cheap sleep-state decay.
// active_count_buf must be a cl_mem buffer with room for one uint.
SYMBIO_API int execute_energy_gated_scheduler_gpu(
    int gpu_index,
    void* energy_buf,
    void* nutrient_buf,
    void* active_flags_buf,
    void* active_indices_buf,
    void* active_count_buf,
    int N,
    float threshold,
    float sleep_decay,
    float nutrient_recovery);

// Tabular morphogenesis rule execution.
// cell_type_buf is uchar[N]; rule_in_type/rule_out_type are int[R].
SYMBIO_API int execute_morphogenetic_rule_step_gpu(
    int gpu_index,
    void* cell_type_buf,
    void* nutrient_buf,
    void* energy_buf,
    void* potential_buf,
    void* rule_in_type_buf,
    void* rule_min_nutrient_buf,
    void* rule_min_energy_buf,
    void* rule_out_type_buf,
    void* rule_delta_potential_buf,
    int N,
    int R,
    float nutrient_cost);


// Thermodynamic Langevin relaxation using noise as a computation primitive.
// state/momentum/bias/free_energy are float[N]; neighbors int[N*K]; weights float[N*K].
SYMBIO_API int execute_thermodynamic_langevin_step_gpu(
    int gpu_index,
    void* state_buf,
    void* momentum_buf,
    void* bias_buf,
    void* free_energy_buf,
    void* neighbors_buf,
    void* weights_buf,
    int N,
    int K,
    float dt,
    float temperature,
    float quartic_alpha,
    float bias_stiffness,
    float coupling,
    float friction,
    unsigned int seed,
    float clamp_abs);

// Brain-inspired reservoir update with local excitatory/inhibitory homeostasis.
// neuron_type_buf is uchar[N] where 1=excitatory and 0=inhibitory.
SYMBIO_API int execute_ei_plastic_reservoir_step_gpu(
    int gpu_index,
    void* signal_buf,
    void* next_signal_buf,
    void* input_drive_buf,
    void* neighbors_buf,
    void* weights_buf,
    void* neuron_type_buf,
    void* gain_e_buf,
    void* gain_i_buf,
    void* activity_ema_buf,
    int N,
    int K,
    float leak,
    float target_activity,
    float plasticity_rate,
    float input_scale,
    float clamp_gain);

// Quantum-inspired tensor-bond entropy gate for sparse rank/bond scheduling.
// left/right_factors are float[B*D]; active_count_buf must hold one uint.
SYMBIO_API int execute_tensor_bond_entropy_gate_gpu(
    int gpu_index,
    void* left_factors_buf,
    void* right_factors_buf,
    void* bond_value_buf,
    void* bond_entropy_buf,
    void* update_flags_buf,
    void* active_bonds_buf,
    void* active_count_buf,
    int B,
    int D,
    float entropy_threshold,
    float residual_threshold,
    float smoothing);

// Deep-substrate dispatcher for LLM layer orchestration.
// layer_role: int[L], where 0=dense/embedding/logits, 1=attention,
// 2=MLP, 3=norm/residual, 4=sparse router.
// emotion_state: float[4] = precision, novelty, dream, risk.
// route_scores: float[L*route_count], route_count must be at least 6.
// selected_route IDs: 0=dense, 1=mycel, 2=tensor_bond, 3=reservoir,
// 4=langevin, 5=subqg_quantum.
// dispatch_params: float[L*4] = precision_scale, exploration_scale,
// risk_scale, relevance.
SYMBIO_API int execute_deep_substrate_dispatch_gpu(
    int gpu_index,
    void* layer_role_buf,
    void* layer_priority_buf,
    void* emotion_state_buf,
    void* nutrient_state_buf,
    void* route_scores_buf,
    void* selected_route_buf,
    void* dispatch_params_buf,
    void* active_layers_buf,
    void* active_count_buf,
    int layer_count,
    int route_count,
    float relevance_threshold,
    float exploration_bias);

// Execute y = W_q4 * x where W_q4 is CCQ4-style signed INT4 blocks.
// packed_weights stores two signed 4-bit values per byte, low nibble first.
// scales stores one float per row/block, with block_count=ceil(cols/block_size).
SYMBIO_API int execute_ccq4_matvec_gpu(
    int gpu_index,
    void* packed_weights_buf,
    void* scales_buf,
    void* input_vec_buf,
    void* output_vec_buf,
    int rows,
    int cols,
    int block_size);

// Register CCQ4 packed weights/scales as persistent driver-owned GPU buffers.
// Returns a positive handle in out_handle. Release with cc_release_persistent_weight.
SYMBIO_API int cc_register_persistent_ccq4_weight(
    int gpu_index,
    const void* packed_host,
    size_t packed_bytes,
    const void* scales_host,
    size_t scale_bytes,
    int rows,
    int cols,
    int block_size,
    int* out_handle);

// Execute y = resident_W_q4 * x using a handle returned by cc_register_persistent_ccq4_weight.
SYMBIO_API int cc_execute_resident_ccq4_matvec(
    int gpu_index,
    int weight_handle,
    void* input_vec_buf,
    void* output_vec_buf);

SYMBIO_API int cc_release_persistent_weight(int weight_handle);
SYMBIO_API int cc_release_all_persistent_weights(void);
SYMBIO_API int cc_get_persistent_weight_count(void);


#ifdef __cplusplus
}
#endif

#endif /* SYMBIO_INTERFACE_H */
