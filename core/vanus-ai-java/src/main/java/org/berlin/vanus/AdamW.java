package org.berlin.vanus;

import java.util.*;

/**
 * AdamW optimizer: Adam moment estimates with decoupled weight decay and
 * global-norm gradient clipping applied uniformly across all parameters.
 */
public final class AdamW {
    // One trainable tensor per model parameter (weights and norm gains).
    private final List<Tensor> parameters;
    // Per-parameter first (mean) and second (uncentered variance) moment buffers, parallel to `parameters`.
    private final List<float[]> first = new ArrayList<>(), second = new ArrayList<>();
    private final float decay;
    private int steps;
    public AdamW(Collection<Tensor> parameters, float decay) {
        if (!Float.isFinite(decay) || decay < 0) throw new IllegalArgumentException("Invalid decay");
        this.parameters = List.copyOf(parameters); this.decay = decay;
        for (Tensor p : parameters) {
            if (p.grad == null) throw new IllegalArgumentException("Optimizer needs trainable tensors");
            first.add(new float[p.data.length]); second.add(new float[p.data.length]);
        }
    }
    /**
     * Applies one optimization step to every registered tensor's gradient.
     * Gradients are first clipped so the norm across all parameters combined
     * never exceeds {@code maxNorm}, then Adam's bias-corrected moving
     * averages drive the update, with decoupled weight decay applied last.
     *
     * @return the pre-clip global gradient norm, useful for logging.
     */
    public double step(float learningRate, float maxNorm) {
        if (!Float.isFinite(learningRate) || learningRate <= 0 || !Float.isFinite(maxNorm) || maxNorm <= 0) throw new IllegalArgumentException("Invalid optimizer options");
        // Compute the L2 norm of gradients across every parameter tensor combined.
        double squared = 0;
        for (Tensor p : parameters) for (float g : p.grad) squared += (double) g * g;
        double norm = Math.sqrt(squared);
        if (!Double.isFinite(norm)) throw new IllegalStateException("Nonfinite gradients; update aborted");
        // Scale factor that rescales gradients down to the max norm, or 1 if already within bounds.
        float clip = (float) Math.min(1, maxNorm / (norm + 1e-12));
        steps++;
        // Adam bias-correction terms compensate for the moments starting at zero.
        double correction1 = 1 - Math.pow(0.9, steps), correction2 = 1 - Math.pow(0.999, steps);
        for (int p = 0; p < parameters.size(); p++) {
            Tensor t = parameters.get(p); float[] m = first.get(p), v = second.get(p);
            for (int i = 0; i < t.data.length; i++) {
                float g = t.grad[i] * clip;
                // Exponential moving averages of the gradient (first moment) and its square (second moment).
                m[i] = 0.9f * m[i] + 0.1f * g; v[i] = 0.999f * v[i] + 0.001f * g * g;
                double update = (m[i] / correction1) / (Math.sqrt(v[i] / correction2) + 1e-8);
                // One-row RMSNorm weights are excluded from weight decay.
                t.data[i] -= learningRate * (float) (update + (t.rows > 1 ? decay * t.data[i] : 0));
            }
        }
        return norm;
    }
}
