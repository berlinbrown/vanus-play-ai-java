package org.berlin.vanus;

import java.util.*;

public final class AdamW {
    private final List<Tensor> parameters;
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
    public double step(float learningRate, float maxNorm) {
        if (!Float.isFinite(learningRate) || learningRate <= 0 || !Float.isFinite(maxNorm) || maxNorm <= 0) throw new IllegalArgumentException("Invalid optimizer options");
        double squared = 0;
        for (Tensor p : parameters) for (float g : p.grad) squared += (double) g * g;
        double norm = Math.sqrt(squared);
        if (!Double.isFinite(norm)) throw new IllegalStateException("Nonfinite gradients; update aborted");
        float clip = (float) Math.min(1, maxNorm / (norm + 1e-12));
        steps++;
        double correction1 = 1 - Math.pow(0.9, steps), correction2 = 1 - Math.pow(0.999, steps);
        for (int p = 0; p < parameters.size(); p++) {
            Tensor t = parameters.get(p); float[] m = first.get(p), v = second.get(p);
            for (int i = 0; i < t.data.length; i++) {
                float g = t.grad[i] * clip;
                m[i] = 0.9f * m[i] + 0.1f * g; v[i] = 0.999f * v[i] + 0.001f * g * g;
                double update = (m[i] / correction1) / (Math.sqrt(v[i] / correction2) + 1e-8);
                // One-row RMSNorm weights are excluded from weight decay.
                t.data[i] -= learningRate * (float) (update + (t.rows > 1 ? decay * t.data[i] : 0));
            }
        }
        return norm;
    }
}
