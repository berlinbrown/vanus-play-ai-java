package org.berlin.vanus;

import java.util.*;
import java.util.function.Supplier;

/** Contiguous float32 CPU matrices and a reverse-mode differentiation tape. */
public final class Tensor {
    private static final ThreadLocal<Boolean> TRACK = ThreadLocal.withInitial(() -> true);
    public final int rows, cols;
    public final float[] data;
    public final float[] grad;
    private final Tensor[] parents;
    private Runnable backward = () -> {};

    private Tensor(int rows, int cols, boolean requiresGrad, Tensor... parents) {
        if (rows <= 0 || cols <= 0) throw new IllegalArgumentException("Positive dimensions required");
        this.rows = rows; this.cols = cols;
        data = new float[Math.multiplyExact(rows, cols)];
        grad = requiresGrad ? new float[data.length] : null;
        this.parents = requiresGrad ? parents : new Tensor[0];
    }

    public static Tensor zeros(int rows, int cols, boolean requiresGrad) {
        return new Tensor(rows, cols, requiresGrad);
    }
    public static Tensor of(int rows, int cols, float... values) {
        Tensor t = zeros(rows, cols, false);
        if (values.length != t.data.length) throw new IllegalArgumentException("Shape mismatch");
        System.arraycopy(values, 0, t.data, 0, values.length);
        return t;
    }
    public static Tensor parameter(int rows, int cols, Random random, float std) {
        Tensor t = zeros(rows, cols, true);
        for (int i = 0; i < t.data.length; i++) t.data[i] = (float) random.nextGaussian() * std;
        return t;
    }
    public static <T> T noGrad(Supplier<T> body) {
        boolean old = TRACK.get(); TRACK.set(false);
        try { return body.get(); } finally { TRACK.set(old); }
    }
    private static Tensor result(int rows, int cols, Tensor... parents) {
        boolean track = TRACK.get() && Arrays.stream(parents).anyMatch(p -> p.grad != null);
        return new Tensor(rows, cols, track, parents);
    }
    private static void accumulate(Tensor t, int i, float value) {
        if (t.grad != null) t.grad[i] += value;
    }
    private void sameShape(Tensor b) {
        if (rows != b.rows || cols != b.cols) throw new IllegalArgumentException("Shape mismatch");
    }
    public void zeroGrad() { if (grad != null) Arrays.fill(grad, 0); }
    public Tensor detach() { return of(rows, cols, data); }
    public String shape() { return "[" + rows + ", " + cols + "]"; }
    public String dtype() { return "float32"; }
    public String device() { return "cpu"; }

    public void backward() {
        if (data.length != 1 || grad == null) throw new IllegalStateException("Expected differentiable scalar loss");
        List<Tensor> order = new ArrayList<>();
        visit(this, Collections.newSetFromMap(new IdentityHashMap<>()), order);
        // Leaf gradients accumulate across backward calls; intermediates are recomputed.
        for (Tensor t : order) if (t.parents.length > 0) t.zeroGrad();
        grad[0] = 1;
        for (int i = order.size() - 1; i >= 0; i--) order.get(i).backward.run();
    }
    private static void visit(Tensor t, Set<Tensor> seen, List<Tensor> order) {
        if (!seen.add(t)) return;
        for (Tensor parent : t.parents) visit(parent, seen, order);
        order.add(t);
    }
    public Tensor add(Tensor b) {
        sameShape(b); Tensor out = result(rows, cols, this, b);
        for (int i = 0; i < data.length; i++) out.data[i] = data[i] + b.data[i];
        if (out.grad != null) out.backward = () -> {
            for (int i = 0; i < data.length; i++) {
                accumulate(this, i, out.grad[i]); accumulate(b, i, out.grad[i]);
            }
        };
        return out;
    }
    public Tensor multiply(Tensor b) {
        sameShape(b); Tensor out = result(rows, cols, this, b);
        for (int i = 0; i < data.length; i++) out.data[i] = data[i] * b.data[i];
        if (out.grad != null) out.backward = () -> {
            for (int i = 0; i < data.length; i++) {
                accumulate(this, i, out.grad[i] * b.data[i]);
                accumulate(b, i, out.grad[i] * data[i]);
            }
        };
        return out;
    }
    public Tensor matmul(Tensor b) {
        if (cols != b.rows) throw new IllegalArgumentException("matmul: " + shape() + " x " + b.shape());
        Tensor out = result(rows, b.cols, this, b);
        for (int i = 0; i < rows; i++) for (int k = 0; k < cols; k++) {
            float a = data[i * cols + k];
            for (int j = 0; j < b.cols; j++) out.data[i * b.cols + j] += a * b.data[k * b.cols + j];
        }
        if (out.grad != null) out.backward = () -> {
            for (int i = 0; i < rows; i++) for (int k = 0; k < cols; k++) {
                float sum = 0, a = data[i * cols + k];
                for (int j = 0; j < b.cols; j++) {
                    float g = out.grad[i * b.cols + j];
                    sum += g * b.data[k * b.cols + j];
                    accumulate(b, k * b.cols + j, a * g);
                }
                accumulate(this, i * cols + k, sum);
            }
        };
        return out;
    }
    public Tensor transpose() {
        Tensor out = result(cols, rows, this);
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) out.data[j * rows + i] = data[i * cols + j];
        if (out.grad != null) out.backward = () -> {
            for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) accumulate(this, i * cols + j, out.grad[j * rows + i]);
        };
        return out;
    }
    public Tensor embedding(int[] tokenIds) {
        int[] ids = tokenIds.clone();
        Tensor out = result(ids.length, cols, this);
        for (int i = 0; i < ids.length; i++) {
            if (ids[i] < 0 || ids[i] >= rows) throw new IllegalArgumentException("Invalid token ID");
            System.arraycopy(data, ids[i] * cols, out.data, i * cols, cols);
        }
        if (out.grad != null) out.backward = () -> {
            for (int i = 0; i < ids.length; i++) for (int j = 0; j < cols; j++) accumulate(this, ids[i] * cols + j, out.grad[i * cols + j]);
        };
        return out;
    }
    public Tensor silu() {
        Tensor out = result(rows, cols, this);
        for (int i = 0; i < data.length; i++) out.data[i] = data[i] * sigmoid(data[i]);
        if (out.grad != null) out.backward = () -> {
            for (int i = 0; i < data.length; i++) {
                float s = sigmoid(data[i]);
                accumulate(this, i, out.grad[i] * (s + data[i] * s * (1 - s)));
            }
        };
        return out;
    }
    private static float sigmoid(float x) { return (float) (1 / (1 + Math.exp(-x))); }

    public Tensor rmsNorm(Tensor weight, float epsilon) {
        if (weight.rows != 1 || weight.cols != cols || !(epsilon > 0)) throw new IllegalArgumentException("Invalid RMSNorm arguments");
        Tensor out = result(rows, cols, this, weight);
        float[] inverse = new float[rows];
        for (int r = 0; r < rows; r++) {
            double sum = 0;
            for (int c = 0; c < cols; c++) { float x = data[r * cols + c]; sum += x * x; }
            inverse[r] = (float) (1 / Math.sqrt(sum / cols + epsilon));
            for (int c = 0; c < cols; c++) out.data[r * cols + c] = data[r * cols + c] * inverse[r] * weight.data[c];
        }
        if (out.grad != null) out.backward = () -> {
            for (int r = 0; r < rows; r++) {
                float dot = 0, inv = inverse[r];
                for (int c = 0; c < cols; c++) dot += out.grad[r * cols + c] * weight.data[c] * data[r * cols + c];
                for (int c = 0; c < cols; c++) {
                    int i = r * cols + c;
                    accumulate(this, i, out.grad[i] * weight.data[c] * inv - data[i] * inv * inv * inv * dot / cols);
                    accumulate(weight, c, out.grad[i] * data[i] * inv);
                }
            }
        };
        return out;
    }
    public Tensor rope(int heads) {
        if (heads <= 0 || cols % heads != 0 || (cols / heads) % 2 != 0) throw new IllegalArgumentException("RoPE needs even head width");
        int width = cols / heads;
        Tensor out = result(rows, cols, this);
        for (int t = 0; t < rows; t++) for (int h = 0; h < heads; h++) for (int j = 0; j < width; j += 2) {
            double angle = t / Math.pow(10000, (double) j / width);
            float c = (float) Math.cos(angle), s = (float) Math.sin(angle);
            int i = t * cols + h * width + j;
            out.data[i] = data[i] * c - data[i + 1] * s;
            out.data[i + 1] = data[i] * s + data[i + 1] * c;
        }
        if (out.grad != null) out.backward = () -> {
            for (int t = 0; t < rows; t++) for (int h = 0; h < heads; h++) for (int j = 0; j < width; j += 2) {
                double angle = t / Math.pow(10000, (double) j / width);
                float c = (float) Math.cos(angle), s = (float) Math.sin(angle);
                int i = t * cols + h * width + j;
                accumulate(this, i, out.grad[i] * c + out.grad[i + 1] * s);
                accumulate(this, i + 1, -out.grad[i] * s + out.grad[i + 1] * c);
            }
        };
        return out;
    }

    /** Fused causal softmax(Q K^T / sqrt(headWidth)) V, storing no future scores. */
    public static Tensor attention(Tensor q, Tensor k, Tensor v, int heads) {
        q.sameShape(k); q.sameShape(v);
        if (heads <= 0 || q.cols % heads != 0) throw new IllegalArgumentException("Invalid heads");
        int n = q.rows, d = q.cols, width = d / heads;
        float scale = (float) (1 / Math.sqrt(width));
        Tensor out = result(n, d, q, k, v);
        float[] probabilities = new float[Math.multiplyExact(heads, Math.multiplyExact(n, n))];
        for (int h = 0; h < heads; h++) for (int t = 0; t < n; t++) {
            int base = (h * n + t) * n;
            float max = Float.NEGATIVE_INFINITY;
            for (int s = 0; s <= t; s++) {
                float dot = 0;
                for (int j = 0; j < width; j++) dot += q.data[t * d + h * width + j] * k.data[s * d + h * width + j];
                probabilities[base + s] = dot * scale; max = Math.max(max, dot * scale);
            }
            float total = 0;
            for (int s = 0; s <= t; s++) { probabilities[base + s] = (float) Math.exp(probabilities[base + s] - max); total += probabilities[base + s]; }
            for (int s = 0; s <= t; s++) {
                float p = probabilities[base + s] /= total;
                for (int j = 0; j < width; j++) out.data[t * d + h * width + j] += p * v.data[s * d + h * width + j];
            }
        }
        if (out.grad != null) out.backward = () -> {
            float[] dp = new float[n];
            for (int h = 0; h < heads; h++) for (int t = 0; t < n; t++) {
                int base = (h * n + t) * n;
                float dot = 0;
                for (int s = 0; s <= t; s++) {
                    dp[s] = 0;
                    for (int j = 0; j < width; j++) {
                        float g = out.grad[t * d + h * width + j];
                        dp[s] += g * v.data[s * d + h * width + j];
                        accumulate(v, s * d + h * width + j, probabilities[base + s] * g);
                    }
                    dot += probabilities[base + s] * dp[s];
                }
                for (int s = 0; s <= t; s++) {
                    float ds = probabilities[base + s] * (dp[s] - dot) * scale;
                    for (int j = 0; j < width; j++) {
                        int qi = t * d + h * width + j, ki = s * d + h * width + j;
                        accumulate(q, qi, ds * k.data[ki]); accumulate(k, ki, ds * q.data[qi]);
                    }
                }
            }
        };
        return out;
    }

    /** Mean cross entropy; target -1 excludes prompt/padding positions. */
    public Tensor crossEntropy(int[] targetIds) {
        int[] targets = targetIds.clone();
        if (targets.length != rows) throw new IllegalArgumentException("One target per row required");
        int count = 0;
        for (int target : targets) {
            if (target < -1 || target >= cols) throw new IllegalArgumentException("Invalid target");
            if (target >= 0) count++;
        }
        if (count == 0) throw new IllegalArgumentException("No supervised targets");
        final int denominator = count;
        Tensor out = result(1, 1, this);
        float[] probabilities = new float[data.length];
        double loss = 0;
        for (int r = 0; r < rows; r++) {
            if (targets[r] < 0) continue;
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < cols; c++) max = Math.max(max, data[r * cols + c]);
            double sum = 0;
            for (int c = 0; c < cols; c++) { probabilities[r * cols + c] = (float) Math.exp(data[r * cols + c] - max); sum += probabilities[r * cols + c]; }
            loss += Math.log(sum) + max - data[r * cols + targets[r]];
            for (int c = 0; c < cols; c++) probabilities[r * cols + c] /= (float) sum;
        }
        out.data[0] = (float) (loss / denominator);
        if (out.grad != null) out.backward = () -> {
            for (int r = 0; r < rows; r++) if (targets[r] >= 0) for (int c = 0; c < cols; c++)
                accumulate(this, r * cols + c, out.grad[0] * (probabilities[r * cols + c] - (c == targets[r] ? 1 : 0)) / denominator);
        };
        return out;
    }
}
