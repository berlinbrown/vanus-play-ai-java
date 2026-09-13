package org.berlin.vanus;

/**
 * Immutable hyperparameters describing a {@link Transformer} architecture:
 * vocabulary size, embedding width, feed-forward hidden size, number of
 * decoder layers, attention heads, and the maximum context length.
 */
public record ModelConfig(int vocabulary, int width, int hidden, int layers, int heads, int context) {
    // Validates dimensions eagerly so a malformed config fails at construction, not deep inside the model.
    public ModelConfig {
        if (vocabulary != ByteTokenizer.VOCABULARY || width <= 0 || hidden <= 0 || layers <= 0 || heads <= 0 || context <= 0
                || width % heads != 0 || (width / heads) % 2 != 0)
            throw new IllegalArgumentException("Invalid model dimensions (byte vocabulary and even head width required)");
    }
    /** Minimal configuration for fast smoke tests. */
    public static ModelConfig tiny() { return new ModelConfig(260, 32, 64, 1, 4, 128); }
    /** ~20M parameter preset used for real training/generation runs. */
    public static ModelConfig vanus20m() { return new ModelConfig(260, 448, 1280, 8, 8, 256); }
    /** Estimated total trainable parameter count: embedding table plus per-layer attention and SwiGLU weights. */
    public long parameterCount() {
        return (long) vocabulary * width + (long) layers * (4L * width * width + 3L * width * hidden + 2L * width) + width;
    }
}
