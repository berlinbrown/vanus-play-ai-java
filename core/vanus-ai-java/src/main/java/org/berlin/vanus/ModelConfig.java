package org.berlin.vanus;

public record ModelConfig(int vocabulary, int width, int hidden, int layers, int heads, int context) {
    public ModelConfig {
        if (vocabulary != ByteTokenizer.VOCABULARY || width <= 0 || hidden <= 0 || layers <= 0 || heads <= 0 || context <= 0
                || width % heads != 0 || (width / heads) % 2 != 0)
            throw new IllegalArgumentException("Invalid model dimensions (byte vocabulary and even head width required)");
    }
    public static ModelConfig tiny() { return new ModelConfig(260, 32, 64, 1, 4, 128); }
    public static ModelConfig vanus20m() { return new ModelConfig(260, 448, 1280, 8, 8, 256); }
    public long parameterCount() {
        return (long) vocabulary * width + (long) layers * (4L * width * width + 3L * width * hidden + 2L * width) + width;
    }
}
