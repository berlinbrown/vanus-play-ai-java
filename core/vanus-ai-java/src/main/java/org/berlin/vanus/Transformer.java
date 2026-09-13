package org.berlin.vanus;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Vanus decoder: pre-RMSNorm, RoPE, multi-head attention, SwiGLU, tied output.
 */
public final class Transformer {
    public final ModelConfig config;
    // Ordered name -> tensor map covering every trainable weight, used for
    // optimizer wiring and checkpointing.
    private final LinkedHashMap<String, Tensor> parameters = new LinkedHashMap<>();
    private final Random random;
    // Token embedding table (also reused, transposed, as the output projection) and
    // final RMSNorm gain.
    private final Tensor embedding, norm;
    private final List<Block> blocks = new ArrayList<>();

    public Transformer(ModelConfig config, long seed) {
        this.config = config;
        random = new Random(seed);
        embedding = weight("embedding", config.vocabulary(), config.width());
        for (int i = 0; i < config.layers(); i++)
            blocks.add(new Block("blocks." + i));
        norm = norm("norm");
    }

    // Registers and returns a new randomly-initialized trainable weight matrix.
    private Tensor weight(String name, int rows, int cols) {
        Tensor p = Tensor.parameter(rows, cols, random, 0.02f);
        parameters.put(name, p);
        return p;
    }

    // Registers and returns a new RMSNorm gain vector, initialized to ones.
    private Tensor norm(String name) {
        Tensor p = Tensor.zeros(1, config.width(), true);
        Arrays.fill(p.data, 1);
        parameters.put(name, p);
        return p;
    }

    public Map<String, Tensor> namedParameters() {
        return Collections.unmodifiableMap(parameters);
    }

    public Collection<Tensor> parameters() {
        return Collections.unmodifiableCollection(parameters.values());
    }

    public void zeroGrad() {
        parameters.values().forEach(Tensor::zeroGrad);
    }

    /**
     * Runs the full decoder stack over a token sequence, returning unnormalized
     * logits over the vocabulary per position.
     */
    public Tensor forward(int[] tokens) {
        if (tokens.length == 0 || tokens.length > config.context())
            throw new IllegalArgumentException("Sequence outside context limit " + config.context());
        Tensor x = embedding.embedding(tokens);
        for (Block block : blocks)
            x = block.forward(x);
        // Tied embedding: reuse the input embedding table (transposed) as the output
        // projection.
        return x.rmsNorm(norm, 1e-5f).matmul(embedding.transpose());
    }

    /**
     * One decoder layer: pre-norm causal self-attention followed by a pre-norm
     * SwiGLU feed-forward, each with a residual connection.
     */
    private final class Block {
        final Tensor attentionNorm, ffnNorm, q, k, v, o, gate, up, down;

        Block(String name) {
            int d = config.width(), f = config.hidden();
            attentionNorm = norm(name + ".attentionNorm");
            ffnNorm = norm(name + ".ffnNorm");
            q = weight(name + ".q", d, d);
            k = weight(name + ".k", d, d);
            v = weight(name + ".v", d, d);
            o = weight(name + ".o", d, d);
            gate = weight(name + ".gate", d, f);
            up = weight(name + ".up", d, f);
            down = weight(name + ".down", f, d);
        }

        Tensor forward(Tensor x) {
            // Self-attention sub-layer: normalize, project to Q/K/V, rotate Q/K with RoPE,
            // attend, project back, add residual.
            Tensor a = x.rmsNorm(attentionNorm, 1e-5f);
            Tensor attention = Tensor.attention(a.matmul(q).rope(config.heads()), a.matmul(k).rope(config.heads()),
                    a.matmul(v), config.heads());
            Tensor residual = x.add(attention.matmul(o));
            // SwiGLU feed-forward sub-layer: normalize, gate*up (SiLU-activated) projected
            // back down, add residual.
            Tensor f = residual.rmsNorm(ffnNorm, 1e-5f);
            return residual.add(f.matmul(gate).silu().multiply(f.matmul(up)).matmul(down));
        }
    }

    /**
     * Autoregressively generates text after the given prompt using
     * temperature/top-k sampling.
     */
    public String generate(String prompt, int maxTokens, double temperature, int topK, long seed) {
        if (maxTokens < 0 || !Double.isFinite(temperature) || temperature < 0 || topK < 1 || topK > config.vocabulary())
            throw new IllegalArgumentException("Invalid generation options");
        ByteTokenizer tokenizer = new ByteTokenizer();
        int[] prefix = tokenizer.prompt(prompt);
        if (prefix.length >= config.context())
            throw new IllegalArgumentException("Prompt leaves no generation space");
        return Tensor.noGrad(() -> {
            int[] sequence = Arrays.copyOf(prefix, config.context());
            int length = prefix.length;
            Random rng = new Random(seed);
            for (int step = 0; step < maxTokens && length < sequence.length; step++) {
                Tensor logits = forward(Arrays.copyOf(sequence, length));
                int next = sample(logits, temperature, topK, rng);
                if (next == ByteTokenizer.EOS)
                    break;
                sequence[length++] = next;
            }
            return tokenizer.decode(Arrays.copyOfRange(sequence, prefix.length, length));
        });
    }

    // Picks the next token from the final position's logits: greedy argmax if
    // temperature is 0, else top-k temperature sampling.
    private static int sample(Tensor logits, double temperature, int topK, Random random) {
        int offset = (logits.rows - 1) * logits.cols;
        Integer[] ids = new Integer[257];
        for (int i = 0; i < 256; i++)
            ids[i] = i;
        ids[256] = ByteTokenizer.EOS;
        Arrays.sort(ids, (a, b) -> Float.compare(logits.data[offset + b], logits.data[offset + a]));
        if (temperature == 0)
            return ids[0];
        int count = Math.min(topK, ids.length);
        double[] weights = new double[count];
        double total = 0;
        for (int i = 0; i < count; i++)
            total += weights[i] = Math.exp((logits.data[offset + ids[i]] - logits.data[offset + ids[0]]) / temperature);
        double draw = random.nextDouble() * total;
        for (int i = 0; i < count; i++) {
            draw -= weights[i];
            if (draw <= 0)
                return ids[i];
        }
        return ids[count - 1];
    }

    /**
     * Versioned weights/config only. Optimizer state is deliberately not a resume
     * checkpoint.
     */
    public void save(Path path) throws IOException {
        Path absolute = path.toAbsolutePath();
        Files.createDirectories(absolute.getParent());
        // Write to a temp file first and atomically move it into place so a crash never
        // leaves a truncated checkpoint.
        Path temporary = Files.createTempFile(absolute.getParent(), ".vanus-", ".tmp");
        try {
            try (DataOutputStream out = new DataOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(temporary)))) {
                out.writeInt(0x56414e31);
                out.writeInt(1);
                for (int value : new int[] { config.vocabulary(), config.width(), config.hidden(), config.layers(),
                        config.heads(), config.context() })
                    out.writeInt(value);
                out.writeInt(parameters.size());
                for (var entry : parameters.entrySet()) {
                    out.writeUTF(entry.getKey());
                    Tensor t = entry.getValue();
                    out.writeInt(t.rows);
                    out.writeInt(t.cols);
                    for (float value : t.data)
                        out.writeFloat(value);
                }
            }
            try {
                Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    /**
     * Loads a checkpoint written by {@link #save}, validating magic/version,
     * bounds, and per-tensor shapes before trusting the data.
     */
    public static Transformer load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            if (in.readInt() != 0x56414e31 || in.readInt() != 1)
                throw new IOException("Unsupported Vanus checkpoint");
            ModelConfig c;
            try {
                c = new ModelConfig(in.readInt(), in.readInt(), in.readInt(), in.readInt(), in.readInt(), in.readInt());
            } catch (IllegalArgumentException e) {
                throw new IOException("Invalid checkpoint configuration", e);
            }
            // Reject implausible or truncated files before allocating memory for them.
            if (c.parameterCount() > 100_000_000L || c.parameterCount() * 4 > Files.size(path) || c.context() > 8192)
                throw new IOException("Checkpoint exceeds supported bounds or is truncated");
            Transformer model = new Transformer(c, 0);
            if (in.readInt() != model.parameters.size())
                throw new IOException("Parameter count mismatch");
            for (var entry : model.parameters.entrySet()) {
                Tensor t = entry.getValue();
                if (!in.readUTF().equals(entry.getKey()) || in.readInt() != t.rows || in.readInt() != t.cols)
                    throw new IOException("Parameter mismatch");
                for (int i = 0; i < t.data.length; i++) {
                    t.data[i] = in.readFloat();
                    if (!Float.isFinite(t.data[i]))
                        throw new IOException("Nonfinite checkpoint weight");
                }
            }
            if (in.read() != -1)
                throw new IOException("Trailing checkpoint data");
            return model;
        }
    }
}
