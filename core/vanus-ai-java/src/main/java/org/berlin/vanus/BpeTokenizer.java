package org.berlin.vanus;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Minimal byte-pair encoding tokenizer: raw UTF-8 bytes plus learned adjacent-token merges. */
public final class BpeTokenizer implements TextTokenizer {
    public record Merge(int left, int right) {}
    private final List<Merge> merges;
    private final Map<Long, Integer> ranks = new HashMap<>();

    public BpeTokenizer(List<Merge> merges) {
        this.merges = List.copyOf(merges);
        for (int i = 0; i < merges.size(); i++) {
            Merge merge = merges.get(i);
            int id = ByteTokenizer.VOCABULARY + i;
            if (merge.left() < 0 || merge.right() < 0 || merge.left() >= id || merge.right() >= id)
                throw new IllegalArgumentException("BPE merge refers to an unavailable token");
            if (ranks.put(pair(merge.left(), merge.right()), id) != null)
                throw new IllegalArgumentException("Duplicate BPE merge");
        }
    }

    public static BpeTokenizer train(String text, int vocabulary) {
        if (vocabulary < ByteTokenizer.VOCABULARY || vocabulary > 65_536)
            throw new IllegalArgumentException("BPE vocabulary must be between 260 and 65536");
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        if (bytes.length < 2) throw new IllegalArgumentException("BPE training text is too short");
        List<Integer> tokens = new ArrayList<>(bytes.length);
        for (byte value : bytes) tokens.add(value & 255);
        List<Merge> merges = new ArrayList<>();
        while (ByteTokenizer.VOCABULARY + merges.size() < vocabulary) {
            Map<Long, Integer> counts = new HashMap<>();
            for (int i = 0; i + 1 < tokens.size(); i++)
                counts.merge(pair(tokens.get(i), tokens.get(i + 1)), 1, Integer::sum);
            long best = 0; int bestCount = 1;
            for (var entry : counts.entrySet()) {
                if (entry.getValue() > bestCount || entry.getValue() == bestCount && entry.getKey() < best) {
                    best = entry.getKey(); bestCount = entry.getValue();
                }
            }
            if (bestCount < 2) break;
            int left = (int) (best >>> 32), right = (int) best;
            merges.add(new Merge(left, right));
            int merged = ByteTokenizer.VOCABULARY + merges.size() - 1;
            List<Integer> replaced = new ArrayList<>(tokens.size());
            for (int i = 0; i < tokens.size();) {
                if (i + 1 < tokens.size() && tokens.get(i) == left && tokens.get(i + 1) == right) {
                    replaced.add(merged); i += 2;
                } else replaced.add(tokens.get(i++));
            }
            tokens = replaced;
        }
        return new BpeTokenizer(merges);
    }

    static BpeTokenizer read(DataInput in) throws IOException {
        int count = in.readInt();
        if (count < 0 || count > 65_536 - ByteTokenizer.VOCABULARY)
            throw new IOException("Invalid BPE merge count");
        List<Merge> merges = new ArrayList<>(count);
        for (int i = 0; i < count; i++) merges.add(new Merge(in.readInt(), in.readInt()));
        try { return new BpeTokenizer(merges); }
        catch (IllegalArgumentException e) { throw new IOException("Invalid BPE merges", e); }
    }

    @Override public int vocabulary() { return ByteTokenizer.VOCABULARY + merges.size(); }
    @Override public String kind() { return "bpe"; }
    public List<Merge> merges() { return merges; }

    @Override public int[] encode(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        List<Integer> tokens = new ArrayList<>(bytes.length);
        for (byte value : bytes) tokens.add(value & 255);
        for (int rank = 0; rank < merges.size(); rank++) {
            Merge merge = merges.get(rank); int id = ByteTokenizer.VOCABULARY + rank;
            List<Integer> replaced = new ArrayList<>(tokens.size());
            for (int i = 0; i < tokens.size();) {
                if (i + 1 < tokens.size() && tokens.get(i) == merge.left() && tokens.get(i + 1) == merge.right()) {
                    replaced.add(id); i += 2;
                } else replaced.add(tokens.get(i++));
            }
            tokens = replaced;
        }
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override public String decode(int[] ids) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (int id : ids) expand(id, bytes);
        return bytes.toString(StandardCharsets.UTF_8);
    }

    private void expand(int id, ByteArrayOutputStream bytes) {
        if (id < 0 || id >= vocabulary()) throw new IllegalArgumentException("Invalid token");
        if (id < 256) { bytes.write(id); return; }
        if (id < ByteTokenizer.VOCABULARY) return;
        Merge merge = merges.get(id - ByteTokenizer.VOCABULARY);
        expand(merge.left(), bytes); expand(merge.right(), bytes);
    }

    @Override public int[] prompt(String text) {
        int[] raw = encode(text), result = new int[raw.length + 3];
        result[0] = ByteTokenizer.BOS; result[1] = ByteTokenizer.USER;
        System.arraycopy(raw, 0, result, 2, raw.length);
        result[result.length - 1] = ByteTokenizer.ASSISTANT;
        return result;
    }

    @Override public void write(DataOutput out) throws IOException {
        out.writeUTF(kind()); out.writeInt(merges.size());
        for (Merge merge : merges) { out.writeInt(merge.left()); out.writeInt(merge.right()); }
    }

    private static long pair(int left, int right) { return ((long) left << 32) | (right & 0xffffffffL); }
}
