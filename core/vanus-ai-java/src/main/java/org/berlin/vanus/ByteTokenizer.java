package org.berlin.vanus;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/** Byte IDs 0..255; special IDs never collide with user text. */
public final class ByteTokenizer {
    public static final int BOS = 256, EOS = 257, USER = 258, ASSISTANT = 259, VOCABULARY = 260;
    public int[] encode(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[bytes.length];
        for (int i = 0; i < ids.length; i++) ids[i] = bytes[i] & 255;
        return ids;
    }
    public String decode(int[] ids) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (int id : ids) {
            if (id < 0 || id >= VOCABULARY) throw new IllegalArgumentException("Invalid token");
            if (id < 256) bytes.write(id);
        }
        return bytes.toString(StandardCharsets.UTF_8);
    }
    public int[] prompt(String text) {
        int[] raw = encode(text), ids = new int[raw.length + 3];
        ids[0] = BOS; ids[1] = USER;
        System.arraycopy(raw, 0, ids, 2, raw.length);
        ids[ids.length - 1] = ASSISTANT;
        return ids;
    }
}
