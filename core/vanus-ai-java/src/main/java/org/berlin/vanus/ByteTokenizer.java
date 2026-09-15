package org.berlin.vanus;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Vocabulary-free tokenizer that maps text directly to raw UTF-8 byte
 * values (IDs 0..255). Four extra IDs (256..259) are reserved for
 * structural/control tokens so they can never collide with actual text.
 */
public final class ByteTokenizer {
    // BOS/EOS mark sequence boundaries; USER/ASSISTANT delimit chat turns; VOCABULARY is the total token count.
    public static final int BOS = 256, EOS = 257, USER = 258, ASSISTANT = 259, VOCABULARY = 260;

    /** Converts text to its raw UTF-8 byte values, one token ID per byte. */
    public int[] encode(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[bytes.length];
        for (int i = 0; i < ids.length; i++) ids[i] = bytes[i] & 255;
        return ids;
    }
    /** Reassembles UTF-8 text from byte token IDs, silently dropping any control tokens. */
    public String decode(int[] ids) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (int id : ids) {
            if (id < 0 || id >= VOCABULARY) throw new IllegalArgumentException("Invalid token");
            if (id < 256) bytes.write(id);
        }
        return bytes.toString(StandardCharsets.UTF_8);
    }
    /** Wraps user text as a chat-style prompt: BOS, USER, the encoded text, then ASSISTANT to cue generation. */
    public int[] prompt(String text) {
        int[] raw = encode(text), ids = new int[raw.length + 3];
        ids[0] = BOS; ids[1] = USER;
        System.arraycopy(raw, 0, ids, 2, raw.length);
        ids[ids.length - 1] = ASSISTANT;
        return ids;
    }
}
