package org.berlin.vanus;

import java.io.DataOutput;
import java.io.IOException;

/** Tokenizer stored with a checkpoint so inference uses the same vocabulary as training. */
public interface TextTokenizer {
    int vocabulary();
    int[] encode(String text);
    String decode(int[] ids);
    int[] prompt(String text);
    String kind();
    void write(DataOutput out) throws IOException;
}
