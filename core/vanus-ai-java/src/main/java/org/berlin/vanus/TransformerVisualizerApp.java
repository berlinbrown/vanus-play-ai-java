package org.berlin.vanus;

import javax.swing.*;
import java.awt.*;
import java.util.Set;

/** Swing window: model/parameter readout, an animated forward-pass diagram, and a chat box. */
public final class TransformerVisualizerApp {
    private TransformerVisualizerApp() {}

    public static void launch(Transformer model, String datasetName, int datasetSize, String checkpointPath, Set<String> knownPrompts) {
        SwingUtilities.invokeLater(() -> buildAndShow(model, datasetName, datasetSize, checkpointPath, knownPrompts));
    }

    private static void buildAndShow(Transformer model, String datasetName, int datasetSize, String checkpointPath, Set<String> knownPrompts) {
        JFrame frame = new JFrame("Vanus Transformer Visualizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout(8, 8));

        frame.add(parametersPanel(model.config, datasetName, datasetSize, checkpointPath), BorderLayout.NORTH);

        NetworkVisualizerPanel visualizer = new NetworkVisualizerPanel(model.config);
        frame.add(visualizer, BorderLayout.CENTER);

        frame.add(chatPanel(model, visualizer, knownPrompts), BorderLayout.SOUTH);

        frame.setSize(920, 640);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private static JComponent parametersPanel(ModelConfig config, String datasetName, int datasetSize, String checkpointPath) {
        long embeddingParams = (long) config.vocabulary() * config.width();
        long perBlockParams = 4L * config.width() * config.width() + 3L * config.width() * config.hidden() + 2L * config.width();
        long blockParams = (long) config.layers() * perBlockParams;
        long normParams = config.width();
        long total = embeddingParams + blockParams + normParams;

        JTextArea area = new JTextArea(String.format(
                "Dataset: %s (%,d training pairs)   Checkpoint: %s%n" +
                "Vocabulary=%d  Width=%d  Hidden=%d  Layers=%d  Heads=%d  Context=%d bytes%n" +
                "Parameters: embedding=%,d  block=%,d x %d layers=%,d  norm=%,d  TOTAL=%,d",
                datasetName, datasetSize, checkpointPath,
                config.vocabulary(), config.width(), config.hidden(), config.layers(), config.heads(), config.context(),
                embeddingParams, perBlockParams, config.layers(), blockParams, normParams, total));
        area.setEditable(false);
        area.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        area.setBorder(BorderFactory.createTitledBorder("Model parameters"));
        return area;
    }

    private static JComponent chatPanel(Transformer model, NetworkVisualizerPanel visualizer, Set<String> knownPrompts) {
        JTextArea transcript = new JTextArea(8, 70);
        transcript.setEditable(false);
        transcript.setLineWrap(true);
        transcript.setWrapStyleWord(true);
        JScrollPane transcriptScroll = new JScrollPane(transcript);

        JTextField input = new JTextField();
        JButton send = new JButton("Send");

        Runnable submit = () -> {
            String prompt = input.getText().trim();
            if (prompt.isEmpty()) return;
            input.setEnabled(false); send.setEnabled(false);
            transcript.append("You: " + prompt + "\n");
            input.setText("");
            visualizer.startActivity();
            new SwingWorker<String, Void>() {
                // Greedy decoding for deterministic, reproducible recall matching the `chat` CLI command.
                @Override protected String doInBackground() { return model.generate(prompt, 150, 0, 1, 42); }
                @Override protected void done() {
                    // This model only memorizes exact training byte sequences; it does not understand or
                    // generalize the question, so recall quality hinges entirely on an exact phrasing match.
                    String note = knownPrompts.contains(prompt)
                            ? "[this exact prompt was in the training data \u2192 greedy decode replays the memorized answer]"
                            : "[this prompt was NOT seen verbatim during training \u2192 the model is guessing byte-by-byte; expect noise]";
                    try { transcript.append("Vanus: " + get() + "\n" + note + "\n\n"); }
                    catch (Exception e) { transcript.append("Vanus: [error: " + e.getMessage() + "]\n\n"); }
                    visualizer.stopActivity();
                    input.setEnabled(true); send.setEnabled(true); input.requestFocusInWindow();
                    transcript.setCaretPosition(transcript.getDocument().getLength());
                }
            }.execute();
        };
        send.addActionListener(e -> submit.run());
        input.addActionListener(e -> submit.run());

        JPanel inputRow = new JPanel(new BorderLayout(4, 4));
        inputRow.add(input, BorderLayout.CENTER);
        inputRow.add(send, BorderLayout.EAST);

        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.setBorder(BorderFactory.createTitledBorder("Chat (Pride and Prejudice model)"));
        panel.add(transcriptScroll, BorderLayout.CENTER);
        panel.add(inputRow, BorderLayout.SOUTH);
        return panel;
    }
}

/** Animated schematic of the forward pass: embedding, each decoder block, final norm/output. */
final class NetworkVisualizerPanel extends JPanel {
    private final ModelConfig config;
    private final int layers;
    private final Timer timer;
    private int activeStage = -1;
    private boolean running;

    NetworkVisualizerPanel(ModelConfig config) {
        this.config = config;
        this.layers = config.layers();
        setPreferredSize(new Dimension(880, 340));
        setBackground(Color.WHITE);
        timer = new Timer(220, e -> { activeStage = (activeStage + 1) % (layers + 2); repaint(); });
    }

    void startActivity() { running = true; activeStage = 0; timer.start(); }
    void stopActivity() { running = false; timer.stop(); activeStage = -1; repaint(); }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int stages = layers + 2;
        String[] labels = new String[stages];
        labels[0] = "Embedding";
        for (int i = 1; i <= layers; i++) labels[i] = "Block " + i + "\nNorm\u2192Attn\u2192Norm\u2192FFN";
        labels[stages - 1] = "Final Norm\n+ Output";
        // Node counts per stage stand in for real dimensions (attention heads inside blocks,
        // a capped sample of embedding/output width elsewhere) so the fan-out is not misleading noise.
        int[] nodeCounts = new int[stages];
        nodeCounts[0] = Math.min(6, config.width());
        for (int i = 1; i <= layers; i++) nodeCounts[i] = Math.min(8, config.heads());
        nodeCounts[stages - 1] = Math.min(6, config.width());

        int boxW = 150, boxH = 56;
        int gap = stages > 1 ? (getWidth() - 40 - boxW) / (stages - 1) : 0;
        int y = getHeight() / 2 - boxH / 2;
        int[] centersX = new int[stages];
        for (int i = 0; i < stages; i++) centersX[i] = 20 + boxW / 2 + i * gap;

        // Fan-out/fan-in lines between each stage's sampled nodes, drawn under the boxes.
        for (int i = 0; i < stages - 1; i++) {
            boolean active = running && (activeStage == i || activeStage == i + 1);
            Point[] left = nodePoints(centersX[i] + boxW / 2, y, boxH, nodeCounts[i]);
            Point[] right = nodePoints(centersX[i + 1] - boxW / 2, y, boxH, nodeCounts[i + 1]);
            g2.setStroke(new BasicStroke(1f));
            for (Point a : left)
                for (Point b : right) {
                    g2.setColor(active ? new Color(220, 90, 40, 160) : new Color(190, 190, 190, 90));
                    g2.drawLine(a.x, a.y, b.x, b.y);
                }
        }
        for (int i = 0; i < stages; i++) {
            boolean active = running && activeStage == i;
            int x = centersX[i] - boxW / 2;
            g2.setColor(active ? new Color(255, 213, 153) : new Color(232, 232, 245));
            g2.fillRoundRect(x, y, boxW, boxH, 14, 14);
            g2.setColor(active ? new Color(200, 90, 20) : Color.DARK_GRAY);
            g2.drawRoundRect(x, y, boxW, boxH, 14, 14);
            g2.setColor(Color.BLACK);
            drawCentered(g2, labels[i], x, y, boxW, boxH);
        }
        // Node dots sit on top of the boxes at each stage's edges, endpoints for the fan lines above.
        for (int i = 0; i < stages; i++) {
            boolean active = running && activeStage == i;
            g2.setColor(active ? new Color(200, 90, 20) : new Color(120, 120, 120));
            for (Point p : nodePoints(centersX[i] - boxW / 2, y, boxH, nodeCounts[i]))
                g2.fillOval(p.x - 3, p.y - 3, 6, 6);
            for (Point p : nodePoints(centersX[i] + boxW / 2, y, boxH, nodeCounts[i]))
                g2.fillOval(p.x - 3, p.y - 3, 6, 6);
        }
    }

    private static Point[] nodePoints(int x, int boxY, int boxH, int count) {
        Point[] points = new Point[count];
        for (int i = 0; i < count; i++)
            points[i] = new Point(x, boxY + (i + 1) * boxH / (count + 1));
        return points;
    }

    private void drawCentered(Graphics2D g2, String text, int x, int y, int w, int h) {
        String[] lines = text.split("\n");
        FontMetrics fm = g2.getFontMetrics();
        int startY = y + (h - lines.length * fm.getHeight()) / 2 + fm.getAscent();
        for (String line : lines) {
            g2.drawString(line, x + (w - fm.stringWidth(line)) / 2, startY);
            startY += fm.getHeight();
        }
    }
}
