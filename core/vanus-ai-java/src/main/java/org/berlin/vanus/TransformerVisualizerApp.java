package org.berlin.vanus;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import java.awt.*;
import java.util.Set;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/** Swing window: model/parameter readout, an animated forward-pass diagram, and a chat box. */
public final class TransformerVisualizerApp {
    private static final Logger LOGGER = LogManager.getLogger(TransformerVisualizerApp.class);

    private TransformerVisualizerApp() {}

    public static void launch(Transformer model, String datasetName, int datasetSize, String checkpointPath, Set<String> knownPrompts) {
        SwingUtilities.invokeLater(() -> buildAndShow(model, datasetName, datasetSize, checkpointPath, knownPrompts));
    }

    private static void buildAndShow(Transformer model, String datasetName, int datasetSize, String checkpointPath, Set<String> knownPrompts) {
        LOGGER.info("[swing] started dataset={} pairs={} checkpoint={} parameters={} tokenizer={} context={}",
                logText(datasetName), datasetSize, logText(checkpointPath), model.config.parameterCount(),
                model.tokenizer().kind(), model.config.context());
        JFrame frame = new JFrame("Vanus Transformer Visualizer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout(8, 8));

        frame.add(parametersPanel(model.config, model.tokenizer(), datasetName, datasetSize, checkpointPath), BorderLayout.NORTH);

        NetworkVisualizerPanel visualizer = new NetworkVisualizerPanel(model.config);
        JTextArea trace = new JTextArea(12, 80);
        trace.setEditable(false);
        trace.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        trace.setText("Enter a prompt to inspect actual next-token decisions.\n"
                + "Each step runs embedding → causal attention → SwiGLU → logits → next token.\n"
                + "Probabilities cover text tokens plus EOS, before sampling filters.\n"
                + "The network lines are schematic, not measured attention weights.\n");
        JScrollPane traceScroll = new JScrollPane(trace);
        traceScroll.setBorder(BorderFactory.createTitledBorder("Live decoding: top 5 next-token candidates"));
        JSplitPane split = new JSplitPane(JSplitPane.VERTICAL_SPLIT, new JScrollPane(visualizer), traceScroll);
        split.setResizeWeight(0.55);
        frame.add(split, BorderLayout.CENTER);

        frame.add(chatPanel(model, visualizer, knownPrompts, trace), BorderLayout.SOUTH);

        frame.setSize(1180, 880);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private static JComponent parametersPanel(ModelConfig config, TextTokenizer tokenizer, String datasetName, int datasetSize, String checkpointPath) {
        long embeddingParams = (long) config.vocabulary() * config.width();
        long perBlockParams = 4L * config.width() * config.width() + 3L * config.width() * config.hidden() + 2L * config.width();
        long blockParams = (long) config.layers() * perBlockParams;
        long normParams = config.width();
        long total = embeddingParams + blockParams + normParams;

        JTextArea area = new JTextArea(String.format(
                "Dataset: %s (%,d training pairs)   Checkpoint: %s%n" +
                "Tokenizer=%s  Vocabulary=%d  Width=%d  Hidden=%d  Layers=%d  Heads=%d  Context=%d tokens%n" +
                "Parameters: embedding=%,d  block=%,d x %d layers=%,d  norm=%,d  TOTAL=%,d",
                datasetName, datasetSize, checkpointPath,
                tokenizer.kind(), config.vocabulary(), config.width(), config.hidden(), config.layers(), config.heads(), config.context(),
                embeddingParams, perBlockParams, config.layers(), blockParams, normParams, total));
        area.setEditable(false);
        area.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        area.setBorder(BorderFactory.createTitledBorder("Model parameters"));
        return area;
    }

    private static String tokenLabel(int id) {
        if (id == ByteTokenizer.EOS) return "<EOS>";
        if (id == 32) return "<space>";
        if (id >= 33 && id <= 126) return "'" + (char) id + "'";
        if (id >= ByteTokenizer.VOCABULARY) return "<tok:" + id + ">";
        return String.format("0x%02X", id);
    }

    /** Keep console events on one readable line even if generated text contains controls. */
    private static String logText(String text) {
        return text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n");
    }

    /** Randomly select words, preserving their source order and UTF-8 context budget. */
    static String selfTalkPrompt(String reply, int maxBytes, java.util.Random random) {
        if (maxBytes < 5) throw new IllegalArgumentException("Self-talk needs room for hello");
        var matcher = java.util.regex.Pattern.compile("[\\p{L}\\p{N}]+(?:['’][\\p{L}\\p{N}]+)*").matcher(reply);
        java.util.List<String> words = new java.util.ArrayList<>();
        while (matcher.find()) {
            String word = matcher.group();
            if (word.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= maxBytes) words.add(word);
        }
        if (words.isEmpty()) return "hello";
        int forced = random.nextInt(words.size()), used = 0;
        StringBuilder prompt = new StringBuilder();
        for (int i = 0; i < words.size(); i++) {
            if (i != forced && !random.nextBoolean()) continue;
            String word = words.get(i);
            int bytes = word.getBytes(java.nio.charset.StandardCharsets.UTF_8).length + (used == 0 ? 0 : 1);
            if (used + bytes > maxBytes) continue;
            if (used > 0) prompt.append(' ');
            prompt.append(word); used += bytes;
        }
        return prompt.isEmpty() ? words.get(forced) : prompt.toString();
    }

    private static JComponent chatPanel(Transformer model, NetworkVisualizerPanel visualizer, Set<String> knownPrompts, JTextArea trace) {
        JTextArea transcript = new JTextArea(8, 70);
        transcript.setEditable(false);
        transcript.setLineWrap(true);
        transcript.setWrapStyleWord(true);
        JScrollPane transcriptScroll = new JScrollPane(transcript);

        JTextField input = new JTextField();
        JButton send = new JButton("Send");

        JButton selfTalk = new JButton("Talk to itself");
        javax.swing.JLabel status = new javax.swing.JLabel("Self-talk off");
        class Conversation {
            static final long RESET_INTERVAL_NANOS = java.util.concurrent.TimeUnit.SECONDS.toNanos(40);
            boolean automatic, busy, closed;
            boolean resetPrompt;
            int turn;
            String next = "hello";
            long nextResetAtNanos;
            final java.util.Random random = new java.util.Random();
            final javax.swing.Timer timer = new javax.swing.Timer(2000, e -> {
                if (!busy && automatic && !closed) {
                    long now = System.nanoTime();
                    if (now >= nextResetAtNanos) {
                        next = random.nextBoolean() ? "hello" : "goodbye";
                        resetPrompt = true;
                        nextResetAtNanos = now + RESET_INTERVAL_NANOS;
                        LOGGER.info("[swing] self-talk reset prompt={} nextReset=40s", next);
                    }
                    input.setText(next);
                    submit();
                }
                else if (automatic) status.setText("Still generating; waiting for the next 2-second tick");
            });
            void controls() {
                input.setEnabled(!busy && !automatic);
                send.setEnabled(!busy && !automatic);
                selfTalk.setEnabled(automatic || !busy);
                selfTalk.setText(automatic ? "Stop self-talk" : "Talk to itself");
            }
            void toggle() {
                if (automatic) {
                    automatic = false; timer.stop();
                    status.setText(busy ? "Stopping after current reply" : "Self-talk off");
                    LOGGER.info("[swing] self-talk stopped turns={}", turn);
                } else if (!busy && !closed) {
                    automatic = true; turn = 0; next = "hello"; resetPrompt = false;
                    nextResetAtNanos = System.nanoTime() + RESET_INTERVAL_NANOS;
                    LOGGER.info("[swing] self-talk started interval=2s resetInterval=40s initialPrompt=hello");
                    input.setText(next); timer.start(); submit();
                }
                controls();
            }
            void submit() {
            if (busy || closed) return;
            String prompt = input.getText().trim();
            if (prompt.isEmpty()) return;
            busy = true; controls();
            boolean automaticTurn = automatic;
            boolean scheduledReset = automaticTurn && resetPrompt;
            resetPrompt = false;
            long generationStarted = System.nanoTime();
            String stamp = java.time.LocalTime.now().withNano(0).toString();
            transcript.append((automaticTurn ? "Self-talk #" + (++turn) + " [" + stamp + "]" : "You") + ": " + prompt + "\n");
            LOGGER.info("[swing] input mode={} turn={} reset={} known={} chars={} text={}",
                    automaticTurn ? "self-talk" : "user", automaticTurn ? turn : 0, scheduledReset,
                    knownPrompts.contains(prompt), prompt.length(), logText(prompt));
            if (automaticTurn) status.setText("Generating self-talk reply " + turn);
            input.setText("");
            trace.setText("Prompt token IDs: " + java.util.Arrays.toString(model.tokenizer().prompt(prompt)) + "\n");
            visualizer.startActivity();
            new SwingWorker<String, Transformer.GenerationStep>() {
                // Greedy decoding for deterministic, reproducible recall matching the `chat` CLI command.
                @Override protected String doInBackground() { return model.generate(prompt, 150, 0, 1, 42, step -> publish(step)); }
                @Override protected void process(java.util.List<Transformer.GenerationStep> steps) {
                    if (closed) return;
                    for (Transformer.GenerationStep step : steps) {
                        StringBuilder line = new StringBuilder(String.format("%3d | context %3d/%d | chose %-10s | ",
                                step.step(), step.contextUsed(), model.config.context(), tokenLabel(step.token())));
                        for (Transformer.Candidate candidate : step.candidates())
                            line.append(String.format("%s %.1f%%  ", tokenLabel(candidate.token()), 100 * candidate.probability()));
                        if (!step.stopReason().isEmpty()) line.append(" STOP: ").append(step.stopReason());
                        trace.append(line + "\n");
                    }
                    visualizer.showStep(steps.get(steps.size() - 1));
                    trace.setCaretPosition(trace.getDocument().getLength());
                }
                @Override protected void done() {
                    // Training membership is useful context, not a guarantee of correct recall.
                    String note = knownPrompts.contains(prompt)
                            ? "[Training prompt: recall depends on what the model learned.]"
                            : "[Unseen prompt: this tiny model may generalize poorly.]";
                    if (closed) return;
                    try {
                        String reply = get();
                        transcript.append("Vanus: " + reply + "\n" + note + "\n\n");
                        long elapsedMillis = java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - generationStarted);
                        LOGGER.info("[swing] output mode={} turn={} elapsedMs={} chars={} text={}",
                                automaticTurn ? "self-talk" : "user", automaticTurn ? turn : 0,
                                elapsedMillis, reply.length(), logText(reply));
                        if (automaticTurn && automatic) {
                            next = selfTalkPrompt(reply, model.config.context() - 4, random);
                            input.setText(next);
                            status.setText("Next 2-second tick: " + next);
                            LOGGER.info("[swing] self-talk queued next={}", logText(next));
                        }
                    }
                    catch (Exception e) {
                        automatic = false; timer.stop(); status.setText("Self-talk stopped: generation error");
                        String message = e.getCause() == null ? e.getMessage() : e.getCause().getMessage();
                        transcript.append("Vanus: [error: " + message + "]\n\n");
                        LOGGER.error("[swing] generation error mode={} turn={} message={}",
                                automaticTurn ? "self-talk" : "user", automaticTurn ? turn : 0,
                                logText(String.valueOf(message)), e);
                    }
                    visualizer.stopActivity();
                    busy = false; controls();
                    if (!automatic) { status.setText("Self-talk off"); input.requestFocusInWindow(); }
                    // Bound memory while the autonomous loop runs for long periods.
                    if (transcript.getDocument().getLength() > 60000)
                        transcript.replaceRange("", 0, transcript.getDocument().getLength() - 40000);
                    transcript.setCaretPosition(transcript.getDocument().getLength());
                }
            }.execute();
            }
        }
        Conversation conversation = new Conversation();
        send.addActionListener(e -> conversation.submit());
        input.addActionListener(e -> conversation.submit());
        selfTalk.addActionListener(e -> conversation.toggle());

        JPanel inputRow = new JPanel(new BorderLayout(4, 4));
        inputRow.add(input, BorderLayout.CENTER);
        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 4, 0));
        buttons.add(send); buttons.add(selfTalk);
        inputRow.add(buttons, BorderLayout.EAST);

        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.setBorder(BorderFactory.createTitledBorder("Chat — deterministic greedy decoding"));
        JComboBox<String> examples = new JComboBox<>(knownPrompts.stream().sorted().toArray(String[]::new));
        examples.setSelectedIndex(-1);
        examples.setBorder(BorderFactory.createTitledBorder("Try a training prompt, then reword it"));
        examples.addActionListener(e -> {
            if (examples.getSelectedItem() != null && input.isEnabled()) input.setText((String) examples.getSelectedItem());
        });
        panel.add(examples, BorderLayout.NORTH);
        panel.add(transcriptScroll, BorderLayout.CENTER);
        JPanel bottom = new JPanel(new BorderLayout(4, 4));
        bottom.add(status, BorderLayout.NORTH);
        bottom.add(inputRow, BorderLayout.SOUTH);
        panel.add(bottom, BorderLayout.SOUTH);
        panel.addHierarchyListener(e -> {
            if ((e.getChangeFlags() & java.awt.event.HierarchyEvent.DISPLAYABILITY_CHANGED) != 0 && !panel.isDisplayable()) {
                conversation.closed = true;
                conversation.automatic = false;
                conversation.timer.stop();
                LOGGER.info("[swing] closed selfTalkTurns={}", conversation.turn);
            }
        });
        return panel;
    }
}

/** Animated schematic of the forward pass: embedding, each decoder block, final norm/output. */
final class NetworkVisualizerPanel extends JPanel {
    private final ModelConfig config;
    private final int layers;
    private String progress = "Ready — connections show sampled dimensions, not individual parameters.";
    private int activeStage = -1;
    private boolean running;

    NetworkVisualizerPanel(ModelConfig config) {
        this.config = config;
        this.layers = config.layers();
        setPreferredSize(new Dimension(Math.max(1080, (layers + 2) * 260), 320));
        setBackground(Color.WHITE);

    }

    void startActivity() { running = true; activeStage = -1; progress = "Computing next-token predictions…"; repaint(); }
    void showStep(Transformer.GenerationStep step) {
        activeStage = layers + 1;
        progress = "Token step " + step.step() + " | " + step.contextUsed() + "/" + config.context()
                + " context positions processed | " + (step.stopReason().isEmpty() ? "decoding" : step.stopReason());
        repaint();
    }
    void stopActivity() { running = false; activeStage = -1; repaint(); }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2.setColor(Color.DARK_GRAY);
        g2.drawString(progress, 20, 24);
        g2.drawString("One full forward pass per generated token. Attention can only use earlier/current positions.", 20, 44);
        int stages = layers + 2;
        String[] labels = new String[stages];
        labels[0] = "Token IDs → Embedding\n" + config.width() + " values / position";
        for (int i = 1; i <= layers; i++) labels[i] = "Block " + i + "\nRMSNorm → Q / K / V\nRoPE → causal attention\n" + config.heads() + " heads → residual add\nRMSNorm → SwiGLU\nhidden " + config.hidden() + " → residual add";
        labels[stages - 1] = "Final RMSNorm\nTied output projection\n" + config.vocabulary() + " logits → next token";
        // Node counts per stage stand in for real dimensions (attention heads inside blocks,
        // a capped sample of embedding/output width elsewhere) so the fan-out is not misleading noise.
        int[] nodeCounts = new int[stages];
        nodeCounts[0] = Math.min(16, config.width());
        for (int i = 1; i <= layers; i++) nodeCounts[i] = Math.min(16, config.width());
        nodeCounts[stages - 1] = Math.min(16, config.width());

        int boxW = 180, boxH = 160;
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
