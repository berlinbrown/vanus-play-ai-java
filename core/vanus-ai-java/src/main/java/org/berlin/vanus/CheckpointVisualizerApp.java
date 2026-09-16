package org.berlin.vanus;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Swing inspector for real checkpoint tensors and differences between
 * checkpoints.
 */
public final class CheckpointVisualizerApp {
    private CheckpointVisualizerApp() {
    }

    public static void launch(Transformer current, String currentPath,
            Transformer reference, String referencePath) {
        validateComparable(current, reference);
        SwingUtilities.invokeLater(() -> buildAndShow(current, currentPath, reference, referencePath));
    }

    static void validateComparable(Transformer current, Transformer reference) {
        if (reference == null)
            return;
        if (!current.config.equals(reference.config))
            throw new IllegalArgumentException("Checkpoint model configurations differ");
        var a = current.namedParameters();
        var b = reference.namedParameters();
        if (!a.keySet().equals(b.keySet()))
            throw new IllegalArgumentException("Checkpoint parameter names differ");
        for (String name : a.keySet()) {
            Tensor x = a.get(name), y = b.get(name);
            if (x.rows != y.rows || x.cols != y.cols)
                throw new IllegalArgumentException("Checkpoint tensor shapes differ: " + name);
        }
    }

    record WeightStats(int count, double mean, double rms, double min, double max,
            double deltaRms, double deltaMax, double cosine, int changed) {
    }

    static WeightStats statistics(Tensor current, Tensor reference) {
        if (reference != null && (current.rows != reference.rows || current.cols != reference.cols))
            throw new IllegalArgumentException("Tensor shapes differ");
        double sum = 0, squares = 0, min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        double deltaSquares = 0, deltaMax = 0, dot = 0, aSquares = 0, bSquares = 0;
        int changed = 0;
        for (int i = 0; i < current.data.length; i++) {
            double value = current.data[i];
            sum += value;
            squares += value * value;
            min = Math.min(min, value);
            max = Math.max(max, value);
            if (reference != null) {
                double other = reference.data[i], delta = value - other;
                deltaSquares += delta * delta;
                deltaMax = Math.max(deltaMax, Math.abs(delta));
                if (Float.floatToIntBits(current.data[i]) != Float.floatToIntBits(reference.data[i]))
                    changed++;
                dot += value * other;
                aSquares += value * value;
                bSquares += other * other;
            }
        }
        double cosine = reference == null ? Double.NaN
                : dot / (Math.sqrt(aSquares * bSquares) + 1e-30);
        return new WeightStats(current.data.length, sum / current.data.length,
                Math.sqrt(squares / current.data.length), min, max,
                reference == null ? Double.NaN : Math.sqrt(deltaSquares / current.data.length),
                reference == null ? Double.NaN : deltaMax, cosine, changed);
    }

    private static void buildAndShow(Transformer current, String currentPath,
            Transformer reference, String referencePath) {
        JFrame frame = new JFrame(reference == null ? "Vanus Checkpoint Weights"
                : "Vanus Checkpoint Comparison");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setLayout(new BorderLayout(8, 8));

        JTextArea header = new JTextArea(headerText(current, currentPath, referencePath));
        header.setEditable(false);
        header.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        header.setBorder(BorderFactory.createTitledBorder("Checkpoint"));
        frame.add(header, BorderLayout.NORTH);

        List<String> names = new ArrayList<>(current.namedParameters().keySet());
        JComboBox<String> tensorChoice = new JComboBox<>(names.toArray(String[]::new));
        JComboBox<String> viewChoice = new JComboBox<>(reference == null
                ? new String[] { "Current weights" }
                : new String[] { "Current weights", "Difference: current − reference" });
        JTextArea selectedStats = new JTextArea(4, 50);
        selectedStats.setEditable(false);
        selectedStats.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));

        WeightConnectionsPanel connections = new WeightConnectionsPanel();
        Runnable select = () -> {
            String name = (String) tensorChoice.getSelectedItem();
            Tensor a = current.namedParameters().get(name);
            Tensor b = reference == null ? null : reference.namedParameters().get(name);
            boolean difference = reference != null && viewChoice.getSelectedIndex() == 1;
            connections.setTensor(name, a, b, difference);
            selectedStats.setText(statsText(name, a, b));
        };
        tensorChoice.addActionListener(e -> select.run());
        viewChoice.addActionListener(e -> select.run());

        JPanel controls = new JPanel(new BorderLayout(8, 8));
        JPanel choices = new JPanel(new GridLayout(1, 2, 8, 0));
        choices.add(labeled("Tensor", tensorChoice));
        choices.add(labeled("View", viewChoice));
        controls.add(choices, BorderLayout.NORTH);
        controls.add(selectedStats, BorderLayout.CENTER);

        JTable table = tensorTable(current, reference);
        JScrollPane tableScroll = new JScrollPane(table);
        tableScroll.setBorder(BorderFactory.createTitledBorder("All tensor statistics (computed from every weight)"));

        JPanel visual = new JPanel(new BorderLayout(4, 4));
        visual.setBorder(BorderFactory.createTitledBorder("Sampled real weights"));
        visual.add(controls, BorderLayout.NORTH);
        visual.add(connections, BorderLayout.CENTER);

        JSplitPane split = new JSplitPane(JSplitPane.VERTICAL_SPLIT, visual, tableScroll);
        split.setResizeWeight(0.68);
        frame.add(split, BorderLayout.CENTER);
        frame.setSize(1250, 900);
        frame.setLocationRelativeTo(null);
        select.run();
        frame.setVisible(true);
    }

    private static String headerText(Transformer model, String currentPath, String referencePath) {
        StringBuilder text = new StringBuilder();
        text.append("Current:   ").append(currentPath).append('\n');
        if (referencePath != null)
            text.append("Reference: ").append(referencePath).append('\n');
        text.append(String.format("Parameters: %,d   width=%d hidden=%d layers=%d heads=%d context=%d%n",
                model.config.parameterCount(), model.config.width(), model.config.hidden(),
                model.config.layers(), model.config.heads(), model.config.context()));
        text.append("Orange = positive; blue = negative. Stronger/thicker = larger magnitude. ")
                .append("The diagram samples at most 24×24 connections; statistics use all weights.");
        return text.toString();
    }

    private static JPanel labeled(String title, JComponent component) {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder(title));
        panel.add(component);
        return panel;
    }

    private static String statsText(String name, Tensor current, Tensor reference) {
        WeightStats s = statistics(current, reference);
        String base = String.format("%s  shape=[%d,%d]  count=%,d%nmean=%+.6f  RMS=%.6f  min=%+.6f  max=%+.6f",
                name, current.rows, current.cols, s.count(), s.mean(), s.rms(), s.min(), s.max());
        if (reference == null)
            return base + "\nSingle-checkpoint view; provide a second checkpoint to visualize changes.";
        return base + String.format("%ndelta RMS=%.6f  max |delta|=%.6f  changed=%,d/%,d  cosine=%.6f",
                s.deltaRms(), s.deltaMax(), s.changed(), s.count(), s.cosine());
    }

    private static JTable tensorTable(Transformer current, Transformer reference) {
        String[] columns = reference == null
                ? new String[] { "Tensor", "Shape", "Count", "Mean", "RMS", "Min", "Max" }
                : new String[] { "Tensor", "Shape", "Count", "Mean", "RMS", "Delta RMS", "Max |delta|", "Changed",
                        "Cosine" };
        DefaultTableModel model = new DefaultTableModel(columns, 0) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
            }
        };
        for (Map.Entry<String, Tensor> entry : current.namedParameters().entrySet()) {
            String name = entry.getKey();
            Tensor tensor = entry.getValue();
            Tensor baseline = reference == null ? null : reference.namedParameters().get(name);
            WeightStats s = statistics(tensor, baseline);
            if (reference == null)
                model.addRow(new Object[] { name, tensor.shape(), s.count(), f(s.mean()), f(s.rms()), f(s.min()),
                        f(s.max()) });
            else
                model.addRow(new Object[] { name, tensor.shape(), s.count(), f(s.mean()), f(s.rms()),
                        f(s.deltaRms()), f(s.deltaMax()), s.changed(), f(s.cosine()) });
        }
        JTable table = new JTable(model);
        table.setAutoCreateRowSorter(true);
        table.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        return table;
    }

    private static String f(double value) {
        return String.format("%.6f", value);
    }
}

/**
 * Draws a bounded, evenly-spaced sample of actual matrix weights as
 * connections.
 */
final class WeightConnectionsPanel extends JPanel {
    private String name = "";
    private Tensor current, reference;
    private boolean difference;

    WeightConnectionsPanel() {
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(1100, 430));
    }

    void setTensor(String name, Tensor current, Tensor reference, boolean difference) {
        this.name = name;
        this.current = current;
        this.reference = reference;
        this.difference = difference;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        super.paintComponent(graphics);
        if (current == null)
            return;
        Graphics2D g = (Graphics2D) graphics.create();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        int width = getWidth(), height = getHeight();
        g.setColor(Color.DARK_GRAY);
        g.drawString((difference ? "Weight change: " : "Weight value: ") + name, 20, 24);
        if (current.rows == 1)
            drawVector(g, width, height);
        else
            drawMatrix(g, width, height);
        g.dispose();
    }

    private void drawMatrix(Graphics2D g, int width, int height) {
        int countLeft = Math.min(24, current.rows), countRight = Math.min(24, current.cols);
        int leftX = 85, rightX = width - 85, top = 55, bottom = height - 45;
        double max = maximumSampleMagnitude(countLeft, countRight);
        for (int i = 0; i < countLeft; i++) {
            int row = sampleIndex(i, countLeft, current.rows), y1 = position(i, countLeft, top, bottom);
            for (int j = 0; j < countRight; j++) {
                int col = sampleIndex(j, countRight, current.cols), y2 = position(j, countRight, top, bottom);
                double value = value(row, col), strength = Math.min(1, Math.abs(value) / (max + 1e-30));
                g.setColor(color(value, strength));
                g.setStroke(new BasicStroke((float) (0.35 + 2.4 * strength)));
                g.drawLine(leftX, y1, rightX, y2);
            }
        }
        g.setStroke(new BasicStroke(1));
        for (int i = 0; i < countLeft; i++)
            node(g, leftX, position(i, countLeft, top, bottom));
        for (int j = 0; j < countRight; j++)
            node(g, rightX, position(j, countRight, top, bottom));
        g.setColor(Color.DARK_GRAY);
        g.drawString("rows / source dimension (" + current.rows + ", sampled " + countLeft + ")", 20, height - 12);
        String right = "columns / destination dimension (" + current.cols + ", sampled " + countRight + ")";
        g.drawString(right, Math.max(20, width - g.getFontMetrics().stringWidth(right) - 20), height - 12);
    }

    private void drawVector(Graphics2D g, int width, int height) {
        int count = Math.min(96, current.cols), left = 35, right = width - 35;
        double max = 0;
        for (int i = 0; i < count; i++)
            max = Math.max(max, Math.abs(value(0, sampleIndex(i, count, current.cols))));
        int center = height / 2;
        g.setColor(Color.GRAY);
        g.drawLine(left, center, right, center);
        for (int i = 0; i < count; i++) {
            int col = sampleIndex(i, count, current.cols);
            double value = value(0, col), strength = Math.min(1, Math.abs(value) / (max + 1e-30));
            int x = position(i, count, left, right), bar = (int) Math.round(strength * (height / 2.0 - 65));
            g.setColor(color(value, strength));
            g.fillRect(x - 2, value >= 0 ? center - bar : center, 4, bar);
        }
        g.setColor(Color.DARK_GRAY);
        g.drawString("RMSNorm gain vector: " + current.cols + " values, sampled " + count, 20, height - 12);
    }

    private double maximumSampleMagnitude(int rows, int cols) {
        double max = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                max = Math.max(max,
                        Math.abs(value(sampleIndex(i, rows, current.rows), sampleIndex(j, cols, current.cols))));
        return max;
    }

    private double value(int row, int col) {
        int index = row * current.cols + col;
        return difference ? current.data[index] - reference.data[index] : current.data[index];
    }

    private static int sampleIndex(int index, int count, int total) {
        return count == 1 ? 0 : (int) Math.round(index * (total - 1.0) / (count - 1));
    }

    private static int position(int index, int count, int start, int end) {
        return count == 1 ? (start + end) / 2 : start + index * (end - start) / (count - 1);
    }

    private static Color color(double value, double strength) {
        int alpha = (int) Math.round(25 + 205 * strength);
        return value >= 0 ? new Color(220, 95, 25, alpha) : new Color(35, 105, 210, alpha);
    }

    private static void node(Graphics2D g, int x, int y) {
        g.setColor(new Color(65, 65, 75));
        g.fillOval(x - 4, y - 4, 8, 8);
    }
}
