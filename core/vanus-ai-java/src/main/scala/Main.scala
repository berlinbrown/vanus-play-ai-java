package org.berlin.vanus

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.Arrays
import org.apache.logging.log4j.LogManager
import scala.jdk.CollectionConverters.*

object Main:
  private val logger = LogManager.getLogger(Main.getClass)
  private val defaultPromptNoisePercent = 15
  // Default demo corpus; filtered per-config below since the tiny model's context is small.
  private val demoData = Path.of("data/pride-and-prejudice.tsv")
  private val demoDictData = Path.of("data/dictionary2.tsv")
  def main(args: Array[String]): Unit =
    requireJava21()
    args.toList match
      case Nil | "info" :: Nil =>
        for (name, config) <- List("tiny" -> ModelConfig.tiny(), "dictionary" -> ModelConfig.dictionary(),
          "200k" -> ModelConfig.vanus200k(), "1m" -> ModelConfig.vanus1m(), "20m" -> ModelConfig.vanus20m()) do
          println(s"$name: ${config.parameterCount()} parameters; $config")
        println("""
Recommended GUI:
  demo2greet                         Load the verified greeting model and open Swing
  demo2greet 2000                    Retrain greetings, evaluate, then open Swing
  demo2daily200k 10000               Train the larger experimental model, then open Swing
  gui <checkpoint> [pairs.tsv]       Open any saved checkpoint without training

All application commands:
  info                               Show model configurations and this help
  greetings [steps]                  Greeting model; train/load and evaluate in terminal
  demo2greet [steps]                 Greeting model; train/load, evaluate, open Swing
  demo2daily [steps]                 Tiny DailyDialog starter model with Swing
  demo2daily200k [steps]             200K DailyDialog starter model with Swing
  demo [steps] [checkpoint]          Always train tiny novel continuation model
  demo2 [steps]                      Tiny novel continuation model with Swing
  demo2dict [steps]                  Dictionary model with Swing
  weights <checkpoint>               Inspect real checkpoint weights in Swing
  weights <current> <reference>      Visualize weight changes between checkpoints
  train <tiny|200k|1m|20m> <pairs.tsv> <steps> <checkpoint> [batch]
                                     Train prompt/reply data from random weights
  pretrain <tiny|200k|1m|20m> <text> <steps> <checkpoint> [batch] [bpe-vocab]
                                     Learn BPE and continuously predict a text file
  continue <checkpoint> <pairs.tsv> <steps> <output> [batch]
                                     Continue with saved weights and AdamW state
  continue-pretrain <checkpoint> <text> <steps> <output> [batch]
                                     Continue raw-text next-token pretraining
  gui <checkpoint> [pairs.tsv]       Open saved weights in Swing; never trains
  server <checkpoint> <pairs.tsv> [server options]
                                     Run persistent headless self-talk and REST API
  chat <checkpoint> <prompt...>      Generate one terminal reply from saved weights
  eval <checkpoint> <pairs.tsv>      Evaluate saved weights against prompt/reply pairs

Arguments:
  Demo [steps] is optional. Supplying it retrains that demo from scratch.
  Batch defaults to 1. Supervised training automatically adds light prompt typo noise to 20% of samples.
  BPE vocabulary defaults to 512 (260 base/control tokens plus merges).
  Without [steps], a demo loads its checkpoint, or trains defaults if it is missing.
  Run application commands as: sbt 'run <command> <arguments>'
  In Swing, use Send, the example dropdown, or Talk to itself (every 2 seconds).
""")
      case command :: rest if Set("greetings", "demo2greet").contains(command) && rest.length <= 1 =>
        val pairs = loadPairs(Path.of("data/conversation.tsv"))
        val checkpoint = Path.of("checkpoints/conversation.vanus")
        val model = if rest.isEmpty && Files.exists(checkpoint) then Transformer.load(checkpoint)
          else train(ModelConfig.tiny(), pairs, rest.headOption.fold(2000)(_.toInt), checkpoint, 4)
        evaluate(model, pairs, "Training recall")
        evaluate(model, loadPairs(Path.of("data/conversation-unseen.tsv")), "Unseen phrasings (not trained)")
        if command == "demo2greet" then
          TransformerVisualizerApp.launch(model, "Everyday greetings", pairs.size, checkpoint.toString, pairs.map(_._1).toSet.asJava)
      case command :: rest if Set("demo2daily", "demo2daily200k").contains(command) && rest.length <= 1 =>
        val larger = command == "demo2daily200k"
        val config = if larger then ModelConfig.vanus200k() else ModelConfig.tiny()
        val pairs = loadPairs(Path.of("data/dailydialog/starter.tsv"))
        val checkpoint = Path.of(if larger then "checkpoints/dailydialog-200k.vanus" else "checkpoints/dailydialog-starter.vanus")
        val model = if rest.isEmpty && Files.exists(checkpoint) then Transformer.load(checkpoint)
          else train(config, pairs, rest.headOption.fold(10000)(_.toInt), checkpoint)
        TransformerVisualizerApp.launch(model, "DailyDialog starter", pairs.size, checkpoint.toString, pairs.map(_._1).toSet.asJava)
      case "eval" :: checkpoint :: input :: Nil =>
        evaluate(Transformer.load(Path.of(checkpoint)), loadPairs(Path.of(input)), input)
      case "weights" :: current :: Nil =>
        CheckpointVisualizerApp.launch(Transformer.load(Path.of(current)), current, null, null)
      case "weights" :: current :: reference :: Nil =>
        CheckpointVisualizerApp.launch(Transformer.load(Path.of(current)), current,
          Transformer.load(Path.of(reference)), reference)
      case "demo" :: rest if rest.length <= 2 =>
        val config = ModelConfig.tiny()
        val pairs = loadPairs(demoData).filter((p, a) => (p + a).getBytes(StandardCharsets.UTF_8).length <= config.context() - 3)
        train(config, pairs, rest.headOption.fold(200)(_.toInt),
          Path.of(rest.drop(1).headOption.getOrElse("checkpoints/greeting.vanus")))
      case "demo2" :: rest if rest.length <= 1 =>
        val config = ModelConfig.tiny()
        val pairs = loadPairs(demoData).filter((p, a) => (p + a).getBytes(StandardCharsets.UTF_8).length <= config.context() - 3)
        val checkpoint = Path.of("checkpoints/demo2.vanus")
        // An explicit step count always (re)trains; a bare `demo2` reuses an existing checkpoint if present.
        val model = if rest.isEmpty && Files.exists(checkpoint) then Transformer.load(checkpoint)
          else train(config, pairs, rest.headOption.fold(200)(_.toInt), checkpoint)
        TransformerVisualizerApp.launch(model, "Pride and Prejudice", pairs.size, checkpoint.toString, pairs.map(_._1).toSet.asJava)
      case "demo2dict" :: rest if rest.length <= 1 =>
        val config = ModelConfig.dictionary()
        val pairs = loadPairs(demoDictData).filter((p, a) => (p + a).getBytes(StandardCharsets.UTF_8).length <= config.context() - 3)
        val checkpoint = Path.of("checkpoints/demo2-dictionary.vanus")
        // An explicit step count always (re)trains; a bare `demo2dict` reuses an existing checkpoint if present.
        val model = if rest.isEmpty && Files.exists(checkpoint) then Transformer.load(checkpoint)
          else train(config, pairs, rest.headOption.fold(200)(_.toInt), checkpoint, 8)
        TransformerVisualizerApp.launch(model, "Dictionary", pairs.size, checkpoint.toString, pairs.map(_._1).toSet.asJava)
      case "train" :: size :: input :: steps :: output :: rest if rest.length <= 1 =>
        train(modelConfig(size), loadPairs(Path.of(input)), steps.toInt, Path.of(output),
          rest.headOption.fold(1)(_.toInt))
      case "pretrain" :: size :: input :: steps :: output :: rest if rest.length <= 2 =>
        val text = Files.readString(Path.of(input), StandardCharsets.UTF_8)
        val batch = rest.headOption.fold(1)(_.toInt)
        val requestedVocabulary = rest.drop(1).headOption.fold(512)(_.toInt)
        pretrain(modelConfig(size), text, steps.toInt, Path.of(output), batch, requestedVocabulary)
      case "continue" :: checkpoint :: input :: steps :: output :: rest if rest.length <= 1 =>
        val state = Transformer.loadTraining(Path.of(checkpoint))
        trainExisting(state.model(), state.optimizer(), loadPairs(Path.of(input)), steps.toInt,
          Path.of(output), rest.headOption.fold(1)(_.toInt), state.completedSteps())
      case "continue-pretrain" :: checkpoint :: input :: steps :: output :: rest if rest.length <= 1 =>
        val state = Transformer.loadTraining(Path.of(checkpoint))
        pretrainExisting(state.model(), state.optimizer(),
          Files.readString(Path.of(input), StandardCharsets.UTF_8), steps.toInt, Path.of(output),
          rest.headOption.fold(1)(_.toInt), state.completedSteps())
      case "gui" :: checkpoint :: rest if rest.length <= 1 =>
        val pairs = rest.headOption.fold(Vector.empty[(String, String)])(path => loadPairs(Path.of(path)))
        val datasetName = rest.headOption.getOrElse("No prompt dataset")
        TransformerVisualizerApp.launch(Transformer.load(Path.of(checkpoint)), datasetName, pairs.size,
          checkpoint, pairs.map(_._1).toSet.asJava)
      case "server" :: checkpoint :: input :: serverArgs =>
        val pairs = loadPairs(Path.of(input))
        VanusAiServer.run(Transformer.load(Path.of(checkpoint)), pairs.map(_._1).toSet, serverArgs.toArray)
      case "chat" :: checkpoint :: prompt if prompt.nonEmpty =>
        val model = Transformer.load(Path.of(checkpoint))
        println(model.generate(prompt.mkString(" "), 100, 0, 1, 42))
      case _ => throw new IllegalArgumentException("Run 'info' for command usage")

  /** Teacher-forced loss measures token prediction; exact match measures an entire generated reply. */
  private def evaluate(model: Transformer, pairs: Vector[(String, String)], label: String): Unit =
    require(pairs.nonEmpty, "Evaluation dataset is empty")
    var matched = 0
    var correct = 0L
    var tokens = 0L
    var lossSum = 0.0
    println(s"\n$label:")
    for (prompt, answer) <- pairs do
      val (input, targets) = example(model.tokenizer(), prompt, answer, model.config.context())
      val logits = Tensor.noGrad(() => model.forward(input))
      val count = targets.count(_ >= 0)
      lossSum += Tensor.noGrad(() => logits.crossEntropy(targets)).data(0) * count
      for i <- targets.indices if targets(i) >= 0 do
        val predicted = (0 until logits.cols).maxBy(j => logits.data(i * logits.cols + j))
        if predicted == targets(i) then correct += 1
      tokens += count
      val reply = model.generate(prompt, model.config.context(), 0, 1, 42)
      val exact = reply == answer
      if exact then matched += 1
      println(s"${if exact then "OK" else "MISS"} | $prompt -> $reply")
      if !exact then println(s"       expected: $answer")
    println(f"Exact replies: $matched/${pairs.size} (${100.0 * matched / pairs.size}%.1f%%) | answer-token accuracy: ${100.0 * correct / tokens}%.1f%% | loss: ${lossSum / tokens}%.4f")

  private def loadPairs(path: Path): Vector[(String, String)] =
    Files.readAllLines(path).asScala.filter(_.nonEmpty).zipWithIndex.map { (line, index) =>
      val fields = line.split("\t", -1)
      require(fields.length == 2 && fields.forall(_.nonEmpty), s"Line ${index + 1}: expected prompt<TAB>answer")
      fields(0) -> fields(1)
    }.toVector

  private def requireJava21(): Unit =
    val feature = Runtime.version().feature()
    if feature < 21 then
      throw new IllegalStateException(
        s"Vanus requires Java 21 or newer; current runtime is Java $feature"
      )

  private def example(tokenizer: TextTokenizer, prompt: String, answer: String, context: Int): (Array[Int], Array[Int]) =
    val prefix = tokenizer.prompt(prompt)
    val sequence = prefix ++ tokenizer.encode(answer) ++ Array(ByteTokenizer.EOS)
    require(sequence.length - 1 <= context, s"Example exceeds $context tokens")
    val input = sequence.dropRight(1)
    val targets = sequence.drop(1)
    Arrays.fill(targets, 0, prefix.length - 1, -1)
    (input, targets)

  private def train(config: ModelConfig, pairs: Vector[(String, String)], steps: Int, output: Path): Transformer =
    train(config, pairs, steps, output, 1)

  private def train(config: ModelConfig, pairs: Vector[(String, String)], steps: Int, output: Path, batchSize: Int): Transformer =
    train(config, pairs, steps, output, batchSize, defaultPromptNoisePercent)

  private def train(config: ModelConfig, pairs: Vector[(String, String)], steps: Int, output: Path,
                    batchSize: Int, promptNoisePercent: Int = defaultPromptNoisePercent): Transformer =
    val model = new Transformer(config, 42)
    trainExisting(model, new AdamW(model.parameters(), 0.01f), pairs, steps, output, batchSize, 0,
      promptNoisePercent)

  private def trainExisting(model: Transformer, optimizer: AdamW, pairs: Vector[(String, String)], steps: Int,
                            output: Path, batchSize: Int, completedSteps: Long,
                            promptNoisePercent: Int = defaultPromptNoisePercent): Transformer =
    require(steps > 0 && pairs.nonEmpty, "Positive steps and nonempty dataset required")
    require(batchSize > 0, "Positive batch size required")
    require(promptNoisePercent >= 0 && promptNoisePercent <= 100,
      "Prompt noise percentage must be between 0 and 100")
    val examples = if promptNoisePercent == 0 then
      pairs.map((p, a) => example(model.tokenizer(), p, a, model.config.context()))
    else Vector.empty
    val rng = new scala.util.Random(42)
    val startedAt = System.nanoTime()
    val objective = s"supervised prompt→reply promptNoise=$promptNoisePercent%"
    val progress = new TrainingProgress(model, objective, steps, batchSize, completedSteps, startedAt)
    logger.info(s"Training ${model.config.parameterCount()} parameters on ${pairs.size} examples, " +
      s"tokenizer=${model.tokenizer().kind()}, batch=$batchSize, promptNoise=$promptNoisePercent%, " +
      s"startingStep=$completedSteps, CPU float32")
    for step <- 1 to steps do
      val batch = Array.fill(batchSize) {
        if promptNoisePercent == 0 then examples(rng.nextInt(examples.size))
        else
          val (prompt, answer) = pairs(rng.nextInt(pairs.size))
          val augmented = if rng.nextInt(100) < promptNoisePercent then corruptPrompt(prompt, rng) else prompt
          example(model.tokenizer(), augmented, answer, model.config.context())
      }
      model.zeroGrad()
      var lossTotal = 0.0
      for (input, targets) <- batch do
        val loss = model.forward(input).crossEntropy(targets)
        lossTotal += loss.data(0)
        loss.backward()
      averageGradients(model, batchSize)
      val rate = learningRate(step, steps)
      val norm = optimizer.step(rate, 1.0f)
      progress.update(step, lossTotal / batchSize, norm, rate)
    model.saveTraining(output, optimizer, completedSteps + steps)
    logger.info(s"Saved $output after ${formatElapsed(System.nanoTime() - startedAt)}")
    logger.info(s"Prompt: ${pairs.head._1}")
    logger.info(s"Response: ${model.generate(pairs.head._1, 100, 0, 1, 42)}")
    model

  /** Makes one small, meaning-preserving-ish prompt typo for robust supervised training. */
  private[vanus] def corruptPrompt(prompt: String, rng: scala.util.Random): String =
    require(prompt.nonEmpty, "Cannot corrupt an empty prompt")
    val candidates = scala.collection.mutable.ArrayBuffer.empty[String]
    val withoutEnding = prompt.replaceFirst("[.!?]+$", "")
    if withoutEnding.nonEmpty && withoutEnding != prompt then candidates += withoutEnding
    if prompt.contains(',') then candidates += prompt.replaceFirst(",", "")

    val internalLetters = prompt.indices.filter(i => i > 0 && i + 1 < prompt.length &&
      Character.isLetter(prompt.charAt(i - 1)) && Character.isLetter(prompt.charAt(i)) &&
      Character.isLetter(prompt.charAt(i + 1)))
    if internalLetters.nonEmpty then
      val index = internalLetters(rng.nextInt(internalLetters.size))
      candidates += prompt.substring(0, index) + prompt.substring(index + 1)

    val adjacentLetters = prompt.indices.dropRight(1).filter(i =>
      Character.isLetter(prompt.charAt(i)) && Character.isLetter(prompt.charAt(i + 1)))
    if adjacentLetters.nonEmpty then
      val index = adjacentLetters(rng.nextInt(adjacentLetters.size))
      candidates += prompt.substring(0, index) + prompt.charAt(index + 1) + prompt.charAt(index) +
        prompt.substring(index + 2)

    val usable = candidates.distinct.filter(value => value.nonEmpty && value != prompt)
    if usable.isEmpty then prompt else usable(rng.nextInt(usable.size))

  private def pretrain(baseConfig: ModelConfig, text: String, steps: Int, output: Path,
                       batchSize: Int, requestedVocabulary: Int): Transformer =
    require(steps > 0 && batchSize > 0, "Positive steps and batch size required")
    val tokenizer = BpeTokenizer.train(text, requestedVocabulary)
    val config = baseConfig.withVocabulary(tokenizer.vocabulary())
    val model = new Transformer(config, 42, tokenizer)
    val optimizer = new AdamW(model.parameters(), 0.01f)
    pretrainExisting(model, optimizer, text, steps, output, batchSize, 0)

  private def pretrainExisting(model: Transformer, optimizer: AdamW, text: String, steps: Int,
                               output: Path, batchSize: Int, completedSteps: Long): Transformer =
    require(steps > 0 && batchSize > 0, "Positive steps and batch size required")
    val tokenizer = model.tokenizer()
    val rng = new scala.util.Random(42)
    val tokens = tokenizer.encode(text)
    require(tokens.length >= 2, "Pretraining text must contain at least two tokens")
    val window = math.min(model.config.context() + 1, tokens.length)
    val startedAt = System.nanoTime()
    val progress = new TrainingProgress(model, "continuous next-token", steps, batchSize, completedSteps, startedAt)
    val merges = tokenizer match
      case bpe: BpeTokenizer => s", merges=${bpe.merges().size()}"
      case _ => ""
    logger.info(s"Continuous-text pretraining ${model.config.parameterCount()} parameters on ${tokens.length} tokens, " +
      s"tokenizer=${tokenizer.kind()}, vocabulary=${tokenizer.vocabulary()}$merges, batch=$batchSize, " +
      s"startingStep=$completedSteps, CPU float32")
    for step <- 1 to steps do
      model.zeroGrad()
      var lossTotal = 0.0
      for _ <- 1 to batchSize do
        val start = if tokens.length == window then 0 else rng.nextInt(tokens.length - window + 1)
        val sequence = Arrays.copyOfRange(tokens, start, start + window)
        val input = sequence.dropRight(1)
        val targets = sequence.drop(1)
        val loss = model.forward(input).crossEntropy(targets)
        lossTotal += loss.data(0)
        loss.backward()
      averageGradients(model, batchSize)
      val rate = learningRate(step, steps)
      val norm = optimizer.step(rate, 1.0f)
      progress.update(step, lossTotal / batchSize, norm, rate)
    model.saveTraining(output, optimizer, completedSteps + steps)
    logger.info(s"Saved $output after ${formatElapsed(System.nanoTime() - startedAt)}")
    model

  private def averageGradients(model: Transformer, batchSize: Int): Unit =
    for parameter <- model.parameters().asScala do
      for index <- parameter.grad.indices do parameter.grad(index) /= batchSize

  private def learningRate(step: Int, steps: Int): Float =
    val warmup = math.min(1.0, step / 10.0)
    val cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.Pi * step / steps))
    (0.003 * warmup * cosine).toFloat

  private def modelConfig(size: String): ModelConfig = size match
    case "tiny" => ModelConfig.tiny()
    case "200k" => ModelConfig.vanus200k()
    case "1m" => ModelConfig.vanus1m()
    case "20m" => ModelConfig.vanus20m()
    case _ => throw new IllegalArgumentException("Model size must be tiny, 200k, 1m or 20m")

  /** Emits bounded, time-based progress instead of flooding output on fast runs. */
  private final class TrainingProgress(model: Transformer, objective: String, steps: Int,
                                       batchSize: Int, completedSteps: Long, startedAt: Long):
    private val intervalNanos = 4_000_000_000L
    private var lastLoggedAt = startedAt
    private var lastLoggedStep = 0
    private var windowLoss = 0.0
    private var windowSteps = 0

    def update(step: Int, loss: Double, gradientNorm: Double, learningRate: Float): Unit =
      windowLoss += loss
      windowSteps += 1
      val now = System.nanoTime()
      if step == 1 || step == steps || now - lastLoggedAt >= intervalNanos then
        val elapsedNanos = now - startedAt
        val intervalSeconds = (now - lastLoggedAt) / 1_000_000_000.0
        val overallSeconds = elapsedNanos / 1_000_000_000.0
        val intervalSteps = step - lastLoggedStep
        val rate = if intervalSeconds > 0.05 then intervalSteps / intervalSeconds
          else if overallSeconds > 0.05 then step / overallSeconds else 0.0
        val samplesPerSecond = rate * batchSize
        val eta = if rate > 0 && step < steps then
          formatElapsed((((steps - step) / rate) * 1_000_000_000L).toLong)
        else if step == steps then "00:00:00" else "calculating"
        val percent = 100.0 * step / steps
        val averageLoss = windowLoss / windowSteps
        var weightSquared = 0.0
        var gradientSquared = 0.0
        var maximumWeight = 0.0
        var parameterCount = 0L
        for parameter <- model.parameters().asScala do
          parameterCount += parameter.data.length
          for value <- parameter.data do
            weightSquared += value.toDouble * value
            maximumWeight = math.max(maximumWeight, math.abs(value.toDouble))
          for gradient <- parameter.grad do gradientSquared += gradient.toDouble * gradient
        val weightRms = math.sqrt(weightSquared / parameterCount)
        val gradientRms = math.sqrt(gradientSquared / parameterCount)
        logger.info(f"training elapsed=${formatElapsed(elapsedNanos)}%s progress=$percent%.1f%% " +
          f"step=$step%d/$steps%d total=${completedSteps + step}%d avgLoss=$averageLoss%.4f " +
          f"batch=$batchSize%d eta=$eta%s")
        logger.info(f"  objective=$objective%s tokenizer=${model.tokenizer().kind()}%s " +
          f"vocabulary=${model.config.vocabulary()}%d context=${model.config.context()}%d " +
          f"optimizer=AdamW learningRate=$learningRate%.7f " +
          f"gradNorm=$gradientNorm%.4f rate=$rate%.2f steps/s throughput=$samplesPerSecond%.2f samples/s")
        logger.info(f"  weights=all trainable tensors=${model.parameters().size()}%d parameters=$parameterCount%d " +
          f"weightRms=$weightRms%.6f maxAbs=$maximumWeight%.6f gradRms=$gradientRms%.8f")
        logger.info(s"  architecture=layers:${model.config.layers()} width:${model.config.width()} " +
          s"hidden:${model.config.hidden()} heads:${model.config.heads()} groups=embedding/tied-output, " +
          "attention(q,k,v,o), SwiGLU(gate,up,down), RMSNorm gains")
        lastLoggedAt = now
        lastLoggedStep = step
        windowLoss = 0.0
        windowSteps = 0

  private[vanus] def formatElapsed(nanoseconds: Long): String =
    require(nanoseconds >= 0, "Elapsed time cannot be negative")
    val totalSeconds = nanoseconds / 1_000_000_000L
    val hours = totalSeconds / 3600
    val minutes = totalSeconds % 3600 / 60
    val seconds = totalSeconds % 60
    f"$hours%02d:$minutes%02d:$seconds%02d"
