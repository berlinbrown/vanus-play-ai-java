package org.berlin.vanus

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.Arrays
import scala.jdk.CollectionConverters.*

object Main:
  // Default demo corpus; filtered per-config below since the tiny model's context is small.
  private val demoData = Path.of("data/pride-and-prejudice.tsv")
  private val demoDictData = Path.of("data/dictionary2.tsv")
  private val tokenizer = new ByteTokenizer()


  def main(args: Array[String]): Unit =
    requireJava21()
    args.toList match
      case Nil | "info" :: Nil =>
        for (name, config) <- List("tiny" -> ModelConfig.tiny(), "dictionary" -> ModelConfig.dictionary(), "200k" -> ModelConfig.vanus200k(), "20m" -> ModelConfig.vanus20m()) do
          println(s"$name: ${config.parameterCount()} parameters; $config")
        println("Commands: demo2daily200k [steps] | demo2daily [steps] | greetings [steps] | demo2greet [steps] | eval <checkpoint> <pairs.tsv> | demo [steps] [checkpoint] | demo2 [steps] | demo2dict [steps] | train <tiny|200k|20m> <pairs.tsv> <steps> <checkpoint> | chat <checkpoint> <prompt>")
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
      case "train" :: size :: input :: steps :: output :: Nil =>
        val config = size match
          case "tiny" => ModelConfig.tiny()
          case "200k" => ModelConfig.vanus200k()
          case "20m" => ModelConfig.vanus20m()
          case _ => throw new IllegalArgumentException("Model size must be tiny, 200k or 20m")
        train(config, loadPairs(Path.of(input)), steps.toInt, Path.of(output))
      case "chat" :: checkpoint :: prompt if prompt.nonEmpty =>
        val model = Transformer.load(Path.of(checkpoint))
        println(model.generate(prompt.mkString(" "), 100, 0, 1, 42))
      case _ => throw new IllegalArgumentException("Run 'info' for command usage")

  /** Teacher-forced loss measures byte prediction; exact match measures an entire generated reply. */
  private def evaluate(model: Transformer, pairs: Vector[(String, String)], label: String): Unit =
    require(pairs.nonEmpty, "Evaluation dataset is empty")
    var matched = 0
    var correct = 0L
    var tokens = 0L
    var lossSum = 0.0
    println(s"\n$label:")
    for (prompt, answer) <- pairs do
      val (input, targets) = example(prompt, answer, model.config.context())
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
    println(f"Exact replies: $matched/${pairs.size} (${100.0 * matched / pairs.size}%.1f%%) | answer-byte accuracy: ${100.0 * correct / tokens}%.1f%% | loss: ${lossSum / tokens}%.4f")

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

  private def example(prompt: String, answer: String, context: Int): (Array[Int], Array[Int]) =
    val prefix = tokenizer.prompt(prompt)
    val sequence = prefix ++ tokenizer.encode(answer) ++ Array(ByteTokenizer.EOS)
    require(sequence.length - 1 <= context, s"Example exceeds $context byte tokens")
    val input = sequence.dropRight(1)
    val targets = sequence.drop(1)
    Arrays.fill(targets, 0, prefix.length - 1, -1)
    (input, targets)

  private def train(config: ModelConfig, pairs: Vector[(String, String)], steps: Int, output: Path): Transformer =
    train(config, pairs, steps, output, 1)

  private def train(config: ModelConfig, pairs: Vector[(String, String)], steps: Int, output: Path, batchSize: Int): Transformer =
    require(steps > 0 && pairs.nonEmpty, "Positive steps and nonempty dataset required")
    require(batchSize > 0, "Positive batch size required")
    val examples = pairs.map((p, a) => example(p, a, config.context()))
    val model = new Transformer(config, 42)
    val optimizer = new AdamW(model.parameters(), 0.01f)
    val rng = new scala.util.Random(42)
    println(s"Training ${config.parameterCount()} parameters on ${pairs.size} examples, CPU float32")
    for step <- 1 to steps do
      val batch = Array.fill(batchSize)(examples(rng.nextInt(examples.size)))
      model.zeroGrad()
      var lossTotal = 0.0
      for (input, targets) <- batch do
        val loss = model.forward(input).crossEntropy(targets)
        lossTotal += loss.data(0)
        loss.backward()
      for parameter <- model.parameters().asScala do
        for index <- parameter.grad.indices do parameter.grad(index) /= batchSize
      val warmup = math.min(1.0, step / 10.0)
      val cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.Pi * step / steps))
      val norm = optimizer.step((0.003 * warmup * cosine).toFloat, 1.0f)
      if step == 1 || step % 25 == 0 || step == steps then
        println(f"step=$step%d loss=${lossTotal / batchSize}%.4f gradNorm=$norm%.4f batch=$batchSize%d")
    model.save(output)
    println(s"Saved $output")
    println(s"Prompt: ${pairs.head._1}")
    println(s"Response: ${model.generate(pairs.head._1, 100, 0, 1, 42)}")
    model
