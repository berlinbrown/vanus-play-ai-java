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
        for (name, config) <- List("tiny" -> ModelConfig.tiny(), "20m" -> ModelConfig.vanus20m()) do
          println(s"$name: ${config.parameterCount()} parameters; $config")
        println("Commands: demo [steps] [checkpoint] | demo2 [steps] | demo2dict [steps] | train <tiny|20m> <pairs.tsv> <steps> <checkpoint> | chat <checkpoint> <prompt>")
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
        val config = ModelConfig.tiny()
        val pairs = loadPairs(demoDictData).filter((p, a) => (p + a).getBytes(StandardCharsets.UTF_8).length <= config.context() - 3)
        val checkpoint = Path.of("checkpoints/demo2-dictionary.vanus")
        // An explicit step count always (re)trains; a bare `demo2dict` reuses an existing checkpoint if present.
        val model = if rest.isEmpty && Files.exists(checkpoint) then Transformer.load(checkpoint)
          else train(config, pairs, rest.headOption.fold(200)(_.toInt), checkpoint)
        TransformerVisualizerApp.launch(model, "Dictionary", pairs.size, checkpoint.toString, pairs.map(_._1).toSet.asJava)
      case "train" :: size :: input :: steps :: output :: Nil =>
        val config = size match
          case "tiny" => ModelConfig.tiny()
          case "20m" => ModelConfig.vanus20m()
          case _ => throw new IllegalArgumentException("Model size must be tiny or 20m")
        train(config, loadPairs(Path.of(input)), steps.toInt, Path.of(output))
      case "chat" :: checkpoint :: prompt if prompt.nonEmpty =>
        val model = Transformer.load(Path.of(checkpoint))
        println(model.generate(prompt.mkString(" "), 100, 0, 1, 42))
      case _ => throw new IllegalArgumentException("Run 'info' for command usage")

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
    require(steps > 0 && pairs.nonEmpty, "Positive steps and nonempty dataset required")
    val examples = pairs.map((p, a) => example(p, a, config.context()))
    val model = new Transformer(config, 42)
    val optimizer = new AdamW(model.parameters(), 0.01f)
    val rng = new scala.util.Random(42)
    println(s"Training ${config.parameterCount()} parameters on ${pairs.size} examples, CPU float32")
    for step <- 1 to steps do
      val (input, targets) = examples(rng.nextInt(examples.size))
      model.zeroGrad()
      val loss = model.forward(input).crossEntropy(targets)
      loss.backward()
      val warmup = math.min(1.0, step / 10.0)
      val cosine = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.Pi * step / steps))
      val norm = optimizer.step((0.003 * warmup * cosine).toFloat, 1.0f)
      if step == 1 || step % 25 == 0 || step == steps then
        println(f"step=$step%d loss=${loss.data(0)}%.4f gradNorm=$norm%.4f")
    model.save(output)
    println(s"Saved $output")
    println(s"Prompt: ${pairs.head._1}")
    println(s"Response: ${model.generate(pairs.head._1, 100, 0, 1, 42)}")
    model
