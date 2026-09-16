package org.berlin.vanus

import java.nio.file.Files
import scala.jdk.CollectionConverters.*

class VanusSuite extends munit.FunSuite:
  private def config = new ModelConfig(260, 8, 16, 1, 2, 64)

  test("UTF-8 round trip and distinct role tokens") {
    val tokenizer = new ByteTokenizer()
    val text = "Hello, \u4e16\u754c \ud83c\udf0d"
    assertEquals(tokenizer.decode(tokenizer.encode(text)), text)
    assertEquals(tokenizer.prompt("a").toSeq, Seq(256, 258, 97, 259))
  }

  test("BPE learns reusable subwords and round trips UTF-8") {
    val text = "hello hello helper hello 世界 世界 " * 8
    val tokenizer = BpeTokenizer.train(text, 280)
    assert(tokenizer.vocabulary() > ByteTokenizer.VOCABULARY)
    assert(tokenizer.encode("hello hello").length < new ByteTokenizer().encode("hello hello").length)
    assertEquals(tokenizer.decode(tokenizer.encode("hello 世界")), "hello 世界")
    val model = new Transformer(config.withVocabulary(tokenizer.vocabulary()), 7, tokenizer)
    assertEquals(model.config.vocabulary(), tokenizer.vocabulary())
  }

  test("backward accumulates through shared operands") {
    val x = Tensor.zeros(1, 2, true)
    x.data(0) = 0.3f; x.data(1) = -0.7f
    val loss = x.add(x).crossEntropy(Array(0))
    loss.backward()
    val expected = (2 * (1 / (1 + math.exp(-2)) - 1)).toFloat
    assertEqualsDouble(x.grad(0).toDouble, expected.toDouble, 1e-6)
    loss.backward()
    assertEqualsDouble(x.grad(0).toDouble, 2 * expected.toDouble, 1e-6)
  }

  test("end-to-end finite differences for every parameter group") {
    val model = new Transformer(config, 7)
    val input = Array(256, 97, 98, 259)
    val target = Array(97, 98, 259, 257)
    model.forward(input).crossEntropy(target).backward()
    for (name, p) <- model.namedParameters().asScala do
      val indices = p.grad.indices.sortBy(i => -math.abs(p.grad(i))).take(3)
      for i <- indices do
        val original = p.data(i)
        val epsilon = 0.002f
        p.data(i) = original + epsilon
        val plus = Tensor.noGrad(() => model.forward(input).crossEntropy(target).data(0))
        p.data(i) = original - epsilon
        val minus = Tensor.noGrad(() => model.forward(input).crossEntropy(target).data(0))
        p.data(i) = original
        val numerical = (plus - minus) / (2 * epsilon)
        assert(math.abs(p.grad(i) - numerical) < 0.0015 + 0.03 * math.abs(numerical),
          s"$name[$i]: analytic=${p.grad(i)}, numerical=$numerical")
  }

  test("causality and noGrad") {
    val model = new Transformer(config, 11)
    val a = Tensor.noGrad(() => model.forward(Array(256, 97, 98)))
    val b = Tensor.noGrad(() => model.forward(Array(256, 97, 99)))
    assert(a.grad == null)
    assertEquals(a.data.take(520).toSeq, b.data.take(520).toSeq)
    assert(model.forward(Array(256)).grad != null)
  }

  test("attention Q K V gradients at nontrivial scores") {
    val rng = new java.util.Random(123)
    val q = Tensor.parameter(3, 4, rng, 0.7f)
    val k = Tensor.parameter(3, 4, rng, 0.7f)
    val v = Tensor.parameter(3, 4, rng, 0.7f)
    def loss() = Tensor.attention(q, k, v, 2).crossEntropy(Array(0, 2, 1))
    loss().backward()
    for p <- List(q, k, v); i <- p.data.indices do
      val original = p.data(i)
      p.data(i) = original + 0.002f
      val plus = Tensor.noGrad(() => loss().data(0))
      p.data(i) = original - 0.002f
      val minus = Tensor.noGrad(() => loss().data(0))
      p.data(i) = original
      assertEqualsDouble(p.grad(i).toDouble, ((plus - minus) / 0.004).toDouble, 0.0001)
  }

  test("20M model forward and backward smoke test") {
    val model = new Transformer(ModelConfig.vanus20m(), 42)
    assertEquals(model.parameters().asScala.map(_.data.length.toLong).sum, 20309184L)
    val loss = model.forward(Array(256, 258)).crossEntropy(Array(258, 97))
    assert(java.lang.Float.isFinite(loss.data(0)))
    loss.backward()
    assert(model.parameters().asScala.forall(p => p.grad.forall(java.lang.Float.isFinite)))
    assert(model.parameters().asScala.exists(p => p.grad.exists(_ != 0)))
  }

  test("checkpoint round trip and truncation rejection") {
    val path = Files.createTempFile("vanus-test-", ".bin")
    try
      val model = new Transformer(config, 12)
      model.save(path)
      val loaded = Transformer.load(path)
      assertEquals(model.forward(Array(256, 42)).data.toSeq, loaded.forward(Array(256, 42)).data.toSeq)
      Files.write(path, Array[Byte](0, 1))
      intercept[java.io.IOException](Transformer.load(path))
    finally Files.deleteIfExists(path)
  }

  test("training checkpoint restores tokenizer and AdamW moments") {
    val path = Files.createTempFile("vanus-training-test-", ".bin")
    try
      val tokenizer = BpeTokenizer.train("hello hello helper hello " * 8, 270)
      val dynamicConfig = config.withVocabulary(tokenizer.vocabulary())
      val model = new Transformer(dynamicConfig, 12, tokenizer)
      val optimizer = new AdamW(model.parameters(), 0.01f)
      val sequence = tokenizer.encode("hello hello")
      val input = sequence.dropRight(1)
      val targets = sequence.drop(1)
      model.forward(input).crossEntropy(targets).backward()
      optimizer.step(0.003f, 1f)
      model.saveTraining(path, optimizer, 123)

      val restored = Transformer.loadTraining(path)
      assertEquals(restored.completedSteps(), 123L)
      assertEquals(restored.optimizer().steps(), optimizer.steps())
      assertEquals(restored.model().tokenizer().kind(), "bpe")
      assertEquals(restored.model().tokenizer().encode("hello").toSeq, tokenizer.encode("hello").toSeq)

      for (candidateModel, candidateOptimizer) <- Seq(model -> optimizer, restored.model() -> restored.optimizer()) do
        candidateModel.zeroGrad()
        candidateModel.forward(input).crossEntropy(targets).backward()
        candidateOptimizer.step(0.002f, 1f)
      assertEquals(model.parameters().asScala.flatMap(_.data).toSeq,
        restored.model().parameters().asScala.flatMap(_.data).toSeq)
    finally Files.deleteIfExists(path)
  }

  test("parameter formula counts tied embedding once") {
    val model = new Transformer(config, 0)
    assertEquals(model.parameters().asScala.map(_.data.length.toLong).sum, config.parameterCount())
    assertEquals(ModelConfig.vanus20m().parameterCount(), 20309184L)
  }

  test("AdamW reduces loss and rejects nonfinite gradients before updating") {
    val model = new Transformer(config, 0)
    val optimizer = new AdamW(model.parameters(), 0.01f)
    val input = Array(256, 258, 97, 259, 98)
    val targets = Array(-1, -1, -1, 98, 257)
    val initial = model.forward(input).crossEntropy(targets).data(0)
    for _ <- 0 until 40 do
      model.zeroGrad()
      model.forward(input).crossEntropy(targets).backward()
      optimizer.step(0.01f, 1)
    assert(model.forward(input).crossEntropy(targets).data(0) < initial * 0.3)
    val p = model.parameters().iterator().next()
    val before = p.data.clone()
    p.grad(0) = Float.NaN
    intercept[IllegalStateException](optimizer.step(0.01f, 1))
    assertEquals(p.data.toSeq, before.toSeq)
  }

  test("invalid shapes, context and targets") {
    intercept[IllegalArgumentException](Tensor.zeros(2, 3, false).matmul(Tensor.zeros(2, 2, false)))
    intercept[IllegalArgumentException](Tensor.zeros(2, 3, true).crossEntropy(Array(-1, -1)))
    intercept[IllegalArgumentException](new Transformer(config, 0).forward(new Array[Int](65)))
    intercept[IllegalArgumentException](new ModelConfig(260, 9, 16, 1, 2, 64))
  }

  test("generation diagnostics preserve decoding and report context exhaustion") {
    val model = new Transformer(new ModelConfig(260, 8, 16, 1, 2, 6), 5)
    // Equal logits choose byte zero deterministically, avoiding an early EOS.
    model.parameters().asScala.foreach(p => java.util.Arrays.fill(p.data, 0f))
    val events = new java.util.ArrayList[Transformer.GenerationStep]()
    val result = model.generate("a", 10, 0, 1, 42, step => { events.add(step); () })
    assertEquals(result, model.generate("a", 10, 0, 1, 42))
    assertEquals(events.size(), 2)
    assertEquals(events.get(1).stopReason(), "context limit")
    assertEquals(events.get(0).contextUsed(), 4)
    assertEqualsDouble(events.get(0).candidates().get(0).probability(), 1.0 / 257, 1e-9)
    assertEquals(events.get(1).text(), result)
    intercept[IllegalArgumentException](model.generate("abc", 1, 0, 1, 42))
  }

  test("generation rejects nonfinite predictions") {
    val model = new Transformer(config, 5)
    model.parameters().iterator().next().data(0) = Float.NaN
    intercept[IllegalStateException](model.generate("a", 2, 0, 1, 42))
  }

  test("200K preset has the stated capacity and finite training gradients") {
    val model = new Transformer(ModelConfig.vanus200k(), 42)
    assertEquals(model.config.parameterCount(), 200544L)
    assertEquals(model.parameters().asScala.map(_.data.length.toLong).sum, 200544L)
    assertEquals(model.config.context(), ModelConfig.tiny().context())
    val loss = model.forward(Array(256, 258, 97, 259)).crossEntropy(Array(-1, -1, -1, 257))
    assert(java.lang.Float.isFinite(loss.data(0)))
    loss.backward()
    assert(model.parameters().asScala.forall(p => p.grad.forall(java.lang.Float.isFinite)))
    assert(model.parameters().asScala.exists(p => p.grad.exists(_ != 0)))
  }

  test("self-talk picks ordered words within the UTF-8 budget and handles empty replies") {
    val words = Seq("Hello", "there", "how", "are", "you", "doing", "today")
    val replies = (0 until 30).map { seed =>
      val prompt = TransformerVisualizerApp.selfTalkPrompt(words.mkString(" "), 20, new java.util.Random(seed))
      assert(prompt.nonEmpty)
      assert(prompt.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= 20)
      val indices = prompt.split(" ").map(words.indexOf(_)).toSeq
      assert(indices.forall(_ >= 0))
      assertEquals(indices, indices.sorted.distinct)
      prompt
    }
    assert(replies.distinct.size > 1)
    assertEquals(TransformerVisualizerApp.selfTalkPrompt("\u0000 !!!", 20, new java.util.Random(1)), "hello")
    val unicode = TransformerVisualizerApp.selfTalkPrompt("世界 你好 morning", 7, new java.util.Random(2))
    assert(unicode.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= 7)
    assert(!unicode.contains("\ufffd"))
  }

  test("checkpoint weight statistics cover all values and exact differences") {
    val current = Tensor.of(2, 2, 1f, -2f, 3f, -4f)
    val reference = Tensor.of(2, 2, 0f, -2f, 1f, -5f)
    val stats = CheckpointVisualizerApp.statistics(current, reference)
    assertEquals(stats.count(), 4)
    assertEqualsDouble(stats.mean(), -0.5, 1e-12)
    assertEqualsDouble(stats.rms(), math.sqrt(7.5), 1e-12)
    assertEqualsDouble(stats.min(), -4, 1e-12)
    assertEqualsDouble(stats.max(), 3, 1e-12)
    assertEqualsDouble(stats.deltaRms(), math.sqrt(1.5), 1e-12)
    assertEqualsDouble(stats.deltaMax(), 2, 1e-12)
    assertEquals(stats.changed(), 3)
  }

  test("checkpoint comparison requires identical architectures") {
    val a = new Transformer(config, 1)
    val b = new Transformer(config, 2)
    CheckpointVisualizerApp.validateComparable(a, b)
    intercept[IllegalArgumentException](
      CheckpointVisualizerApp.validateComparable(a, new Transformer(ModelConfig.tiny(), 2)))
  }

  test("elapsed training time formats short and multi-day runs") {
    assertEquals(Main.formatElapsed(0), "00:00:00")
    assertEquals(Main.formatElapsed(3_661_999_999_999L), "01:01:01")
    assertEquals(Main.formatElapsed(183_845_000_000_000L), "51:04:05")
    intercept[IllegalArgumentException](Main.formatElapsed(-1))
  }
