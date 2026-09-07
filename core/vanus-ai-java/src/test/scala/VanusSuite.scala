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
