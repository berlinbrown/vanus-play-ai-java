package org.berlin.vanus

import java.util.concurrent.{Executors, ScheduledExecutorService, TimeUnit}
import java.util.concurrent.atomic.AtomicBoolean
import org.apache.logging.log4j.LogManager
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** Headless equivalent of Swing self-talk: generate, persist, feed words back, repeat. */
final class VanusAiService(model: Transformer, store: VanusMessageStore, knownPrompts: Set[String]) extends AutoCloseable:
  private val log = LogManager.getLogger(classOf[VanusAiService])
  private val running = new AtomicBoolean(false)
  private val executor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor(
    Thread.ofPlatform().daemon(true).name("vanus-ai-self-talk").factory())
  private val random = new java.util.Random()
  private var nextPrompt = "hello"
  private var nextResetAt = 0L
  private var turn = 0L

  def start(): Unit =
    if running.compareAndSet(false, true) then
      nextResetAt = System.nanoTime() + VanusAiService.ResetIntervalNanos
      log.info("Headless self-talk started interval=30s resetInterval=5m initialPrompt=hello database={}",
        store.databasePath)
      executor.scheduleWithFixedDelay(
        () => runTurn(),
        0,
        VanusAiService.MessageIntervalSeconds,
        TimeUnit.SECONDS)

  private def runTurn(): Unit =
    if !running.get() then return
    try
      val now = System.nanoTime()
      val reset = now >= nextResetAt
      if reset then
        nextPrompt = if random.nextBoolean() then "hello" else "goodbye"
        nextResetAt = now + VanusAiService.ResetIntervalNanos
      val prompt = nextPrompt
      turn += 1
      val promptId = store.add(prompt, "prompt")
      log.info("AI input turn={} id={} reset={} known={} text={}",
        turn, promptId, reset, knownPrompts.contains(prompt), prompt)
      val started = System.nanoTime()
      val reply = model.generate(prompt, 150, 0, 1, 42)
      val responseId = store.add(reply, "response")
      val elapsedMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started)
      log.info("AI output turn={} id={} elapsedMs={} text={}", turn, responseId, elapsedMillis, reply)
      nextPrompt = TransformerVisualizerApp.selfTalkPrompt(reply, model.config.context() - 4, random)
      log.info("AI queued turn={} next={}", turn + 1, nextPrompt)
    catch
      case NonFatal(error) =>
        log.error("Headless self-talk turn {} failed; resetting next prompt to hello", turn, error)
        nextPrompt = "hello"

  override def close(): Unit =
    if running.compareAndSet(true, false) then
      executor.shutdownNow()
      try executor.awaitTermination(5, TimeUnit.SECONDS)
      catch case _: InterruptedException => Thread.currentThread().interrupt()
      log.info("Headless self-talk stopped turns={}", turn)

object VanusAiService:
  private[vanus] val MessageIntervalSeconds = 30L
  private[vanus] val ResetIntervalNanos = TimeUnit.MINUTES.toNanos(5)

object VanusAiServer:
  private val log = LogManager.getLogger(VanusAiServer.getClass)

  def run(model: Transformer, knownPrompts: Set[String], serverArgs: Array[String]): Unit =
    val config = VanusServerConfig.fromArgs(serverArgs)
    val dataDirectory = config.rootDirs.head.toPath
    val store = new VanusMessageStore(dataDirectory)
    val routes = VanusRoutes.bot(store)
    val server = WebServerMain.build(config, routes, serveFiles = false)
    val ai = new VanusAiService(model, store, knownPrompts)
    log.info("Starting Vanus AI server database={} statusPath={} messagesPath={}",
      store.databasePath, VanusBotRoutes.statusPath, VanusBotRoutes.messagesPath)
    ai.start()
    VanusRuntime.run(server, () => ai.close())
