package org.berlin.vanus

import java.util.concurrent.CountDownLatch
import org.apache.logging.log4j.LogManager

object VanusRuntime:
  private val log = LogManager.getLogger("vanus.lifecycle")

  def run(server: WebServer, onStop: () => Unit = () => ()): Unit =
    val stopped = new CountDownLatch(1)
    val shutdown = new Thread(() =>
      try
        try onStop()
        finally server.stop()
      finally stopped.countDown()
    , "vanus-shutdown")
    Runtime.getRuntime.addShutdownHook(shutdown)
    try
      server.start(server.getLimits.idleTimeoutMillis(), false)
      log.info("Listening on {}:{}", server.getHostname, server.getListeningPort)
      stopped.await()
    catch
      case _: InterruptedException => Thread.currentThread().interrupt()
    finally
      onStop()
      server.stop()
      try Runtime.getRuntime.removeShutdownHook(shutdown)
      catch case _: IllegalStateException => () // JVM shutdown is already in progress.
