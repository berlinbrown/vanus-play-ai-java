package org.berlin.vanus

/** Fixed windows with bounded client state. Unknown clients are rejected when the table is full. */
final class VanusRateLimiter(maxRequests: Int, windowMillis: Long, maxClients: Int = 10000):
  require(maxRequests > 0 && windowMillis > 0 && maxClients > 0)

  private final case class Window(startMillis: Long, count: Int)
  private val windows = scala.collection.mutable.HashMap.empty[String, Window]
  private var nextCleanup = Long.MinValue

  def trackedClients: Int = synchronized { windows.size }

  def allow(key: String, nowMillis: Long = System.nanoTime() / 1000000L): Boolean = synchronized {
    if nowMillis >= nextCleanup then
      windows.filterInPlace((_, window) => nowMillis - window.startMillis < windowMillis)
      nextCleanup = nowMillis + windowMillis
    windows.get(key) match
      case Some(window) if nowMillis - window.startMillis < windowMillis =>
        if window.count >= maxRequests then false
        else
          windows.update(key, window.copy(count = window.count + 1))
          true
      case _ if windows.contains(key) || windows.size < maxClients =>
        windows.update(key, Window(nowMillis, 1))
        true
      case _ => false
  }
