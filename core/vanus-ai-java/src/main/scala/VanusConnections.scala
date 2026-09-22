package org.berlin.vanus

import java.util.concurrent.{ConcurrentHashMap, Executors, Semaphore, TimeUnit}
import java.util.concurrent.atomic.{AtomicBoolean, LongAdder}
import org.nanohttpd.protocols.http.ClientHandler
import org.nanohttpd.protocols.http.threading.IAsyncRunner
import scala.jdk.CollectionConverters.*

/** One virtual thread per admitted connection; permits bound sockets and associated state. */
final class VanusConnections(maxConnections: Int, shutdownMillis: Int) extends IAsyncRunner:
  require(maxConnections > 0 && shutdownMillis > 0)

  private val permits = new Semaphore(maxConnections)
  private val clients = ConcurrentHashMap.newKeySet[ClientHandler]()
  private val stopping = new AtomicBoolean(false)
  private val executor = Executors.newThreadPerTaskExecutor(Thread.ofVirtual().name("vanus-connection-", 0).factory())
  private val clock = Executors.newSingleThreadScheduledExecutor(
    Thread.ofPlatform().daemon(true).name("vanus-deadlines").factory()
  )
  private val rejected = new LongAdder
  private var monitoring = false

  def activeConnections: Int = clients.size()
  def rejectedConnections: Long = rejected.sum()

  override def exec(client: ClientHandler): Unit = synchronized {
    if stopping.get() || !permits.tryAcquire() then
      rejected.increment()
      client.close()
    else
      clients.add(client)
      try
        if !monitoring then
          clock.scheduleAtFixedRate(
            new Runnable:
              override def run(): Unit =
                val now = System.nanoTime()
                clients.asScala.foreach(_.expireIfOverdue(now)),
            0, 50, TimeUnit.MILLISECONDS
          )
          monitoring = true
        executor.execute(client)
      catch
        case e: RuntimeException =>
          client.close()
          closed(client)
          throw e
  }

  override def closed(client: ClientHandler): Unit =
    if clients.remove(client) then permits.release()

  override def closeAll(): Unit =
    val shouldStop = synchronized { stopping.compareAndSet(false, true) }
    if shouldStop then
      clients.asScala.foreach(_.drain())
      executor.shutdown()
      try
        if !executor.awaitTermination(shutdownMillis, TimeUnit.MILLISECONDS) then
          clients.asScala.foreach(_.close())
          executor.shutdownNow()
      catch
        case _: InterruptedException =>
          clients.asScala.foreach(_.close())
          executor.shutdownNow()
          Thread.currentThread().interrupt()
      finally clock.shutdownNow()
