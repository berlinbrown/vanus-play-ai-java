package org.berlin.vanus

import java.io.{File, IOException}
import org.apache.logging.log4j.LogManager
import org.nanohttpd.protocols.http.tempfiles.{DefaultTempFile, ITempFile, ITempFileManager}
import org.nanohttpd.util.IFactory

/** Reserve a body's worst-case disk footprint before creating its first temporary file. */
final class VanusTempFiles(maxBytes: Long, maxBodyBytes: Long, maxParts: Int) extends IFactory[ITempFileManager]:
  private val reservation = math.max(1L, maxBodyBytes * 2)
  require(maxBytes >= reservation)
  private var reserved = 0L
  private val log = LogManager.getLogger("vanus.uploads")

  def reservedBytes: Long = synchronized { reserved }

  private def reserve(): Unit = synchronized {
    if reserved > maxBytes - reservation then throw new IOException("Temporary storage budget exhausted")
    reserved += reservation
  }

  private def release(): Unit = synchronized { reserved -= reservation }

  override def create(): ITempFileManager = new ITempFileManager:
    private val files = scala.collection.mutable.ArrayBuffer.empty[ITempFile]
    private var hasReservation = false

    override def createTempFile(hint: String): ITempFile =
      if files.size >= maxParts + 1 then throw new IOException("Too many temporary files")
      if !hasReservation then
        reserve()
        hasReservation = true
      try
        val file = new DefaultTempFile(new File(System.getProperty("java.io.tmpdir")))
        files += file
        file
      catch
        case error: Exception =>
          if files.isEmpty && hasReservation then
            release()
            hasReservation = false
          throw error

    override def clear(): Unit =
      val failed = files.filter { file =>
        try
          file.delete()
          false
        catch
          case _: Exception =>
            log.warn("Temporary file cleanup failed; its storage reservation is retained")
            true
      }.toVector
      files.clear()
      files ++= failed
      if files.isEmpty && hasReservation then
        release()
        hasReservation = false
