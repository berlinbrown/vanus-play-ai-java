package org.berlin.vanus

import java.io.{File, IOException, InputStream}
import java.nio.channels.{Channels, FileChannel}
import java.nio.file.{LinkOption, StandardOpenOption}
import java.time.{Instant, ZoneOffset, ZonedDateTime}
import java.time.format.DateTimeFormatter
import org.nanohttpd.protocols.http.response.{IStatus, Response, Status}

object VanusFileResponses:
  def fixedResponse(status: IStatus, mime: String, message: String): Response =
    VanusSecurity.protect(Response.newFixedLengthResponse(status, mime, message))

  def serveFile(headers: java.util.Map[String, String], file: File, mime: String, head: Boolean = false): Response =
    try
      val size = file.length()
      val modified = file.lastModified() / 1000 * 1000
      val etag = s"W/\"${java.lang.Long.toHexString(file.lastModified())}-${java.lang.Long.toHexString(size)}\""
      def header(name: String): Option[String] = Option(headers.get(name))
      def date(value: String): Option[Long] =
        try Some(ZonedDateTime.parse(value, DateTimeFormatter.RFC_1123_DATE_TIME).toInstant.toEpochMilli)
        catch case _: java.time.format.DateTimeParseException => None
      val notModified = header("if-none-match") match
        case Some(value) => value.split(',').exists(tag => tag.trim == "*" || tag.trim.stripPrefix("W/") == etag.stripPrefix("W/"))
        case None => header("if-modified-since").flatMap(date).exists(_ >= modified)
      val requestedRange =
        if head || header("if-range").exists(value => !date(value).contains(modified)) then None
        else header("range")
      val range = requestedRange.flatMap(parseRange(_, size))
      val response =
        if notModified then fixedResponse(Status.NOT_MODIFIED, mime, "")
        else range match
          case Some(Left(_)) =>
            val invalid = fixedResponse(Status.RANGE_NOT_SATISFIABLE, mime, "")
            invalid.addHeader("Content-Range", s"bytes */$size")
            invalid
          case bounds =>
            val (start, end) = bounds.collect { case Right(pair) => pair }.getOrElse((0L, size - 1))
            val length = math.max(0L, end - start + 1)
            val body = if head then InputStream.nullInputStream() else open(file, start)
            val status = if bounds.isDefined then Status.PARTIAL_CONTENT else Status.OK
            val result = Response.newFixedLengthResponse(status, mime, body, length)
            result.addHeader("Content-Length", length.toString)
            if bounds.isDefined then result.addHeader("Content-Range", s"bytes $start-$end/$size")
            result
      response.addHeader("Accept-Ranges", "bytes")
      response.addHeader("ETag", etag)
      response.addHeader("Last-Modified", DateTimeFormatter.RFC_1123_DATE_TIME.format(Instant.ofEpochMilli(modified).atZone(ZoneOffset.UTC)))
      VanusSecurity.protect(response)
    catch
      case _: IOException => fixedResponse(Status.FORBIDDEN, "text/plain", VanusConstants.ErrorReadingFileFailed)

  private def open(file: File, start: Long): InputStream =
    val channel = FileChannel.open(file.toPath, StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS)
    try
      channel.position(start)
      Channels.newInputStream(channel)
    catch
      case error: Throwable =>
        channel.close()
        throw error

  // Unsupported units, multiple ranges, and malformed syntax are ignored; unsatisfiable ranges get 416.
  private def parseRange(value: String, size: Long): Option[Either[Unit, (Long, Long)]] =
    val pattern = "bytes=([0-9]*)-([0-9]*)".r
    value match
      case pattern(first, last) if first.nonEmpty || last.nonEmpty =>
        if first.isEmpty then
          last.toLongOption.map { suffix =>
            if suffix == 0 || size == 0 then Left(())
            else Right((math.max(0L, size - suffix), size - 1))
          }
        else
          for
            start <- first.toLongOption
            end <- if last.isEmpty then Some(size - 1) else last.toLongOption
          yield
            if start >= size || start > end then Left(())
            else Right((start, math.min(end, size - 1)))
      case _ => None
