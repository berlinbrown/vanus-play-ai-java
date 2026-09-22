package org.berlin.vanus

import java.io.File
import org.nanohttpd.protocols.http.HttpLimits

final case class VanusServerConfig(
    host: String,
    port: Int,
    quiet: Boolean,
    dirListing: Boolean,
    rateLimit: Option[Int],
    cors: Option[String],
    rootDirs: Vector[File],
    maxConnections: Int = 256,
    maxInFlight: Int = 32,
    shutdownMillis: Int = 10000,
    limits: HttpLimits = HttpLimits.DEFAULT,
    maxTempBytes: Long = 64L * 1024 * 1024
):
  def options: Map[String, String] = Map(
    "host" -> host, "port" -> port.toString, "quiet" -> quiet.toString,
    "dir-listing" -> dirListing.toString, "rate-limit" -> rateLimit.fold("")(_.toString),
    "home" -> rootDirs.map(_.getCanonicalPath).mkString(File.pathSeparator)
  )

object VanusServerConfig:
  def fromArgs(args: Array[String]): VanusServerConfig =
    var config = VanusServerConfig("127.0.0.1", 8080, false, false, None, None, Vector.empty)
    var limits = HttpLimits.DEFAULT
    var index = 0
    def value(flag: String): String =
      index += 1
      require(index < args.length && args(index).nonEmpty, s"Missing value for $flag")
      args(index)
    def number(flag: String, maximum: Int = Int.MaxValue): Int =
      val parsed = value(flag).toIntOption
      require(parsed.exists(n => n > 0 && n <= maximum), s"Invalid value for $flag (expected 1..$maximum)")
      parsed.get
    while index < args.length do
      val flag = args(index)
      flag match
        case "p" | "-p" | "--port" => config = config.copy(port = number(flag, 65535))
        case "h" | "-h" | "--host" => config = config.copy(host = value(flag))
        case "d" | "-d" | "--dir" => config = config.copy(rootDirs = config.rootDirs :+ new File(value(flag)).getAbsoluteFile)
        case "-q" | "--quiet" => config = config.copy(quiet = true)
        case "--dir-listing" => config = config.copy(dirListing = true)
        case "--rate-limit" => config = config.copy(rateLimit = Some(number(flag)))
        case "--cors" => config = config.copy(cors = Some("*"))
        case arg if arg.startsWith("--cors=") =>
          val origin = arg.substring(7)
          require(origin.nonEmpty && !origin.exists(_.isControl), "Invalid CORS origin")
          config = config.copy(cors = Some(origin))
        case "--max-connections" => config = config.copy(maxConnections = number(flag, 100000))
        case "--max-in-flight" => config = config.copy(maxInFlight = number(flag, 100000))
        case "--shutdown-ms" => config = config.copy(shutdownMillis = number(flag))
        case "--max-temp-bytes" => config = config.copy(maxTempBytes = number(flag).toLong)
        case "--max-header-bytes" => limits = copyLimits(limits, headerBytes = number(flag, 65536))
        case "--max-body-bytes" => limits = copyLimits(limits, bodyBytes = number(flag, 16 * 1024 * 1024))
        case "--max-multipart-parts" => limits = copyLimits(limits, parts = number(flag, 1024))
        case "--max-keep-alive-requests" => limits = copyLimits(limits, requests = number(flag))
        case "--idle-timeout-ms" => limits = copyLimits(limits, idle = number(flag))
        case "--header-timeout-ms" => limits = copyLimits(limits, header = number(flag))
        case "--body-timeout-ms" => limits = copyLimits(limits, body = number(flag))
        case "--request-timeout-ms" => limits = copyLimits(limits, request = number(flag))
        case "--write-timeout-ms" => limits = copyLimits(limits, write = number(flag))
        case _ => throw new IllegalArgumentException(s"Unknown option: $flag")
      index += 1
    require(config.maxTempBytes >= 2 * limits.maxBodyBytes(), "Temporary storage budget must hold two request bodies")
    config.copy(
      rootDirs = if config.rootDirs.isEmpty then Vector(new File(".").getAbsoluteFile) else config.rootDirs,
      limits = limits
    )

  private def copyLimits(
      old: HttpLimits, headerBytes: Int = -1, bodyBytes: Int = -1, parts: Int = -1,
      requests: Int = -1, idle: Int = -1, header: Int = -1, body: Int = -1,
      request: Int = -1, write: Int = -1
  ): HttpLimits =
    def use(value: Int, previous: Int): Int = if value < 0 then previous else value
    new HttpLimits(
      use(headerBytes, old.maxHeaderBytes()),
      if bodyBytes < 0 then old.maxBodyBytes() else bodyBytes.toLong,
      use(parts, old.maxMultipartParts()), use(requests, old.maxRequestsPerConnection()),
      use(idle, old.idleTimeoutMillis()), use(header, old.headerTimeoutMillis()),
      use(body, old.bodyTimeoutMillis()), use(request, old.requestTimeoutMillis()),
      use(write, old.writeTimeoutMillis())
    )
