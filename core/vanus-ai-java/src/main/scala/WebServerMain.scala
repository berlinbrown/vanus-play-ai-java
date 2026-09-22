/**
 * Vanus Play Framework Web Server Main Class
 * Berlin Brown - 2026
 */
package org.berlin.vanus

import java.io.File
import java.util.{ArrayList, HashMap, List as JList, Map as JMap}
import java.util.concurrent.Semaphore
import java.util.concurrent.atomic.LongAdder
import org.apache.logging.log4j.LogManager
import org.nanohttpd.protocols.http.{HttpLimits, IHTTPSession, NanoHTTPD}
import org.nanohttpd.protocols.http.request.Method
import org.nanohttpd.protocols.http.response.{Response, Status}
import org.nanohttpd.webserver.{InternalRewrite, WebServerPlugin}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

object WebServerMain:
  def build(config: VanusServerConfig,
            routes: Map[(Method, String), NanoletHandler] = VanusRoutes.snapshot,
            serveFiles: Boolean = true): WebServer =
    val indices = new ArrayList[String](java.util.List.of("index.html", "index.htm"))
    val plugins = new HashMap[String, WebServerPlugin]()
    VanusPlugins.registerAvailablePlugins(plugins, indices, config.options.asJava)
    new WebServer(
      host = config.host,
      port = config.port,
      wwwroots = config.rootDirs.asJava,
      quiet = config.quiet,
      dirListing = config.dirListing,
      rateLimit = config.rateLimit.map(Integer.valueOf).orNull,
      cors = config.cors.orNull,
      maxConnections = config.maxConnections,
      maxInFlight = config.maxInFlight,
      shutdownMillis = config.shutdownMillis,
      maxTempBytes = config.maxTempBytes,
      limits = config.limits,
      plugins = plugins.asScala.toMap,
      indexNames = indices.asScala.toVector,
      routes = routes,
      serveFiles = serveFiles
    )

class WebServer(
    host: String,
    port: Int,
    wwwroots: JList[File],
    quiet: Boolean,
    dirListing: Boolean,
    rateLimit: Integer,
    cors: String,
    maxConnections: Int = 256,
    maxInFlight: Int = 32,
    shutdownMillis: Int = 10000,
    limits: HttpLimits = HttpLimits.DEFAULT,
    plugins: Map[String, WebServerPlugin] = Map.empty,
    indexNames: Vector[String] = Vector("index.html", "index.htm"),
    routes: Map[(Method, String), NanoletHandler] = VanusRoutes.snapshot,
    maxTempBytes: Long = 64L * 1024 * 1024,
    serveFiles: Boolean = true
) extends NanoHTTPD(host, port):
  require(maxInFlight > 0)
  private val rootDirs = wwwroots.asScala.map(_.getCanonicalFile).toVector
  require(rootDirs.nonEmpty && rootDirs.forall(root => root.isDirectory && root.canRead),
    "Every document root must be a readable directory")
  private val corsOption = Option(cors)
  private val rateLimiter = Option(rateLimit).map { limit =>
    new VanusRateLimiter(limit.intValue, VanusConstants.RateLimitWindowMillis)
  }
  private val connections = new VanusConnections(maxConnections, shutdownMillis)
  private val requests = new Semaphore(maxInFlight)
  private val requestCount = new LongAdder
  private val rejectedCount = new LongAdder
  private val log = LogManager.getLogger("vanus.requests")
  private val tempFiles = new VanusTempFiles(maxTempBytes, limits.maxBodyBytes(), limits.maxMultipartParts())
  setTempFileManagerFactory(tempFiles)
  setLimits(limits)
  setAsyncRunner(connections)

  def activeConnections: Int = connections.activeConnections
  def rejectedConnections: Long = connections.rejectedConnections
  def reservedTempBytes: Long = tempFiles.reservedBytes
  def totalRequests: Long = requestCount.sum()
  def rejectedRequests: Long = rejectedCount.sum()
  def activeRequests: Int = maxInFlight - requests.availablePermits()

  override def prepareResponse(response: Response): Response = VanusSecurity.protect(response)

  override def serve(session: IHTTPSession): Response =
    val started = System.nanoTime()
    requestCount.increment()
    val response =
      if !requests.tryAcquire() then
        rejectedCount.increment()
        val busy = plain(Status.SERVICE_UNAVAILABLE, "Server busy")
        busy.addHeader("Retry-After", "1")
        busy.closeConnection(true)
        busy
      else
        try
          if rateLimiter.exists(limiter => !limiter.allow(Option(session.getRemoteIpAddress).getOrElse("unknown"))) then
            rejectedCount.increment()
            val limited = plain(Status.TOO_MANY_REQUESTS, "Too Many Requests")
            limited.addHeader("Retry-After", "10")
            limited
          else respond(session.getHeaders, session, session.getUri, 0)
        catch
          case error: NanoHTTPD.ResponseException =>
            Response.newFixedLengthResponse(error.getStatus, "text/plain", error.getMessage)
          case _: java.io.EOFException => plain(Status.BAD_REQUEST, "Incomplete request body")
          case NonFatal(error) =>
            log.warn("Request failed", error)
            plain(Status.INTERNAL_ERROR, "Internal server error")
        finally requests.release()
    val finalResponse = VanusCors(response, corsOption)
    if !quiet then
      val elapsed = (System.nanoTime() - started) / 1000000L
      // Deliberately exclude query strings, credentials, headers, and filesystem paths.
      log.info("method={} status={} handler_ms={}", session.getMethod,
        finalResponse.getStatus.getRequestStatus, elapsed)
    finalResponse

  private def respond(headers: JMap[String, String], session: IHTTPSession, uri: String, depth: Int): Response =
    if depth > 8 then return plain(Status.INTERNAL_ERROR, "Too many internal rewrites")
    if uri == null || !uri.startsWith("/") || uri.startsWith("//") || uri.contains('\\') || uri.exists(_.isControl) then
      return plain(Status.BAD_REQUEST, "Invalid request path")
    if Option(session.getHeaders.get("host")).exists(value => value.isEmpty || value.contains(',') || value.exists(_.isWhitespace)) then
      return plain(Status.BAD_REQUEST, "Invalid Host header")
    val method = session.getMethod
    val routeMethod = if method == Method.HEAD then Method.GET else method
    val matching = routes.get((method, uri)).orElse(routes.get((routeMethod, uri)))
    val methods = routes.keysIterator.collect { case (verb, path) if path == uri => verb }.toSet
    val allowed = if methods.isEmpty then Set(Method.GET, Method.HEAD, Method.OPTIONS)
      else methods ++ (if methods.contains(Method.GET) then Set(Method.HEAD, Method.OPTIONS) else Set(Method.OPTIONS))
    if method == Method.OPTIONS then
      val response = plain(Status.NO_CONTENT, "")
      response.addHeader("Allow", allowed.toVector.map(_.toString).sorted.mkString(", "))
      response
    else matching match
      case Some(handler) => handler.get(session)
      case None if methods.nonEmpty || (method != Method.GET && method != Method.HEAD) =>
        val response = plain(Status.METHOD_NOT_ALLOWED, "Method not allowed")
        response.addHeader("Allow", allowed.toVector.map(_.toString).sorted.mkString(", "))
        response
      case None =>
        if !serveFiles then return plain(Status.NOT_FOUND, VanusConstants.ErrorNotFound)
        if uri.split('/').contains("..") then return plain(Status.FORBIDDEN, "Path traversal is not allowed")
        rootDirs.find(root => canServeUri(uri, root)) match
          case None => plain(Status.NOT_FOUND, VanusConstants.ErrorNotFound)
          case Some(root) => respondFromRoot(headers, session, uri, root, depth)

  private def respondFromRoot(headers: JMap[String, String], session: IHTTPSession,
      uri: String, root: File, depth: Int): Response =
    val file = resolve(root, uri)
    if file.isDirectory then
      if !uri.endsWith("/") then
        val target = VanusDirectory.encodeUri(uri + "/")
        val text = VanusSecurity.escapeHtml(target)
        val response = VanusFileResponses.fixedResponse(Status.REDIRECT, "text/html",
          s"""<html><body><a href="$text">$text</a></body></html>""")
        response.addHeader("Location", target)
        response
      else
        indexNames.find(name => !name.contains('/') && !name.contains('\\') &&
          VanusSecurity.mimeType(name).contains("text/html") && new File(file, name).isFile) match
          case Some(name) => respond(headers, session, uri + name, depth + 1)
          case None if dirListing && file.canRead =>
            VanusFileResponses.fixedResponse(Status.OK, "text/html", VanusDirectory.listDirectory(uri, file))
          case None => plain(Status.FORBIDDEN, VanusConstants.ErrorNoDirectoryListing)
    else
      VanusSecurity.mimeType(uri) match
        case None => plain(Status.FORBIDDEN, "File type is not allowed")
        case Some(mime) =>
          plugins.get(mime).filter(_.canServeUri(uri, root)) match
            case Some(plugin) =>
              plugin.serveFile(uri, headers, session, file, mime) match
                case rewrite: InternalRewrite => respond(rewrite.getHeaders, session, rewrite.getUri, depth + 1)
                case null => plain(Status.NOT_FOUND, VanusConstants.ErrorNotFound)
                case response => response
            case None => VanusFileResponses.serveFile(headers, file, mime, session.getMethod == Method.HEAD)

  private def resolve(root: File, uri: String): File = new File(root, uri.stripPrefix("/"))

  private def canServeUri(uri: String, root: File): Boolean =
    val file = resolve(root, uri)
    file.getCanonicalFile.toPath.startsWith(root.toPath) &&
      (file.isFile || file.isDirectory ||
        VanusSecurity.mimeType(uri).flatMap(plugins.get).exists(_.canServeUri(uri, root)))

  private def plain(status: Status, message: String): Response =
    Response.newFixedLengthResponse(status, NanoHTTPD.MIME_PLAINTEXT, message)
