package org.berlin.vanus

import java.io.{ByteArrayOutputStream, File}
import java.net.{Socket, SocketException}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Files
import java.util.concurrent.{CountDownLatch, TimeUnit}
import org.nanohttpd.protocols.http.{HttpLimits, IHTTPSession}
import org.nanohttpd.protocols.http.request.Method
import org.nanohttpd.protocols.http.response.{Response, Status}
import scala.jdk.CollectionConverters.*

class ServerHardeningSuite extends munit.FunSuite:
  private case class Reply(status: Int, headers: Map[String, String], body: String)

  private def connect(server: WebServer): Socket =
    val socket = new Socket("127.0.0.1", server.getListeningPort)
    socket.setSoTimeout(3000)
    socket

  private def send(socket: Socket, request: String): Unit =
    socket.getOutputStream.write(request.getBytes(UTF_8))
    socket.getOutputStream.flush()

  private def read(socket: Socket, head: Boolean = false): Reply =
    val in = socket.getInputStream
    val bytes = new ByteArrayOutputStream()
    var complete = false
    while !complete && bytes.size() < 65536 do
      val next = in.read()
      assert(next >= 0, "connection closed before response headers")
      bytes.write(next)
      complete = bytes.toString(UTF_8).endsWith("\r\n\r\n")
    assert(complete)
    val lines = bytes.toString(UTF_8).split("\r\n")
    val status = lines.head.split(' ')(1).toInt
    val headers = lines.tail.filter(_.contains(':')).map { line =>
      val colon = line.indexOf(':')
      line.take(colon).toLowerCase -> line.drop(colon + 1).trim
    }.toMap
    val length = if head || status == 204 || status == 304 then 0 else headers.getOrElse("content-length", "0").toInt
    val body = in.readNBytes(length)
    assertEquals(body.length, length)
    Reply(status, headers, new String(body, UTF_8))

  private def get(server: WebServer, path: String, extra: String = "", method: String = "GET"): Reply =
    val socket = connect(server)
    try
      send(socket, s"$method $path HTTP/1.1\r\nHost: localhost\r\n$extra\r\n")
      read(socket, method == "HEAD")
    finally socket.close()

  private def withServer(
      limits: HttpLimits = HttpLimits.DEFAULT,
      connections: Int = 16,
      inFlight: Int = 8,
      routes: Map[(Method, String), NanoletHandler] = Map.empty,
      shutdownMillis: Int = 1000
  )(test: (WebServer, File) => Unit): Unit =
    val root = Files.createTempDirectory("vanus-hardening-")
    Files.writeString(root.resolve("index.html"), "<html><script>alert(1)</script>Hello</html>")
    Files.writeString(root.resolve("test.txt"), "0123456789")
    val server = new WebServer("127.0.0.1", 0, java.util.List.of(root.toFile), true, true, null, "*",
      connections, inFlight, shutdownMillis, limits, routes = routes)
    try
      server.start(limits.idleTimeoutMillis(), false)
      test(server, root.toFile)
    finally
      server.stop()
      val paths = Files.walk(root)
      try paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.deleteIfExists(_))
      finally paths.close()

  private def handler(f: IHTTPSession => Response): NanoletHandler = new NanoletHandler:
    override def get(session: IHTTPSession): Response = f(session)

  private def ok(text: String): Response = Response.newFixedLengthResponse(Status.OK, "text/plain", text)

  test("HTML, HEAD, redirects, errors, and listings carry the no-script policy") {
    withServer() { (server, root) =>
      Files.createDirectory(root.toPath.resolve("docs"))
      Files.writeString(root.toPath.resolve("docs/<unsafe>.txt"), "text")
      for (path, method) <- Seq(("/index.html", "GET"), ("/index.html", "HEAD"),
          ("/docs", "GET"), ("/docs/", "GET"), ("/missing", "GET")) do
        val response = get(server, path, method = method)
        assert(response.headers("content-security-policy").contains("script-src 'none'"))
        assertEquals(response.headers("x-content-type-options"), "nosniff")
      val listing = get(server, "/docs/")
      assert(listing.body.contains("&lt;unsafe&gt;.txt"))
      assert(!listing.body.contains("<unsafe>"))
    }
  }

  test("only allowlisted file extensions are served, case insensitively") {
    withServer() { (server, root) =>
      for extension <- Seq("js", "mjs", "svg", "json", "xml", "exe") do
        Files.writeString(root.toPath.resolve(s"blocked.$extension"), "blocked")
        assertEquals(get(server, s"/blocked.$extension").status, 403)
      for extension <- VanusSecurity.contentTypes.keys.toVector :+ "HTML" do
        Files.writeString(root.toPath.resolve(s"allowed.$extension"), "allowed")
        assertEquals(get(server, s"/allowed.$extension").status, 200)
      assert(!get(server, "/").body.contains("blocked.js"))
    }
  }

  test("suffix, bounded, unsatisfiable and malformed ranges are handled consistently") {
    withServer() { (server, _) =>
      val suffix = get(server, "/test.txt", "Range: bytes=-3\r\n")
      assertEquals((suffix.status, suffix.body), (206, "789"))
      assertEquals(suffix.headers("content-range"), "bytes 7-9/10")
      assertEquals(get(server, "/test.txt", "Range: bytes=7-999\r\n").body, "789")
      assertEquals(get(server, "/test.txt", "Range: bytes=99-\r\n").status, 416)
      assertEquals(get(server, "/test.txt", "Range: bytes=nonsense\r\n").status, 200)
      val etag = get(server, "/test.txt").headers("etag")
      val cached = get(server, "/test.txt", s"If-None-Match: $etag\r\n")
      assertEquals(cached.status, 304)
      assert(cached.headers.contains("content-security-policy"))
      assertEquals(get(server, "/test.txt", s"Range: bytes=1-2\r\nIf-Range: $etag\r\n").status, 200)
    }
  }

  test("HEAD has no body and does not corrupt the next keep-alive response") {
    withServer() { (server, _) =>
      val socket = connect(server)
      try
        send(socket, "HEAD /test.txt HTTP/1.1\r\nHost: localhost\r\n\r\nGET /test.txt HTTP/1.1\r\nHost: localhost\r\n\r\n")
        val head = read(socket, true)
        assertEquals(head.headers("content-length"), "10")
        assertEquals(read(socket).body, "0123456789")
      finally socket.close()
    }
  }

  test("unsupported methods with unread bodies close rather than execute the next request") {
    withServer() { (server, _) =>
      val socket = connect(server)
      try
        send(socket, "POST /test.txt HTTP/1.1\r\nHost: localhost\r\nContent-Length: 4\r\n\r\njunkGET /test.txt HTTP/1.1\r\nHost: localhost\r\n\r\n")
        val response = read(socket)
        assertEquals(response.status, 405)
        assertEquals(response.headers("connection"), "close")
        assertEquals(socket.getInputStream.read(), -1)
      finally socket.close()
    }
  }

  test("invalid framing and oversized bodies are rejected before routing") {
    withServer() { (server, _) =>
      val cases = Seq(
        "Content-Length: 1\r\nContent-Length: 1\r\n" -> 400,
        "Content-Length: -1\r\n" -> 400,
        "Content-Length: nope\r\n" -> 400,
        "Content-Length: 9999999999999999999999\r\n" -> 400,
        "Content-Length: 1048577\r\n" -> 413,
        "Transfer-Encoding: chunked\r\n" -> 501,
        "Transfer-Encoding: chunked\r\nContent-Length: 1\r\n" -> 400,
        "Expect: 100-continue\r\n" -> 417
      )
      for (headers, expected) <- cases do
        val response = get(server, "/test.txt", headers)
        assertEquals(response.status, expected)
        assertEquals(response.headers("connection"), "close")
        assert(response.headers("content-security-policy").contains("script-src 'none'"))
      assertEquals(get(server, "/test.txt", "X-Large: " + "x" * 8200 + "\r\n").status, 431)
    }
  }

  test("parsed POST bodies preserve subsequent requests and execute on virtual threads") {
    val route = handler { session =>
      assert(Thread.currentThread().isVirtual)
      val files = new java.util.HashMap[String, String]()
      session.parseBody(files)
      ok(files.getOrDefault("postData", "empty"))
    }
    withServer(routes = Map((Method.POST, "/echo") -> route)) { (server, _) =>
      val socket = connect(server)
      try
        send(socket, "POST /echo HTTP/1.1\r\nHost: localhost\r\nContent-Type: text/plain\r\nContent-Length: 4\r\n\r\ntestGET /test.txt HTTP/1.1\r\nHost: localhost\r\n\r\n")
        assertEquals(read(socket).body, "test")
        assertEquals(read(socket).body, "0123456789")
      finally socket.close()
      assertEquals(get(server, "/echo").status, 405)
    }
  }

  test("a body cannot be parsed again after its raw stream has been accessed") {
    val route = handler { session =>
      session.getInputStream.read()
      session.parseBody(new java.util.HashMap[String, String]())
      ok("unreachable")
    }
    withServer(routes = Map((Method.POST, "/body") -> route)) { (server, _) =>
      val response = get(
        server,
        "/body",
        "Content-Type: text/plain\r\nContent-Length: 1\r\n\r\nx",
        method = "POST"
      )
      assertEquals(response.status, 400)
    }
  }

  test("a full connection budget rejects connections and recovers after a client closes") {
    withServer(connections = 1) { (server, _) =>
      val first = connect(server)
      send(first, "GET /test.txt HTTP/1.1\r\nHost: localhost\r\n\r\n")
      assertEquals(read(first).status, 200)
      val second = connect(server)
      try assertEquals(second.getInputStream.read(), -1)
      finally second.close()
      first.close()
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2)
      while server.activeConnections != 0 && System.nanoTime() < deadline do Thread.sleep(10)
      assertEquals(server.activeConnections, 0)
      assertEquals(get(server, "/test.txt").status, 200)
      assert(server.rejectedConnections >= 1)
    }
  }

  test("busy handlers return 503 and graceful shutdown lets admitted work finish") {
    val entered = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val route = handler { _ =>
      entered.countDown()
      assert(release.await(5, TimeUnit.SECONDS))
      ok("finished")
    }
    withServer(inFlight = 1, routes = Map((Method.GET, "/slow") -> route)) { (server, _) =>
      val socket = connect(server)
      try
        send(socket, "GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n")
        assert(entered.await(2, TimeUnit.SECONDS))
        val delayedRelease = Thread.ofPlatform().start(() =>
          Thread.sleep(500)
          release.countDown()
        )
        assertEquals(get(server, "/test.txt").status, 503)
        val stopping = Thread.ofPlatform().start(() => server.stop())
        assertEquals(read(socket).body, "finished")
        delayedRelease.join(2000)
        stopping.join(2000)
        assert(!stopping.isAlive)
      finally
        release.countDown()
        socket.close()
    }
  }

  test("absolute header and handler deadlines close sockets") {
    val limits = new HttpLimits(8192, 1024, 4, 10, 1000, 150, 150, 150, 150)
    val route = handler { _ => Thread.sleep(5000); ok("too late") }
    withServer(limits = limits, routes = Map((Method.GET, "/slow") -> route)) { (server, _) =>
      val socket = connect(server)
      try
        send(socket, "GET /test.txt HTTP/1.1\r\nX-Slow: ")
        Thread.sleep(300)
        assertEquals(socket.getInputStream.read(), -1)
      finally socket.close()
      val slow = connect(server)
      try
        send(slow, "GET /slow HTTP/1.1\r\nHost: localhost\r\n\r\n")
        assertEquals(slow.getInputStream.read(), -1)
      finally slow.close()
    }
  }

  test("rate limiter expires clients, caps state, and never overflows a blocked count") {
    val limiter = new VanusRateLimiter(1, 1000, 2)
    assert(limiter.allow("a", 0))
    assert(limiter.allow("b", 0))
    assert(!limiter.allow("c", 0))
    for _ <- 1 to 100 do assert(!limiter.allow("a", 1))
    assert(limiter.allow("c", 1000))
    assertEquals(limiter.trackedClients, 1)
  }

  test("configuration rejects invalid values and accepts explicit resource budgets") {
    for args <- Seq(Array("--port", "0"), Array("--port", "65536"), Array("--port", "abc"),
        Array("--max-connections", "-1"), Array("--port"), Array("--unknown")) do
      intercept[IllegalArgumentException](VanusServerConfig.fromArgs(args))
    val config = VanusServerConfig.fromArgs(Array("--max-connections", "12", "--header-timeout-ms", "250"))
    assertEquals(config.maxConnections, 12)
    assertEquals(config.limits.headerTimeoutMillis(), 250)
  }

  test("response headers reject line injection and replace names case insensitively") {
    val response = ok("test")
    intercept[IllegalArgumentException](response.addHeader("X-Test", "value\r\nInjected: yes"))
    response.addHeader("Content-Security-Policy", "script-src 'unsafe-inline'")
    response.addHeader("content-security-policy", "script-src 'none'")
    assertEquals(response.getHeader("Content-Security-Policy"), "script-src 'none'")
  }
