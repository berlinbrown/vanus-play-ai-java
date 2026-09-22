package org.berlin.vanus

import java.io.File
import java.nio.file.Files
import java.util.{HashMap, Map as JMap}

import org.nanohttpd.protocols.http.response.{Response, Status}

class MySuite extends munit.FunSuite:

  test("example test that succeeds") {
    assertEquals(42, 42)
  }

  test("server config defaults to local serving options") {
    val config = VanusServerConfig.fromArgs(Array.empty)

    assertEquals(config.port, 8080)
    assertEquals(config.host, "127.0.0.1")
    assertEquals(config.quiet, false)
    assertEquals(config.dirListing, false)
    assertEquals(config.cors, None)
    assertEquals(config.rootDirs.size, 1)
    assert(config.options.get("port").contains("8080"))
    assert(config.options.get("quiet").contains("false"))
    assert(config.options.get("dir-listing").contains("false"))
    assert(config.options.get("home").exists(_.nonEmpty))
  }

  test("server config parses cli arguments") {
    val config = VanusServerConfig.fromArgs(Array(
      "-p", "9090",
      "-h", "127.0.0.1",
      "-q",
      "--cors=https://example.com",
      "-d", "/tmp/site-a",
      "--dir", "/tmp/site-b"
    ))

    assertEquals(config.port, 9090)
    assertEquals(config.host, "127.0.0.1")
    assertEquals(config.quiet, true)
    assertEquals(config.cors, Some("https://example.com"))
    assertEquals(config.rootDirs.map(_.getAbsolutePath), Vector(new File("/tmp/site-a").getAbsolutePath, new File("/tmp/site-b").getAbsolutePath))
    assertEquals(config.options.get("host"), Some("127.0.0.1"))
  }

  test("server config supports cors wildcard shorthand") {
    val config = VanusServerConfig.fromArgs(Array("--cors"))

    assertEquals(config.cors, Some("*"))
  }

  test("server config enables directory listing with flag") {
    val config = VanusServerConfig.fromArgs(Array("--dir-listing"))

    assertEquals(config.dirListing, true)
    assertEquals(config.options.get("dir-listing"), Some("true"))
  }

  test("server config parses rate limit") {
    val config = VanusServerConfig.fromArgs(Array("--rate-limit", "50"))

    assertEquals(config.rateLimit, Some(50))
    assertEquals(config.options.get("rate-limit"), Some("50"))
  }

  test("server config has no rate limit by default") {
    val config = VanusServerConfig.fromArgs(Array.empty)

    assertEquals(config.rateLimit, None)
  }

  test("rate limiter allows up to the limit then blocks within the window") {
    val limiter = new VanusRateLimiter(2, 1000L)
    val now = 100000L

    assertEquals(limiter.allow("client-a", now), true)
    assertEquals(limiter.allow("client-a", now + 10), true)
    assertEquals(limiter.allow("client-a", now + 20), false)
    assertEquals(limiter.allow("client-b", now + 30), true)
  }

  test("rate limiter resets after the window") {
    val limiter = new VanusRateLimiter(1, 1000L)
    val now = 100000L

    assertEquals(limiter.allow("client-a", now), true)
    assertEquals(limiter.allow("client-a", now + 100), false)
    assertEquals(limiter.allow("client-a", now + 1001), true)
  }

  test("routes can be registered and looked up") {
    val handler = new NanoletHandler:
      override def get(session: org.nanohttpd.protocols.http.IHTTPSession): Response =
        Response.newFixedLengthResponse(Status.OK, "text/plain", "ok")

    VanusRoutes.addRoute(" /_vanus-ops-manage/_vanus-test ", handler)

    assertEquals(VanusRoutes.find("/_vanus-ops-manage/_vanus-test"), Some(handler))
    assertEquals(VanusRoutes.find("/missing"), None)
  }

  test("directory helpers encode uri and list directory html") {
    assertEquals(VanusDirectory.encodeUri("/docs/hello world.txt"), "/docs/hello%20world.txt")

    val dir = Files.createTempDirectory("vanus-test-dir").toFile
    try
      val file = new File(dir, "alpha.txt")
      val subdir = new File(dir, "beta")
      file.createNewFile()
      subdir.mkdirs()

      val html = VanusDirectory.listDirectory("/root/", dir)

      assert(html.contains("Directory /root/"))
      assert(html.contains("alpha.txt"))
      assert(html.contains("beta/"))
      assert(html.indexOf("alpha.txt") < html.indexOf("beta/"))
    finally
      dir.listFiles().foreach(_.delete())
      dir.delete()
  }

  test("find index file in directory honors configured order") {
    val dir = Files.createTempDirectory("vanus-test-index").toFile
    try
      new File(dir, "index.htm").createNewFile()
      new File(dir, "index.html").createNewFile()

      val preferred = java.util.Arrays.asList("index.html", "index.htm")
      assertEquals(VanusDirectory.findIndexFileInDirectory(dir, preferred), "index.html")
    finally
      dir.listFiles().foreach(_.delete())
      dir.delete()
  }

  test("cors adds expected headers when enabled") {
    val response = VanusCors(Response.newFixedLengthResponse(Status.OK, "text/plain", "ok"), Some("*"))

    assertEquals(response.getHeader("access-control-allow-origin"), "*")
    assertEquals(response.getHeader("access-control-allow-credentials"), null)
    assertEquals(response.getHeader("access-control-allow-methods"), "GET, HEAD, OPTIONS")
  }

  test("cors leaves response untouched when disabled") {
    val response = VanusCors(Response.newFixedLengthResponse(Status.OK, "text/plain", "ok"), None)

    assertEquals(response.getHeader("access-control-allow-origin"), null)
  }

  test("file responses add range and etag headers") {
    val file = Files.createTempFile("vanus-file-test", ".txt").toFile
    try
      Files.writeString(file.toPath, "hello")
      val headers: JMap[String, String] = new HashMap[String, String]()
      val response = VanusFileResponses.serveFile(headers, file, "text/plain")

      assertEquals(response.getStatus, Status.OK)
      assertEquals(response.getHeader("accept-ranges"), "bytes")
      assertEquals(response.getHeader("content-length"), file.length.toString)
      assert(response.getHeader("etag") != null)
    finally
      file.delete()
  }

  test("file responses return not modified for matching etag") {
    val file = Files.createTempFile("vanus-file-test", ".txt").toFile
    try
      Files.writeString(file.toPath, "hello")
      val initial = VanusFileResponses.serveFile(new HashMap[String, String](), file, "text/plain")
      val etag = initial.getHeader("etag")
      initial.close()
      val headers: JMap[String, String] = new HashMap[String, String]()
      headers.put("if-none-match", etag)

      val response = VanusFileResponses.serveFile(headers, file, "text/plain")

      assertEquals(response.getStatus, Status.NOT_MODIFIED)
      assertEquals(response.getHeader("etag"), etag)
    finally
      file.delete()
  }

  test("server rejects symlink targets outside the root") {
    val root = Files.createTempDirectory("vanus-root").toFile
    val outside = Files.createTempFile("vanus-outside", ".txt").toFile
    val link = new File(root, "leak.txt")
    try
      Files.createSymbolicLink(link.toPath, outside.toPath)
      val server = new WebServer(null, 0, java.util.Arrays.asList(root), true, false, null, null)
      val field = classOf[WebServer].getDeclaredMethod("canServeUri", classOf[String], classOf[File])
      field.setAccessible(true)

      val canServeLink = field.invoke(server, "/leak.txt", root).asInstanceOf[Boolean]
      assertEquals(canServeLink, false)
    finally
      link.delete()
      outside.delete()
      root.delete()
  }
