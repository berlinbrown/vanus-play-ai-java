package org.berlin.vanus

import java.nio.file.Files
import scala.jdk.CollectionConverters.*

class VanusMessageStoreSuite extends munit.FunSuite:
  private def withStore(test: VanusMessageStore => Unit): Unit =
    val directory = Files.createTempDirectory("vanus-message-store-")
    try test(new VanusMessageStore(directory))
    finally
      val paths = Files.walk(directory)
      try paths.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.deleteIfExists(_))
      finally paths.close()

  test("message store creates SQLite database and returns newest messages") {
    withStore { store =>
      val first = store.add("hello", "prompt")
      val second = store.add("hello friend", "response")
      assert(Files.isRegularFile(store.databasePath))
      assertEquals(store.count(), 2L)
      val messages = store.recent()
      assertEquals(messages.map(_.id), Vector(second, first))
      assertEquals(messages.map(_.role), Vector("response", "prompt"))
      assert(messages.forall(_.timestamp.endsWith("Z")))
    }
  }

  test("message store retains at most one thousand rows and returns at most 140") {
    withStore { store =>
      for index <- 1 to 1005 do store.add(s"message $index", if index % 2 == 0 then "response" else "prompt")
      assertEquals(store.count(), 1000L)
      val messages = store.recent()
      assertEquals(messages.size, 140)
      assertEquals(messages.head.message, "message 1005")
      assertEquals(messages.last.message, "message 866")
    }
  }

  test("short server option aliases match documented command") {
    val directory = Files.createTempDirectory("vanus-config-")
    try
      val config = VanusServerConfig.fromArgs(Array("p", "8086", "d", directory.toString,
        "--rate-limit", "200", "h", "127.0.0.1"))
      assertEquals(config.port, 8086)
      assertEquals(config.host, "127.0.0.1")
      assertEquals(config.rateLimit, Some(200))
      assertEquals(config.rootDirs.head.getCanonicalFile, directory.toFile.getCanonicalFile)
    finally Files.deleteIfExists(directory)
  }
