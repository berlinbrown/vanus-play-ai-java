package org.berlin.vanus

import java.nio.file.{Files, Path}
import java.sql.{Connection, DriverManager}
import java.time.Instant
import scala.collection.mutable.ArrayBuffer

final case class VanusMessage(id: Long, message: String, timestamp: String, role: String)

/** Small thread-safe SQLite boundary for the autonomous conversation history. */
final class VanusMessageStore(directory: Path):
  require(directory != null, "Database directory is required")
  Files.createDirectories(directory)
  require(Files.isDirectory(directory) && Files.isWritable(directory),
    s"Database directory must be writable: $directory")

  val databasePath: Path = directory.resolve("vanus.db").toAbsolutePath.normalize()
  private val url = s"jdbc:sqlite:$databasePath"
  Class.forName("org.sqlite.JDBC")
  initialize()

  private def connection(): Connection =
    val conn = DriverManager.getConnection(url)
    val statement = conn.createStatement()
    try
      statement.execute("PRAGMA busy_timeout = 5000")
      statement.execute("PRAGMA foreign_keys = ON")
    finally statement.close()
    conn

  private def initialize(): Unit =
    val conn = connection()
    try
      val statement = conn.createStatement()
      try
        statement.execute("PRAGMA journal_mode = WAL")
        statement.execute(
          """CREATE TABLE IF NOT EXISTS messages (
            |  id INTEGER PRIMARY KEY AUTOINCREMENT,
            |  message TEXT NOT NULL,
            |  timestamp TEXT NOT NULL,
            |  role TEXT NOT NULL CHECK (role IN ('prompt', 'response'))
            |)""".stripMargin)
        statement.execute("CREATE INDEX IF NOT EXISTS messages_timestamp_idx ON messages(timestamp)")
      finally statement.close()
    finally conn.close()

  def add(message: String, role: String): Long = synchronized {
    require(message != null && message.nonEmpty, "Message cannot be empty")
    require(role == "prompt" || role == "response", "Role must be prompt or response")
    val conn = connection()
    try
      conn.setAutoCommit(false)
      val insert = conn.prepareStatement(
        "INSERT INTO messages(message, timestamp, role) VALUES (?, ?, ?)",
        java.sql.Statement.RETURN_GENERATED_KEYS)
      val id = try
        insert.setString(1, message)
        insert.setString(2, Instant.now().toString)
        insert.setString(3, role)
        insert.executeUpdate()
        val keys = insert.getGeneratedKeys
        try
          require(keys.next(), "SQLite did not return a message id")
          keys.getLong(1)
        finally keys.close()
      finally insert.close()
      val prune = conn.prepareStatement(
        "DELETE FROM messages WHERE id NOT IN (SELECT id FROM messages ORDER BY id DESC LIMIT 1000)")
      try prune.executeUpdate()
      finally prune.close()
      conn.commit()
      id
    catch
      case error: Throwable =>
        try conn.rollback()
        catch case _: Throwable => ()
        throw error
    finally conn.close()
  }

  def count(): Long =
    val conn = connection()
    try
      val statement = conn.createStatement()
      try
        val rows = statement.executeQuery("SELECT COUNT(*) FROM messages")
        try
          rows.next()
          rows.getLong(1)
        finally rows.close()
      finally statement.close()
    finally conn.close()

  def recent(limit: Int = 140): Vector[VanusMessage] =
    require(limit > 0 && limit <= 140, "Message limit must be between 1 and 140")
    val conn = connection()
    try
      val statement = conn.prepareStatement(
        "SELECT id, message, timestamp, role FROM messages ORDER BY id DESC LIMIT ?")
      try
        statement.setInt(1, limit)
        val rows = statement.executeQuery()
        val result = ArrayBuffer.empty[VanusMessage]
        try
          while rows.next() do result += VanusMessage(
            rows.getLong("id"), rows.getString("message"), rows.getString("timestamp"), rows.getString("role"))
        finally rows.close()
        result.toVector
      finally statement.close()
    finally conn.close()
