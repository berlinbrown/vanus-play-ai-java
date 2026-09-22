package org.berlin.vanus

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import org.nanohttpd.protocols.http.IHTTPSession
import org.nanohttpd.protocols.http.request.Method
import org.nanohttpd.protocols.http.response.{Response, Status}
import scala.jdk.CollectionConverters.*

object VanusBotRoutes:
  val statusPath = "/_vanus-ops-manage/_vanus-bot-status-info"
  val messagesPath = "/_vanus-ops-manage/_vanus-bot-messages"
  val accessToken = "dmFudXMtc2NhdHR5LW9wcy0yMDI2"

  def routes(store: VanusMessageStore): Map[(Method, String), NanoletHandler] = Map(
    (Method.GET, statusPath) -> handler(session => authenticated(session,
      s"""{"vanus":{"messagesCount":${store.count()}}}""")),
    (Method.GET, messagesPath) -> handler(session => authenticated(session, messagesJson(store.recent())))
  )

  private def handler(f: IHTTPSession => Response): NanoletHandler = new NanoletHandler:
    override def get(session: IHTTPSession): Response = f(session)

  private def authenticated(session: IHTTPSession, json: => String): Response =
    val supplied = Option(session.getParameters.get("token")).flatMap(_.asScala.headOption).getOrElse("")
    if !secureEquals(supplied, accessToken) then jsonResponse(Status.UNAUTHORIZED,
      """{"vanus":{"error":"unauthorized"}}""")
    else jsonResponse(Status.OK, json)

  private def secureEquals(left: String, right: String): Boolean =
    MessageDigest.isEqual(left.getBytes(StandardCharsets.UTF_8), right.getBytes(StandardCharsets.UTF_8))

  private def messagesJson(messages: Vector[VanusMessage]): String =
    val values = messages.map(message =>
      s"""{"message":"${escape(message.message)}","timestamp":"${escape(message.timestamp)}","id":"${message.id}","role":"${message.role}"}""")
    s"""{"vanus":{"messages":[${values.mkString(",")}]}}"""

  private def jsonResponse(status: Status, body: String): Response =
    val response = Response.newFixedLengthResponse(status, "application/json; charset=utf-8", body)
    response.addHeader("Cache-Control", "no-store")
    response

  private def escape(value: String): String =
    val result = new StringBuilder
    value.foreach {
      case '"' => result.append("\\\"")
      case '\\' => result.append("\\\\")
      case '\b' => result.append("\\b")
      case '\f' => result.append("\\f")
      case '\n' => result.append("\\n")
      case '\r' => result.append("\\r")
      case '\t' => result.append("\\t")
      case char if char < ' ' => result.append(f"\\u${char.toInt}%04x")
      case char => result.append(char)
    }
    result.result()
