package org.berlin.vanus

import java.util.Locale
import org.nanohttpd.protocols.http.response.{Response, Status}

object VanusSecurity:
  val contentTypes: Map[String, String] = Map(
    "html" -> "text/html", "htm" -> "text/html", "css" -> "text/css", "txt" -> "text/plain",
    "png" -> "image/png", "jpg" -> "image/jpeg", "jpeg" -> "image/jpeg",
    "gif" -> "image/gif", "ico" -> "image/x-icon", "webp" -> "image/webp"
  )
  // JSON is allowed for registered REST handlers, while static .json files remain blocked.
  private val allowedResponseMimeTypes = contentTypes.values.toSet + "application/json"

  def mimeType(path: String): Option[String] =
    val name = path.substring(path.lastIndexOf('/') + 1)
    val dot = name.lastIndexOf('.')
    if dot < 0 then None else contentTypes.get(name.substring(dot + 1).toLowerCase(Locale.ROOT))

  def protect(response: Response): Response =
    val mime = Option(response.getMimeType).getOrElse("").takeWhile(_ != ';').trim.toLowerCase(Locale.ROOT)
    val safe =
      if allowedResponseMimeTypes.contains(mime) && !response.getMimeType.exists(_.isControl) then response
      else
        response.close()
        Response.newFixedLengthResponse(Status.FORBIDDEN, "text/plain", "Response type is not allowed")
    safe.addHeader("Content-Type", safe.getMimeType)
    safe.addHeader("X-Content-Type-Options", "nosniff")
    safe.addHeader("Content-Security-Policy", VanusConstants.ContentSecurityPolicyValue)
    safe

  def escapeHtml(value: String): String =
    value.flatMap {
      case '&' => "&amp;"
      case '<' => "&lt;"
      case '>' => "&gt;"
      case '"' => "&quot;"
      case '\'' => "&#39;"
      case c => c.toString
    }
