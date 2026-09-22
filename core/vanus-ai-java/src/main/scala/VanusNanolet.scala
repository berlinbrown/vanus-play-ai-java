package org.berlin.vanus

import org.nanohttpd.protocols.http.IHTTPSession
import org.nanohttpd.protocols.http.response.{IStatus, Response, Status}

import scala.jdk.CollectionConverters.*

trait NanoletHandler:
  def get(session: IHTTPSession): Response

/**
 * A trait representing a handler for NanoHTTPD requests.
 * Implementations should provide a method to handle GET requests.
 */
abstract class DefaultHandler extends NanoletHandler:
  def getText(session: IHTTPSession): String
  def getMimeType: String
  def getStatus: IStatus
  def getCustomHeaders: Map[String, String] = Map.empty

  override def get(session: IHTTPSession): Response =
    val response = Response.newFixedLengthResponse(getStatus, getMimeType, getText(session))
    getCustomHeaders.foreach { case (name, value) =>
      response.addHeader(name, value)
    }
    response

/**
 * A general-purpose handler that responds with an HTML page displaying
 * the request URL, query parameters, and headers.
 */
class GeneralHandler extends DefaultHandler:
  override def getMimeType: String = "text/html"
  override def getStatus: IStatus = Status.OK
  override def getCustomHeaders: Map[String, String] = Map(
    VanusConstants.HeaderXServerVanusInfo -> VanusConstants.VanusVersionValue,
    VanusConstants.HeaderContentSecurityPolicy -> VanusConstants.ContentSecurityPolicyValue
  )

  override def getText(session: IHTTPSession): String =
    val lines = Vector.newBuilder[String]
    
    // Test with JavaScript, it will not execute
    lines += "<html>"
    lines += "  <body>"
    lines += s"    <h1>z data - Vanus Play Web Server</h1><br>"    
    lines += "  </body>"
    lines += "</html>"
    lines.result().mkString("\n")
