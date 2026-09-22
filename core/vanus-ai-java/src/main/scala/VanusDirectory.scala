package org.berlin.vanus

import java.io.File
import java.net.URLEncoder
import java.util.StringTokenizer

object VanusDirectory:
  def encodeUri(uri: String): String =
    val encoded = new StringBuilder
    val tokens = new StringTokenizer(uri, "/ ", true)
    while tokens.hasMoreTokens do
      val token = tokens.nextToken()
      if token == "/" then encoded.append('/')
      else if token == " " then encoded.append("%20")
      else encoded.append(URLEncoder.encode(token, "UTF-8"))
    encoded.toString

  def findIndexFileInDirectory(directory: File, indexFileNames: java.util.List[String]): String =
    val files = indexFileNames.iterator()
    while files.hasNext do
      val fileName = files.next()
      if new File(directory, fileName).isFile then return fileName
    null

  def listDirectory(uri: String, directory: File): String =
    val heading = VanusSecurity.escapeHtml(VanusConstants.DirectoryPrefix + uri)
    val message = new StringBuilder("<html><head><title>").append(heading)
      .append("</title></head><body><h1>").append(heading).append("</h1><ul>")
    Option(directory.listFiles).toSeq.flatten.filter(file => file.isDirectory || VanusSecurity.mimeType(file.getName).isDefined).sortBy(_.getName).foreach { file =>
      val name = file.getName + (if file.isDirectory then "/" else "")
      message.append("<li><a href=\"").append(VanusSecurity.escapeHtml(encodeUri(uri + name))).append("\">").append(VanusSecurity.escapeHtml(name)).append("</a></li>")
    }
    message.append("</ul></body></html>").toString
