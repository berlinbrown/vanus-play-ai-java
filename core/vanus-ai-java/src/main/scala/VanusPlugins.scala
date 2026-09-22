package org.berlin.vanus

import java.util.{Map, ServiceLoader}
import org.nanohttpd.webserver.{WebServerPlugin, WebServerPluginInfo}

object VanusPlugins:
  def registerAvailablePlugins(mimeTypeHandlers: Map[String, WebServerPlugin],
      indexFileNames: java.util.List[String], options: Map[String, String]): Unit =
    ServiceLoader.load(classOf[WebServerPluginInfo]).forEach { info =>
      info.getMimeTypes.foreach { mime =>
        registerPluginForMimeType(info.getIndexFilesForMimeType(mime), mime, info.getWebServerPlugin(mime), options, mimeTypeHandlers, indexFileNames)
      }
    }

  def registerPluginForMimeType(indexFiles: Array[String], mimeType: String, plugin: WebServerPlugin,
      options: Map[String, String], mimeTypeHandlers: Map[String, WebServerPlugin],
      indexFileNames: java.util.List[String]): Unit =
    if mimeType == null || plugin == null then return

    if !VanusSecurity.contentTypes.values.toSet.contains(mimeType) then return
    if indexFiles != null then
      indexFiles.filter(filename => filename != null && !filename.contains('/') && !filename.contains('\\') &&
        VanusSecurity.mimeType(filename).contains("text/html")).foreach(indexFileNames.add)

    mimeTypeHandlers.put(mimeType, plugin)
    plugin.initialize(options)
