package org.berlin.vanus

import org.nanohttpd.protocols.http.request.Method

/** Registrations are copied into each server before it starts accepting connections. */
object VanusRoutes:
  val systemInfoPath = "/_vanus-ops-manage/_vanus-system-info"
  private var registered: Map[(Method, String), NanoletHandler] =
    Map((Method.GET, systemInfoPath) -> new GeneralHandler())

  def addRoute(path: String, handler: NanoletHandler): Unit = addRoute(Method.GET, path, handler)

  def addRoute(method: Method, path: String, handler: NanoletHandler): Unit = synchronized {
    require(method != null && path != null && path.trim.startsWith("/") && handler != null)
    registered = registered.updated((method, path.trim), handler)
  }

  def snapshot: Map[(Method, String), NanoletHandler] = synchronized { registered }

  /** The headless AI server deliberately exposes only its two JSON operations. */
  def bot(store: VanusMessageStore): Map[(Method, String), NanoletHandler] = VanusBotRoutes.routes(store)

  def find(path: String): Option[NanoletHandler] = snapshot.get((Method.GET, path))
