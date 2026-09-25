package org.berlin.vanus

object VanusConstants:
  // Main Port setting
  val DefaultPort = 8080
  val DefaultHost = "127.0.0.1"


  // Main Argument Settings
  val ArgPort = "-p"
  val ArgPortLong = "--port"
  val ArgHost = "-h"
  val ArgHostLong = "--host"
  val ArgQuiet = "-q"
  val ArgQuietLong = "--quiet"
  val ArgDirListing = "--dir-listing"
  val ArgRateLimit = "--rate-limit"
  val ArgDir = "-d"
  val ArgDirLong = "--dir"
  val ArgCors = "--cors"

  // Additional Option Settings
  val OptionHost = "host"
  val OptionPort = "port"
  val OptionQuiet = "quiet"
  val OptionDirListing = "dir-listing"
  val OptionRateLimit = "rate-limit"
  val OptionHome = "home"

  // Rate limiting
  val RateLimitWindowMillis = 10000L

  // Default Index Files
  val DefaultIndexHtml = "index.html"
  val DefaultIndexHtm = "index.htm"

  // Header Setting Accept
  val HeaderAcceptRanges = "Accept-Ranges"
  val HeaderLocation = "Location"
  val HeaderContentLength = "Content-Length"
  val HeaderContentRange = "Content-Range"
  val HeaderETag = "ETag"
  val HeaderIfNoneMatch = "if-none-match"
  val HeaderRange = "range"

  // True server info - X-Server-Vanus-Info
  val HeaderXServerVanusInfo = "X-Server-Z-Info"
  val HeaderContentSecurityPolicy = "Content-Security-Policy"
  val HeaderAccessControlAllowOrigin = "Access-Control-Allow-Origin"
  val HeaderAccessControlAllowHeaders = "Access-Control-Allow-Headers"
  val HeaderAccessControlAllowCredentials = "Access-Control-Allow-Credentials"
  val HeaderAccessControlAllowMethods = "Access-Control-Allow-Methods"
  val HeaderAccessControlMaxAge = "Access-Control-Max-Age"

  // CORS (Cross-Origin Resource Sharing) Settings
  val CorsAllowedHeadersProperty = "AccessControlAllowHeader"
  val CorsAllowedHeadersDefault = "origin,accept,content-type"
  val CorsAllowCredentialsValue = "true"
  val CorsAllowMethodsValue = "GET, HEAD, OPTIONS"
  val CorsWildcard = "*"
  val CorsMaxAgeSeconds = 151200
 
  val ApplicationVersion = "1.0.1"
  val VanusVersionValue = s"\"vers $ApplicationVersion\""
  val ContentSecurityPolicyValue =
    "default-src 'self'; script-src 'none'; style-src 'self' 'unsafe-inline'; " +
      "object-src 'none'; frame-src 'none'; worker-src 'none'; base-uri 'none'; " +
      "form-action 'none'; frame-ancestors 'none'; sandbox allow-same-origin"

  val ErrorForbiddenTraversal = "Won't serve ../ for security reasons."
  val ErrorNoDirectoryListing = "No directory listing."
  val ErrorReadingFileFailed = "Reading file failed."
  val ErrorForbiddenPrefix = "FORBIDDEN: "
  val ErrorInternalPrefix = "INTERNAL ERROR: "
  val ErrorNotFound = "Error 404, file not found."

  val DirectoryPrefix = "Directory "
