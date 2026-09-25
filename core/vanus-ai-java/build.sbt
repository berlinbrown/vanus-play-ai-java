val scala3Version = "3.9.0"

ThisBuild / organization := "org.berlin.vanus"
ThisBuild / version := "1.0.1"
ThisBuild / scalaVersion := scala3Version

lazy val root = project
  .in(file("."))
  .settings(
    name := "vanus-ai-java",
    Compile / mainClass := Some("org.berlin.vanus.Main"),
    javacOptions ++= Seq("--release", "21"),
    scalacOptions += "-release:21",
    assembly / mainClass := Some("org.berlin.vanus.Main"),
    assembly / assemblyJarName := "vanus-ai-java.jar",
    assembly / test := {},
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
      case PathList("META-INF", file) if file.endsWith(".SF") || file.endsWith(".RSA") || file.endsWith(".DSA") =>
        MergeStrategy.discard
      case PathList("META-INF", "services", _*) => MergeStrategy.concat
      case PathList("META-INF", _*) => MergeStrategy.first
      case "module-info.class" => MergeStrategy.discard
      case path => (assembly / assemblyMergeStrategy).value(path)
    },
    Compile / run / fork := true,
    Compile / run / javaOptions ++= Seq("-Xmx2g", "--enable-native-access=ALL-UNNAMED"),
    libraryDependencies ++= Seq(
      "org.apache.logging.log4j" % "log4j-api" % "2.25.5",
      "org.apache.logging.log4j" % "log4j-core" % "2.25.5",
      "org.xerial" % "sqlite-jdbc" % "3.53.4.0",
      "org.apache.httpcomponents" % "httpclient" % "4.5.14" % Test,
      "org.apache.httpcomponents" % "httpmime" % "4.5.14" % Test,
      "org.scalameta" %% "munit" % "1.3.6" % Test
    )
  )
