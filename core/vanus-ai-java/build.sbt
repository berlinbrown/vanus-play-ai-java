val scala3Version = "3.9.0"

ThisBuild / organization := "org.berlin.vanus"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := scala3Version

lazy val root = project
  .in(file("."))
  .settings(
    name := "vanus-ai-java",
    javacOptions ++= Seq("--release", "21"),
    scalacOptions += "-release:21",
    Compile / run / fork := true,
    Compile / run / javaOptions += "-Xmx2g",
    libraryDependencies += "org.scalameta" %% "munit" % "1.3.6" % Test
  )
