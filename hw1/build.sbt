name := "hw1"

version := "0.1-SNAPSHOT"

organization := "sp2014.11-731"

scalaVersion := "2.10.2"

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.0", // % "test",
  "com.beust" % "jcommander" % "1.30",
  "org.scalanlp" %% "breeze-math" % "0.4",
  "com.jsuereth" %% "scala-arm" % "1.3"
)

mainClass := Some("hw1.RunEM")
