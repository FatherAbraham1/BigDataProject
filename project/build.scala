import sbt._
import Keys._

object BigDataProjectBuild extends Build {
  def scalaSettings = Seq(
    scalaVersion := "2.10.5",
    scalacOptions ++= Seq(
      "-optimize",
      "-unchecked",
      "-deprecation"
    )
  )

  def buildSettings =
    Project.defaultSettings ++
    scalaSettings

  lazy val root = {
    val settings = buildSettings ++ Seq(name := "TestBigData")
    Project(id = "TestBigData", base = file("."), settings = settings)
  }
}
