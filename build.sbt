scalaVersion := "2.10.5"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.1"

fork := true

javaOptions in run += "-Djava.library.path=/usr/local/share/OpenCV/java"
