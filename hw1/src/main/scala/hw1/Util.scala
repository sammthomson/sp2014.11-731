package hw1

import scala.io.{Codec, Source}
import java.io.{PrintWriter, File}
import resource.managed


class RichFile(file: File) {
  def text = Source.fromFile(file)(Codec.UTF8).mkString
  def text_=(s: String) = {
    for (out <- managed(new PrintWriter(file, Codec.UTF8.name))) {
      out.print(s)
    }
  }
  def lines = Source.fromFile(file)(Codec.UTF8).getLines()
  def writeLines(lines: TraversableOnce[String]) = {
    for (out <- managed(new PrintWriter(file, Codec.UTF8.name));
         line <- lines) {
      out.println(line)
    }
  }
}
object RichFile {
  implicit def enrichFile(file: File) = new RichFile(file)
}

