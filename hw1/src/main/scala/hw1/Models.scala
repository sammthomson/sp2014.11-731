package hw1

import scala.math._
import java.io.File
import RichFile.enrichFile


trait Model[-T] {
  def score(input: T): Float
}

trait AlignmentModel extends Model[(Int, Int, Int, Int)]

object UniformAlignmentModel extends AlignmentModel {
  import LogProbRing._

  def score(x: (Int, Int, Int, Int)): Float = (one - log(x._3) - log(x._4)).toFloat // uniform probability
}

case class DiagonalAlignmentModel(penalty: Float) extends AlignmentModel{
  def score(input: (Int, Int, Int, Int)): Float = input match { case (i, j, m, n) =>
    - penalty * math.abs((i.toFloat / m) - (j.toFloat / n))
  }
}

trait WordTranslationModel extends Model[(String, String)] {
  def save(file: File)
}

case class PhraseTable(probs: collection.Map[String, collection.Map[String,Float]], smoothing: Float) extends WordTranslationModel {
  def score(input: (String, String)): Float = probs.getOrElse(input._1, Map[String, Float]()).getOrElse(input._2, smoothing)

  def save(file: File) {
    val lines = probs//.toSeq.sortBy(-_._2)
      .view.flatMap { case (src, vals) => vals.map {
        case (target, score) => "%s\t%s\t%s".format(src, target, score)
    }}
    file.writeLines(lines)
  }
}
object PhraseTable {
  def load(file: File) = {
    val lines = file.lines
    val probs = lines.map(_.split("\t")).toSeq.groupBy(_(0)).mapValues { _.map({
      case Array(src, target, score) => target -> score.toFloat
    }).toMap}
    PhraseTable(probs, -20f)
  }
}
