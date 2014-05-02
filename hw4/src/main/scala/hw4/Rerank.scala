package hw4

import breeze.linalg.DenseVector
import java.io.File
import RichFile.enrichFile
import breeze.optimize.{StochasticGradientDescent, StochasticDiffFunction, BatchDiffFunction, AdaptiveGradientDescent}
import scala.collection.parallel.ParSeq

/**
 * Contains a map from label to index, and from index to label
 * @param labels the set of labels to index
 * @tparam T the type of each label
 */
class Alphabet[T](private val labels: TraversableOnce[T]) {
  /** map from index to label */
  lazy val labelOf: Vector[T] = labels.toSet.toVector
  /** map from label to index */
  lazy val indexOf: Map[T, Int] = Map() ++ labelOf.zipWithIndex
  /** the number of unique labels **/
  lazy val size: Int = labelOf.size

  override def toString = labelOf.mkString(", ")
}

trait Model[-T] {
  def score(input: T): Double
}

case class Hypothesis(words: Array[String], features: Array[(String, Double)])

case class RerankerModel(vocab: Alphabet[String], params: DenseVector[Double]) extends Model[Hypothesis] {
  val size = params.size

  def values(input: Hypothesis): DenseVector[Double] = {
    val vs = DenseVector.zeros[Double](vocab.size)
    input.features.foreach({ case (k, v) => vs(vocab.indexOf(k)) = v})
    vs
  }

  override def score(input: Hypothesis): Double = values(input) dot params

  def instantRunoff(hypotheses: Seq[Hypothesis]) = {
    var remaining = hypotheses.toVector
    while (remaining.size > 1) {
      val beating = for (a <- remaining) yield {
        for (
          b <- remaining;
          diff: DenseVector[Double] = values(a) - values(b);
          score = params dot diff
          if score > 0
        ) yield 1
      }
      remaining = (remaining zip beating).sortBy(_._2.size).tail.map(_._1)
    }
    remaining.head
  }
}

case class RerankerDiffFunction(vocab: Alphabet[String], data: IndexedSeq[Seq[(Hypothesis, Double)]])
    extends BatchDiffFunction[DenseVector[Double]] {

  import RerankerModel.HYPOTHESES_PER_SENTENCE

  private[this] val numPairsPerSentence = HYPOTHESES_PER_SENTENCE * HYPOTHESES_PER_SENTENCE

  def sumPair(a: (Double, DenseVector[Double]), b: (Double, DenseVector[Double])) = (a._1 + b._1, a._2 + b._2)

  override def calculate(x: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {
//    println(batch)
    val m = RerankerModel(vocab, x)
    lazy val zeros = DenseVector.zeros[Double](m.size)

    val obsAndGrads: ParSeq[(Double, DenseVector[Double])] = batch.par.map(fromIdx).map { case ((a, aScore), (b, bScore)) =>
      val diff: DenseVector[Double] = m.values(a) - m.values(b)
      val score = m.params dot diff
      if (((score >= 0) && (aScore < bScore)) || (score < 0 && aScore > bScore)) {
//        println("miss", a, b, score, aScore, bScore)
        (1.0, diff) // perceptron loss
      } else {
//        println("correct", a, b, score, aScore, bScore)
        (0.0, zeros)
      }
    }
    val n = 1.0 / batch.size
    val (obj, grad) = obsAndGrads.reduce(sumPair)
    println("objective: %s   gradnorm: %s".format(obj * n, grad.norm(2) * n))
    (obj, grad)
  }

  private def fromIdx(i: Int) = {
    val sentId = i / numPairsPerSentence
    val sentHyps = data(sentId)
    val remainder = i - numPairsPerSentence * sentId
    val a = remainder / HYPOTHESES_PER_SENTENCE
    val b = remainder % HYPOTHESES_PER_SENTENCE
    (sentHyps(a), sentHyps(b))
  }

  override val fullRange = 0 until numPairsPerSentence * data.size
}

object RerankerModel {
  val HYPOTHESES_PER_SENTENCE = 100
  lazy val defaultVocab = new Alphabet(Set("p(e)", "p(e|f)", "p_lex(f|e)"))
  lazy val default = RerankerModel(defaultVocab, DenseVector(1f, .5f, .5f))
//  private val adaGrad = new AdaptiveGradientDescent.L2Regularization[DenseVector[Double]](0.0, 1.0, 100, 1e-8, 1e-8)
  private val sgd = StochasticGradientDescent[DenseVector[Double]]()


  def loadHypotheses(file: File): Iterator[Seq[Hypothesis]] = {
    val lines = file.lines filter { _.nonEmpty } map { line =>
      val Array(_, hypothesis, featureStr) = line.split(""" \|\|\| """)
      val features = featureStr.split(" ") map { col =>
        val Array(feature, value) =  col.split("=")
        feature -> value.toDouble
      }
      Hypothesis(hypothesis.split(" "), features)
    }
    lines.grouped(HYPOTHESES_PER_SENTENCE)
  }

  def loadScores(file: File): Iterator[Seq[Double]] = file.lines.map(_.toDouble).grouped(HYPOTHESES_PER_SENTENCE)

  def train(diffFn: StochasticDiffFunction[DenseVector[Double]]): DenseVector[Double] = {
//    adaGrad.minimize(diffFn, DenseVector.zeros(3))
    sgd.minimize(diffFn, DenseVector.zeros[Double](3))
  }
}

object Train extends App {
  import RerankerModel._

  val hypothesisFile = new File("data/dev.100best")
  val scoresFile = new File("data/dev.all.scores")

  val data = for ((hs, ss) <- loadHypotheses(hypothesisFile) zip loadScores(scoresFile)) yield { hs zip ss }

  val newModel = train(RerankerDiffFunction(defaultVocab, data.toVector).withRandomBatches(1000000))
  println(defaultVocab)
  println(newModel)
}

object Rerank extends App {
  import RerankerModel._
  val model = RerankerModel(defaultVocab, DenseVector(86201.45706984798, -14638.535604530765, -34832.58844835954))
//  val hypothesisFile = new File("data/dev.100best")
//  val outputFile = new File("dev.reranked")
  val hypothesisFile = new File("data/test.100best")
  val outputFile = new File("test.reranked")
  val hypotheses = loadHypotheses(hypothesisFile)
  outputFile.writeLines(hypotheses map { h => model.instantRunoff(h).words.mkString(" ") })
}
